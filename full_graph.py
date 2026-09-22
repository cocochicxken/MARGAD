"""Full-graph MARGAD training for the six datasets that fit in memory."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from Dataloader import FullGraphData
from dataset_config import DatasetSpec
from model import GAD
from training_common import (
    RunResult,
    alpha_filter_bce,
    combine_numpy_scores,
    embedding_compactness,
    evaluate_numpy,
    gamma_centers,
    gamma_discrepancy,
    gamma_mode_centers,
    gamma_mode_score_loss,
    prepare_gamma_bands,
    print_final_summary,
    print_monitor,
    print_selection,
    print_efficiency,
    peak_gpu_memory_mib,
    reset_peak_gpu_memory,
    resolve_device,
    set_seed,
    synchronize,
)
from utils import negative_sampling, normalized_laplacian_with_self_loop
from ablation_diagnostics import (
    GammaConcentrationAccumulator,
    alpha_degree_diagnostics,
    score_only_metrics,
    write_json_atomic,
)


class FullGraphTrainer:
    """Train, select, evaluate, and diagnose one full-graph dataset."""

    def __init__(self, options, spec: DatasetSpec, data: FullGraphData):
        self.options = options
        self.spec = spec
        self.device = resolve_device(options.device)
        self.features = torch.as_tensor(data.features, dtype=torch.float32, device=self.device)
        self.adjacency = data.adjacency.to(self.device)
        self.labels = np.asarray(data.labels).reshape(-1)
        self.evaluation_index = data.evaluation_index
        self.use_alpha = options.alpha != 0.0
        self.use_beta = options.beta != 0.0
        self.use_gamma = options.gamma != 0.0
        self.laplacian = (
            normalized_laplacian_with_self_loop(self.adjacency)
            if self.use_gamma else None
        )

    def _new_model(self) -> GAD:
        """Create a fresh MARGAD model for one independent run."""

        return GAD(
            feat_size=self.features.size(1),
            hidden_size=self.options.hidden_dim,
            dropout=self.options.dropout,
            alpha_mode=self.options.alpha_mode,
            gamma_mode=self.options.gamma_mode,
        ).to(self.device)

    def _local_affinity(
        self,
        model: GAD,
        embeddings: torch.Tensor,
        adjacency: torch.Tensor,
        walk_terms,
        return_filter_scores: bool = False,
    ):
        return model.local_affinity(
            embeddings,
            adjacency,
            normalize=self.spec.normalize_alpha,
            walk_terms=walk_terms,
            return_filter_scores=return_filter_scores,
        )

    def _positive_representations(self, model: GAD, walk_terms):
        embeddings = model(self.features)
        thick_bands = thin_bands = None
        if self.use_gamma:
            thick_bands, thin_bands = self._gamma_bands(model)
        alpha_score = alpha_filters = None
        if self.use_alpha:
            alpha_score, alpha_filters = self._local_affinity(
                model,
                embeddings,
                self.adjacency,
                walk_terms,
                return_filter_scores=True,
            )
        return embeddings, thick_bands, thin_bands, alpha_score, alpha_filters

    def _gamma_bands(self, model: GAD):
        gamma_embeddings = model.encode(
            self.features, apply_tanh=self.spec.gamma_input_tanh
        )
        return model.gamma_wavelet(
            gamma_embeddings,
            self.laplacian,
            squash_coefficients=self.spec.gamma_coefficient_tanh,
        )

    def _raw_branch_scores(
        self,
        model: GAD,
        embeddings: torch.Tensor,
        thick_bands: torch.Tensor | None,
        thin_bands: torch.Tensor | None,
        alpha_score: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        branches = {}
        if self.use_alpha:
            branches["alpha"] = (-alpha_score).detach().cpu().numpy()
        if self.use_beta:
            beta_score, _ = embedding_compactness(
                embeddings, self.evaluation_index
            )
            branches["beta"] = beta_score.detach().cpu().numpy()
        if self.use_gamma:
            gamma_score, _ = self._gamma_score_loss(
                model,
                thick_bands,
                thin_bands,
                evaluation_index=self.evaluation_index,
            )
            branches["gamma"] = gamma_score.detach().cpu().numpy()
        return branches

    @staticmethod
    def _tensor_list(value):
        return None if value is None else value.detach().cpu().tolist()

    def _gamma_curve_payload(self, model: GAD):
        lambdas = torch.linspace(0.0, 2.0, 41, device=self.device)
        coefficients = model.gamma_wavelet.diagnostic_coefficients(
            squash_with_tanh=self.spec.gamma_coefficient_tanh
        )
        shared_curve, channel_curve = model.gamma_wavelet.response_curves(
            lambdas, squash_with_tanh=self.spec.gamma_coefficient_tanh
        )
        payload = {
            "mode": self.options.gamma_mode,
            "gamma_centering": bool(self.options.gamma_centering),
            "lambda": lambdas.detach().cpu().tolist(),
            "shared_coefficients": self._tensor_list(coefficients["shared_coefficients"]),
            "channel_specific_coefficients": self._tensor_list(coefficients["channel_specific_coefficients"]),
            "shared_response": self._tensor_list(shared_curve),
            "channel_response_summary": None,
            "representative_channel_indices": [],
            "representative_channel_response": None,
        }
        if channel_curve is not None:
            channels = channel_curve.size(1)
            representatives = torch.linspace(
                0, max(channels - 1, 0), min(16, channels), device=self.device
            ).round().long().unique(sorted=True)
            payload["channel_response_summary"] = {
                "mean": channel_curve.mean(dim=1).detach().cpu().tolist(),
                "std": channel_curve.std(dim=1, unbiased=False).detach().cpu().tolist(),
            }
            payload["representative_channel_indices"] = representatives.detach().cpu().tolist()
            payload["representative_channel_response"] = channel_curve.index_select(
                1, representatives
            ).detach().cpu().tolist()
        return payload

    def _gamma_concentration(self, thick_bands, thin_bands):
        if thick_bands is None or thin_bands is None:
            return {"available": False, "reason": "mode_has_single_response"}
        thick = prepare_gamma_bands(thick_bands, self.spec.gamma_normalize_bands)
        thin = prepare_gamma_bands(thin_bands, self.spec.gamma_normalize_bands)
        # This diagnostic intentionally uses centered cross-response energy
        # regardless of which Gamma scoring variant was trained.
        thick_center = gamma_centers(thick, self.evaluation_index)
        thin_center = gamma_centers(thin, self.evaluation_index)
        energy = ((thin - thin_center.unsqueeze(1)) - (
            thick - thick_center.unsqueeze(1)
        )).pow(2).mean(dim=0)
        labels = self.labels
        if self.evaluation_index is not None:
            index = torch.as_tensor(self.evaluation_index, device=self.device, dtype=torch.long)
            energy = energy.index_select(0, index)
            labels = labels[self.evaluation_index]
        accumulator = GammaConcentrationAccumulator(energy.size(1))
        accumulator.update(energy.detach().cpu().numpy(), labels)
        return {"available": True, **accumulator.as_dict()}

    def _write_diagnostics(
        self,
        model: GAD,
        seed: int,
        branches,
        embeddings,
        thick_bands,
        thin_bands,
        alpha_score,
    ):
        alpha = model.alpha_wavelet.diagnostic_coefficients()
        alpha_payload = {
            "mode": alpha["mode"],
            "coefficient_normalized": alpha["coefficient_normalized"],
            "filter_coefficients": self._tensor_list(alpha["filter_coefficients"]),
            "mixing_weights": self._tensor_list(alpha["mixing_weights"]),
            "degree_diagnostics": None,
        }
        if branches is not None and "alpha" in branches:
            degree = model.alpha_wavelet._row_sum(
                model.alpha_wavelet._remove_diagonal(self.adjacency)
            ).detach().cpu().numpy()
            alpha_payload["degree_diagnostics"] = alpha_degree_diagnostics(
                self.labels, branches["alpha"], degree, self.evaluation_index
            )

        gamma_payload = self._gamma_curve_payload(model)
        gamma_payload["centered_channel_concentration"] = self._gamma_concentration(
            thick_bands, thin_bands
        ) if self.use_gamma else {"available": False, "reason": "gamma_disabled"}

        weights = {name: float(getattr(self.options, name)) for name in ("alpha", "beta", "gamma")}
        score_only = (
            score_only_metrics(branches, weights, self.labels, self.evaluation_index)
            if branches is not None and set(branches) == {"alpha", "beta", "gamma"}
            else {}
        )
        write_json_atomic(self.options.diagnostics_json, {
            "schema_version": 1,
            "dataset": self.spec.cli_name,
            "seed": seed,
            "trainer": "full_graph",
            "alpha": alpha_payload,
            "gamma": gamma_payload,
            "score_only_combinations": score_only,
            "notes": {
                "alpha_full_mode": "P=D^-1(A-diag(A)); target-return-removed two-hop context; convex learned mixing",
                "raw_volume_interpretation": "A5/A6 use exact sparse raw A/A^2 message/volume operations on this full graph.",
                "checkpoint_selection": "minimum unsupervised total training loss after the training midpoint",
            },
        })

    def _gamma_score_loss(
        self,
        model,
        thick_bands,
        thin_bands,
        evaluation_index=None,
        centers=None,
    ):
        if self.options.gamma_mode == "full":
            centers = {} if centers is None else centers
            return gamma_discrepancy(
                thick_bands,
                thin_bands,
                evaluation_index,
                thick_center=centers.get("thick"),
                thin_center=centers.get("thin"),
                normalize_bands=self.spec.gamma_normalize_bands,
                center=self.options.gamma_centering,
            )
        score, loss, _ = gamma_mode_score_loss(
            thick_bands,
            thin_bands,
            self.options.gamma_mode,
            evaluation_index=evaluation_index,
            centers=centers,
            normalize_bands=self.spec.gamma_normalize_bands,
            gamma_centering=self.options.gamma_centering,
        )
        return score, loss

    def _checkpoint_path(self, run_index: int) -> Path:
        if self.options.runs == 1:
            return Path.cwd() / "best_model.pth"
        return Path.cwd() / f"best_model_run{run_index}.pth"

    def _train_one(self, run_index: int) -> RunResult:
        """Train one seed, restore the selected checkpoint, and evaluate it."""

        seed = self.options.seed_offset + run_index
        set_seed(seed)
        print(f"\n# Run:{run_index} seed={seed}", flush=True)
        run_started = time.perf_counter()
        model = self._new_model()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.options.lr,
            weight_decay=self.options.weight_decay,
        )
        alpha_bce = nn.BCEWithLogitsLoss(reduction="none") if self.use_alpha else None
        positive_walk_terms = (
            model.alpha_wavelet.precompute_terms(self.adjacency, self.features.dtype)
            if self.use_alpha else None
        )
        reset_peak_gpu_memory(self.device)
        synchronize(self.device)
        train_started = time.perf_counter()

        best_loss = float("inf")
        best_epoch = 0
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
        best_monitor_auc = float("-inf")
        best_monitor_epoch = 0

        for epoch in range(self.options.epoch):
            model.train()
            optimizer.zero_grad()
            (
                embeddings,
                thick_bands,
                thin_bands,
                alpha_score,
                positive_filter_scores,
            ) = self._positive_representations(model, positive_walk_terms)

            zero = torch.zeros((), device=self.device)
            alpha_loss = beta_loss = gamma_loss = zero
            if self.use_alpha:
                negative_adjacency = negative_sampling(self.adjacency)
                negative_walk_terms = model.alpha_wavelet.precompute_terms(
                    negative_adjacency, embeddings.dtype, assume_no_return=True
                )
                _, negative_filter_scores = self._local_affinity(
                    model,
                    embeddings,
                    negative_adjacency,
                    negative_walk_terms,
                    return_filter_scores=True,
                )
                alpha_loss, _ = alpha_filter_bce(
                    positive_filter_scores, negative_filter_scores, alpha_bce
                )
            if self.use_beta:
                _, beta_loss = embedding_compactness(
                    embeddings, self.evaluation_index
                )
            if self.use_gamma:
                _, gamma_loss = self._gamma_score_loss(
                    model,
                    thick_bands,
                    thin_bands,
                    evaluation_index=self.evaluation_index,
                )

            alpha_term = alpha_loss * self.options.alpha
            beta_term = beta_loss * self.options.beta
            gamma_term = gamma_loss * self.options.gamma
            total_loss = alpha_term + beta_term + gamma_term

            monitor_auc = None
            if not self.options.disable_monitor_auc:
                with torch.no_grad():
                    score = combine_numpy_scores(
                        self._raw_branch_scores(
                            model,
                            embeddings,
                            thick_bands,
                            thin_bands,
                            alpha_score,
                        ),
                        self.options,
                    )
                    monitor_auc, _ = evaluate_numpy(
                        self.labels, score, self.evaluation_index
                    )
                if monitor_auc > best_monitor_auc:
                    best_monitor_auc = monitor_auc
                    best_monitor_epoch = epoch

            loss_value = total_loss.item()
            if loss_value < best_loss and epoch > self.options.epoch // 2:
                best_loss = loss_value
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1

            total_loss.backward()
            optimizer.step()
            print(
                "Epoch:", f"{epoch:04d}",
                "train_loss=", f"{loss_value:.5f}",
                "loss_alpha=", f"{alpha_term.item():.5f}",
                "loss_beta=", f"{beta_term.item():.5f}",
                "loss_gamma=", f"{gamma_term.item():.5f}",
                "gamma_centering=", "on" if self.options.gamma_centering else "off",
                "auc:", monitor_auc if monitor_auc is not None else "disabled",
                flush=True,
            )
            if wait >= self.options.patience:
                print("Early stopping!", flush=True)
                break

        if not self.options.disable_monitor_auc:
            print_monitor(best_monitor_epoch, best_monitor_auc)
        print_selection(best_epoch, best_loss)
        print(f"Loading {best_epoch}th epoch", flush=True)
        model.load_state_dict(best_state)
        checkpoint = self._checkpoint_path(run_index)
        torch.save(model.state_dict(), checkpoint)
        synchronize(self.device)
        train_seconds = time.perf_counter() - train_started

        test_scores = []
        diagnostic_branches = None
        diagnostic_embeddings = diagnostic_thick = diagnostic_thin = diagnostic_alpha = None
        best_test_auc = 0.0
        model.eval()
        inference_started = time.perf_counter()
        for test_index in range(self.options.tests):
            with torch.no_grad():
                embeddings = model(self.features)
                thick_bands = thin_bands = None
                if self.use_gamma:
                    thick_bands, thin_bands = self._gamma_bands(model)
                alpha_score = None
                if self.use_alpha:
                    alpha_score = self._local_affinity(
                        model, embeddings, self.adjacency, positive_walk_terms
                    )
                branches = self._raw_branch_scores(
                    model,
                    embeddings,
                    thick_bands,
                    thin_bands,
                    alpha_score,
                )
                score = combine_numpy_scores(branches, self.options)
                diagnostic_branches = branches
                diagnostic_embeddings = embeddings
                diagnostic_thick, diagnostic_thin = thick_bands, thin_bands
                diagnostic_alpha = alpha_score
            test_auc, _ = evaluate_numpy(
                self.labels, score, self.evaluation_index
            )
            best_test_auc = max(best_test_auc, test_auc)
            test_scores.append(score)
            print(
                "Test:", f"{test_index:04d}",
                "Auc:", test_auc,
                "Best_Auc:", best_test_auc,
                flush=True,
            )

        final_score = np.mean(np.stack(test_scores, axis=0), axis=0)
        final_auc, final_auprc = evaluate_numpy(
            self.labels, final_score, self.evaluation_index
        )
        synchronize(self.device)
        inference_seconds = time.perf_counter() - inference_started
        total_seconds = time.perf_counter() - run_started
        peak_allocated_mib, peak_reserved_mib = peak_gpu_memory_mib(self.device)
        print("auprc:", final_auprc)
        print("auc:", final_auc)
        print_efficiency(
            run_index,
            train_seconds,
            inference_seconds,
            total_seconds,
            peak_allocated_mib,
            peak_reserved_mib,
        )
        if self.options.diagnostics_json:
            self._write_diagnostics(
                model,
                seed,
                diagnostic_branches,
                diagnostic_embeddings,
                diagnostic_thick,
                diagnostic_thin,
                diagnostic_alpha,
            )
        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return RunResult(
            final_auc,
            final_auprc,
            best_monitor_epoch if not self.options.disable_monitor_auc else -1,
            best_monitor_auc if not self.options.disable_monitor_auc else float("nan"),
            best_epoch,
            best_loss,
            checkpoint,
            train_seconds,
            inference_seconds,
            total_seconds,
            peak_allocated_mib,
            peak_reserved_mib,
        )

    def run(self) -> list[RunResult]:
        """Execute all requested seeds and print their aggregate summary."""

        print(
            f"Loaded {self.spec.cli_name}: nodes={self.features.size(0)}, "
            f"features={self.features.size(1)}, edges={self.adjacency._nnz()}, "
            f"device={self.device}",
            flush=True,
        )
        print(
            "Gamma policy:"
            f" normalize_bands={self.spec.gamma_normalize_bands},"
            f" input_tanh={self.spec.gamma_input_tanh},"
            f" coefficient_tanh={self.spec.gamma_coefficient_tanh}",
            flush=True,
        )
        results = [self._train_one(run_index) for run_index in range(self.options.runs)]
        print_final_summary(results)
        return results


def run_full_graph(options, spec: DatasetSpec, data: FullGraphData):
    """Run the full-graph training path selected by the CLI dispatcher."""

    return FullGraphTrainer(options, spec, data).run()
