"""Neighbor-sampled MARGAD training and streaming inference for T-Social."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from Dataloader import LargeGraphData
from dataset_config import DatasetSpec
from model import GAD
from training_common import (
    RunResult,
    alpha_filter_bce,
    combine_torch_scores,
    embedding_compactness,
    evaluate_torch,
    gamma_centers,
    gamma_discrepancy,
    gamma_components_need_centers,
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
from ablation_diagnostics import (
    GammaConcentrationAccumulator,
    alpha_degree_diagnostics,
    score_only_metrics,
    write_json_atomic,
)


def _dgl_loader(graph, node_ids, batch_size, fanout, shuffle, num_workers):
    """Build the two-layer DGL neighbor loader used by T-Social."""

    import dgl

    sampler = (
        dgl.dataloading.MultiLayerFullNeighborSampler(2)
        if fanout <= 0
        else dgl.dataloading.NeighborSampler([fanout, fanout])
    )
    return dgl.dataloading.DataLoader(
        graph,
        node_ids,
        sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
    )


def _random_walk_return_probability(graph, device):
    """Compute the sampled TA-DiffRef two-step return probability per node."""

    src, dst = graph.edges(order="eid")
    non_self = src != dst
    src, dst = src[non_self], dst[non_self]
    count = graph.num_nodes()
    degree = torch.zeros(count, dtype=torch.float32, device=src.device)
    if not src.numel():
        return degree.to(device)
    degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=degree.dtype))
    reciprocal = graph.has_edges_between(dst, src).to(dtype=degree.dtype)
    contribution = reciprocal / (
        degree[dst].clamp_min(1.0) * degree[src].clamp_min(1.0)
    )
    return torch.zeros_like(degree).scatter_add_(0, dst, contribution).to(device)


def _block_sum(block, source_features):
    """Sum source features into the destination nodes of a DGL block."""

    import dgl.function as fn

    with block.local_scope():
        block.srcdata["h"] = source_features
        block.update_all(fn.copy_u("h", "m"), fn.sum("m", "sum"))
        return block.dstdata["sum"]


def _alpha_from_terms(
    model,
    h_out,
    one_hop,
    two_hop=None,
    return_probability=None,
    one_volume=None,
    two_volume=None,
    return_filter_scores=False,
):
    """Combine sampled diffusion terms into TA-DiffRef scores."""

    h_near, per_filter_near = model.alpha_wavelet.combine_sampled_terms(
        h_out,
        one_hop,
        two_hop,
        return_probability=return_probability,
        one_volume=one_volume,
        two_volume=two_volume,
        return_filters=True,
    )
    score = (h_out * h_near).sum(dim=-1)
    if not return_filter_scores:
        return score
    filters = (h_out.unsqueeze(0) * per_filter_near).sum(dim=-1).transpose(0, 1)
    return score, filters


def _negative_alpha(
    model,
    feature_store,
    h_out,
    output_nodes,
    num_nodes,
    fanout,
    device,
):
    """Construct non-target corrupted references for sampled affinity loss."""

    fanout = max(int(fanout), 1)
    batch_size, hidden_size = h_out.shape
    feature_device = feature_store.device
    targets = output_nodes.to(feature_device, dtype=torch.long).reshape(-1)

    def random_non_target(shape, target_shape):
        sampled = torch.randint(0, num_nodes, shape, device=feature_device)
        expanded = targets.reshape(target_shape).expand_as(sampled)
        collision = sampled == expanded
        while collision.any():
            sampled[collision] = torch.randint(
                0, num_nodes, (int(collision.sum().item()),), device=feature_device
            )
            collision = sampled == expanded
        return sampled

    first_ids = random_non_target((batch_size, fanout), (batch_size, 1))
    second_ids = random_non_target(
        (batch_size, fanout, fanout), (batch_size, 1, 1)
    )
    first_features = feature_store[first_ids.reshape(-1)].to(device, non_blocking=True)
    second_features = feature_store[second_ids.reshape(-1)].to(device, non_blocking=True)
    first_h = F.normalize(
        torch.tanh(model.lin(first_features)).reshape(batch_size, fanout, hidden_size),
        p=2,
        dim=-1,
    )
    second_h = F.normalize(
        torch.tanh(model.lin(second_features)).reshape(
            batch_size, fanout, fanout, hidden_size
        ),
        p=2,
        dim=-1,
    )
    if model.alpha_wavelet.uses_raw_path_volume:
        one_hop = first_h.sum(dim=1)
        two_hop = second_h.sum(dim=(1, 2))
        one_volume = torch.full(
            (batch_size,), float(fanout), device=device, dtype=h_out.dtype
        )
        two_volume = torch.full(
            (batch_size,), float(fanout * fanout), device=device, dtype=h_out.dtype
        )
    else:
        one_hop = first_h.mean(dim=1)
        two_hop = second_h.mean(dim=(1, 2))
        one_volume = two_volume = None
    # Random non-target draws have no target-return path by construction.
    return_probability = torch.zeros(batch_size, device=device, dtype=h_out.dtype)
    return _alpha_from_terms(
        model,
        h_out,
        one_hop,
        two_hop,
        return_probability=return_probability,
        one_volume=one_volume,
        two_volume=two_volume,
        return_filter_scores=True,
    )[1]


def _normalized_adjacency_from_block(
    block, source_features, destination_features, source_degree, destination_degree,
):
    """Apply normalized adjacency with self-loops to one sampled block."""

    source = source_features / (source_degree + 1.0).clamp_min(1e-12).sqrt().unsqueeze(-1)
    neighbours = _block_sum(block, source)
    neighbours = neighbours / (
        (destination_degree + 1.0).clamp_min(1e-12).sqrt().unsqueeze(-1)
    )
    self_term = destination_features / (
        destination_degree + 1.0
    ).clamp_min(1e-12).unsqueeze(-1)
    return neighbours + self_term


def _gamma_bands(model, h, lh, l2h, squash_coefficients=True):
    """Evaluate shared and channel-specific WaveShift polynomial responses."""

    wavelet = model.gamma_wavelet
    thick_outputs, thin_outputs = [], []
    thick_coefficients, thin_coefficients = wavelet.effective_coefficients(
        squash_with_tanh=squash_coefficients
    )
    for filter_id in range(3):
        if thick_coefficients is not None:
            thick = thick_coefficients[filter_id]
            thick_outputs.append(thick[0] * l2h + thick[1] * lh + thick[2] * h)
        if thin_coefficients is not None:
            thin = thin_coefficients[filter_id]
            thin_outputs.append(
                l2h * thin[:, 0] + lh * thin[:, 1] + h * thin[:, 2]
            )
    thick = torch.stack(thick_outputs, dim=0) if thick_outputs else None
    thin = torch.stack(thin_outputs, dim=0) if thin_outputs else None
    return thick, thin


class TSocialTrainer:
    """Train MARGAD on T-Social with sampled batches and streaming statistics."""

    def __init__(self, options, spec: DatasetSpec, data: LargeGraphData):
        self.options = options
        self.spec = spec
        self.device = resolve_device(options.device)
        self.graph = data.graph
        self.feature_store = data.features
        self.labels = data.labels.long()
        self.evaluation_index = data.evaluation_index
        self.graph_device = torch.device("cpu")
        self._place_graph_and_features()
        self.num_nodes = self.graph.num_nodes()
        self.node_ids = torch.arange(
            self.num_nodes, dtype=torch.long, device=self.graph_device
        )
        self.degree = self.graph.in_degrees().float().to(self.device, non_blocking=True)
        self.return_probability = _random_walk_return_probability(
            self.graph, self.device
        )
        self.labels = self.labels.to(self.device)
        if self.evaluation_index is not None:
            self.evaluation_index = self.evaluation_index.to(self.device)
        self.use_alpha = options.alpha != 0.0
        self.use_beta = options.beta != 0.0
        self.use_gamma = options.gamma != 0.0

    def _place_graph_and_features(self):
        if (
            self.options.dgl_graph_on_gpu
            and self.device.type == "cuda"
            and self.options.num_workers == 0
        ):
            try:
                self.graph = self.graph.to(self.device)
                self.graph_device = self.device
                self.feature_store = self.graph.ndata["feature"].float()
                print(f"Moved the T-Social sampling graph to {self.device}.")
            except RuntimeError as error:
                torch.cuda.empty_cache()
                print(f"Could not move T-Social graph to GPU; using CPU. Reason: {error}")
        elif self.options.dgl_graph_on_gpu and self.options.num_workers != 0:
            print("Keeping T-Social graph on CPU because GPU sampling requires --num_workers 0.")

        if self.device.type == "cuda" and self.feature_store.device.type != "cuda":
            try:
                self.feature_store = self.feature_store.to(self.device, non_blocking=True)
                print("Keeping the T-Social feature table on GPU.")
            except RuntimeError as error:
                torch.cuda.empty_cache()
                print(f"Could not move T-Social features to GPU; using CPU. Reason: {error}")

    def _loader(self, batch_size: int, shuffle: bool):
        return _dgl_loader(
            self.graph,
            self.node_ids,
            batch_size,
            self.options.batch_fanout,
            shuffle,
            self.options.num_workers,
        )

    def _representations(self, model, input_nodes, blocks):
        import dgl

        blocks = [block.to(self.device) for block in blocks]
        input_features = self.feature_store[
            input_nodes.to(self.feature_store.device)
        ].to(self.device, non_blocking=True)
        h_input = torch.tanh(model.lin(input_features))
        num_mid = blocks[0].num_dst_nodes()
        num_out = blocks[1].num_dst_nodes()
        h_mid = h_input[:num_mid]
        h_out = h_mid[:num_out]
        result = {
            "h": h_out,
        }
        if self.use_alpha:
            h_alpha_input = (
                F.normalize(h_input, p=2, dim=-1)
                if self.spec.normalize_alpha else h_input
            )
            h_alpha_mid = h_alpha_input[:num_mid]
            h_alpha_out = h_alpha_mid[:num_out]
            result["h_norm"] = h_alpha_out

            if model.alpha_wavelet.uses_raw_path_volume:
                one_hop = _block_sum(blocks[1], h_alpha_mid)
                raw_mid = _block_sum(blocks[0], h_alpha_input)
                two_hop = _block_sum(blocks[1], raw_mid)
                one_volume = blocks[1].in_degrees().to(
                    device=self.device, dtype=h_out.dtype
                )
                mid_volume = blocks[0].in_degrees().to(
                    device=self.device, dtype=h_out.dtype
                )
                two_volume = _block_sum(blocks[1], mid_volume.unsqueeze(-1)).squeeze(-1)
                alpha_score, alpha_filters = _alpha_from_terms(
                    model,
                    h_alpha_out,
                    one_hop,
                    two_hop,
                    one_volume=one_volume,
                    two_volume=two_volume,
                    return_filter_scores=True,
                )
            else:
                sampled_mid_degree = blocks[0].in_degrees().to(
                    device=self.device, dtype=h_out.dtype
                )
                sampled_out_degree = blocks[1].in_degrees().to(
                    device=self.device, dtype=h_out.dtype
                )
                p1_mid = _block_sum(blocks[0], h_alpha_input) / sampled_mid_degree.clamp_min(
                    model.alpha_wavelet.eps
                ).unsqueeze(-1)
                one_hop = _block_sum(blocks[1], h_alpha_mid) / sampled_out_degree.clamp_min(
                    model.alpha_wavelet.eps
                ).unsqueeze(-1)
                two_hop = _block_sum(blocks[1], p1_mid) / sampled_out_degree.clamp_min(
                    model.alpha_wavelet.eps
                ).unsqueeze(-1)
                output_ids = blocks[1].dstdata[dgl.NID].to(
                    self.return_probability.device, dtype=torch.long
                )
                return_probability = self.return_probability.index_select(0, output_ids).to(
                    self.device, dtype=h_out.dtype, non_blocking=True
                )
                alpha_score, alpha_filters = _alpha_from_terms(
                    model,
                    h_alpha_out,
                    one_hop,
                    two_hop,
                    return_probability=return_probability,
                    return_filter_scores=True,
                )
            result["alpha"] = alpha_score
            result["alpha_filters"] = alpha_filters
        if not self.use_gamma:
            return result

        gamma_h_input = (
            h_input
            if self.spec.gamma_input_tanh
            else model.encode(input_features, apply_tanh=False)
        )
        gamma_h_mid = gamma_h_input[:num_mid]
        gamma_h_out = gamma_h_mid[:num_out]

        def gather_degree(node_ids):
            return self.degree.index_select(
                0, node_ids.detach().to(self.degree.device, dtype=torch.long)
            ).to(self.device, dtype=h_out.dtype, non_blocking=True)

        degree_input = gather_degree(blocks[0].srcdata[dgl.NID])
        degree_mid = gather_degree(blocks[0].dstdata[dgl.NID])
        degree_out = gather_degree(blocks[1].dstdata[dgl.NID])
        adjacency_h_mid = _normalized_adjacency_from_block(
            blocks[0], gamma_h_input, gamma_h_mid, degree_input, degree_mid
        )
        lh_mid = gamma_h_mid - adjacency_h_mid
        adjacency_h_out = _normalized_adjacency_from_block(
            blocks[1], gamma_h_mid, gamma_h_out, degree_mid, degree_out
        )
        lh_out = gamma_h_out - adjacency_h_out
        adjacency_lh_out = _normalized_adjacency_from_block(
            blocks[1], lh_mid, lh_out, degree_mid, degree_out
        )
        l2h_out = lh_out - adjacency_lh_out
        result["gamma_thick"], result["gamma_thin"] = _gamma_bands(
            model,
            gamma_h_out,
            lh_out,
            l2h_out,
            squash_coefficients=self.spec.gamma_coefficient_tanh,
        )
        return result

    def _estimate_centers(self, model) -> dict[str, torch.Tensor]:
        """Estimate graph-level branch centers without storing all embeddings."""

        if not (self.use_beta or self.use_gamma):
            return {}
        sums: dict[str, torch.Tensor] = {}
        count = 0
        model.eval()
        with torch.no_grad():
            for input_nodes, output_nodes, blocks in self._loader(
                self.options.eval_batch_size, False
            ):
                reps = self._representations(model, input_nodes, blocks)
                count += output_nodes.numel()
                if self.use_beta:
                    values = F.normalize(reps["h"], p=2, dim=-1).sum(dim=0)
                    sums["beta"] = sums.get(
                        "beta", torch.zeros_like(values)
                    ) + values
                if self.use_gamma:
                    required = gamma_components_need_centers(
                        self.options.gamma_mode, self.options.gamma_centering
                    )
                    if "thick" in required:
                        thick = prepare_gamma_bands(
                            reps["gamma_thick"], self.spec.gamma_normalize_bands
                        ).sum(dim=1)
                        sums["gamma_thick"] = sums.get(
                            "gamma_thick", torch.zeros_like(thick)
                        ) + thick
                    if "thin" in required:
                        thin = prepare_gamma_bands(
                            reps["gamma_thin"], self.spec.gamma_normalize_bands
                        ).sum(dim=1)
                        sums["gamma_thin"] = sums.get(
                            "gamma_thin", torch.zeros_like(thin)
                        ) + thin
        return {name: value / max(count, 1) for name, value in sums.items()}

    def _collect_scores(
        self,
        model,
        centers,
        return_branches: bool = False,
        diagnostic_cross_centers=None,
    ):
        branches = {}
        if self.use_alpha:
            branches["alpha"] = torch.empty(self.num_nodes, device=self.device)
        if self.use_beta:
            branches["beta"] = torch.empty(self.num_nodes, device=self.device)
        if self.use_gamma:
            branches["gamma"] = torch.empty(self.num_nodes, device=self.device)
        concentration = None
        if diagnostic_cross_centers is not None:
            concentration = GammaConcentrationAccumulator(self.options.hidden_dim)
        model.eval()
        with torch.no_grad():
            for input_nodes, output_nodes, blocks in self._loader(
                self.options.eval_batch_size, False
            ):
                reps = self._representations(model, input_nodes, blocks)
                index = output_nodes.to(self.device, dtype=torch.long)
                if self.use_alpha:
                    branches["alpha"][index] = -reps["alpha"]
                if self.use_beta:
                    beta_score, _ = embedding_compactness(
                        reps["h"], center=centers["beta"]
                    )
                    branches["beta"][index] = beta_score
                if self.use_gamma:
                    gamma_score, _ = self._gamma_score_loss(
                        model,
                        reps["gamma_thick"],
                        reps["gamma_thin"],
                        centers=centers,
                    )
                    branches["gamma"][index] = gamma_score
                    if (
                        concentration is not None
                        and reps["gamma_thick"] is not None
                        and reps["gamma_thin"] is not None
                    ):
                        thick = prepare_gamma_bands(
                            reps["gamma_thick"], self.spec.gamma_normalize_bands
                        )
                        thin = prepare_gamma_bands(
                            reps["gamma_thin"], self.spec.gamma_normalize_bands
                        )
                        energy = ((thin - diagnostic_cross_centers["thin"].unsqueeze(1)) - (
                            thick - diagnostic_cross_centers["thick"].unsqueeze(1)
                        )).pow(2).mean(dim=0)
                        concentration.update(
                            energy.detach().cpu().numpy(),
                            self.labels.index_select(0, index).detach().cpu().numpy(),
                        )
        score = combine_torch_scores(branches, self.options)
        if return_branches:
            return score, branches, concentration
        return score

    def _gamma_score_loss(
        self,
        model,
        thick_bands,
        thin_bands,
        centers=None,
    ):
        centers = {} if centers is None else centers
        mode_centers = {
            "thick": centers.get("gamma_thick"),
            "thin": centers.get("gamma_thin"),
        }
        if self.options.gamma_mode == "full":
            return gamma_discrepancy(
                thick_bands,
                thin_bands,
                thick_center=mode_centers["thick"],
                thin_center=mode_centers["thin"],
                normalize_bands=self.spec.gamma_normalize_bands,
                center=self.options.gamma_centering,
            )
        score, loss, _ = gamma_mode_score_loss(
            thick_bands,
            thin_bands,
            self.options.gamma_mode,
            centers={key: value for key, value in mode_centers.items() if value is not None},
            normalize_bands=self.spec.gamma_normalize_bands,
            gamma_centering=self.options.gamma_centering,
        )
        return score, loss

    def _estimate_cross_response_centers(self, model):
        """Streaming centres for the optional centered cross-response diagnostic."""
        if not self.use_gamma:
            return None
        sums = {}
        count = 0
        model.eval()
        with torch.no_grad():
            for input_nodes, output_nodes, blocks in self._loader(
                self.options.eval_batch_size, False
            ):
                reps = self._representations(model, input_nodes, blocks)
                if reps["gamma_thick"] is None or reps["gamma_thin"] is None:
                    return None
                thick = prepare_gamma_bands(
                    reps["gamma_thick"], self.spec.gamma_normalize_bands
                ).sum(dim=1)
                thin = prepare_gamma_bands(
                    reps["gamma_thin"], self.spec.gamma_normalize_bands
                ).sum(dim=1)
                sums["thick"] = sums.get("thick", torch.zeros_like(thick)) + thick
                sums["thin"] = sums.get("thin", torch.zeros_like(thin)) + thin
                count += output_nodes.numel()
        return {name: value / max(count, 1) for name, value in sums.items()}

    @staticmethod
    def _tensor_list(value):
        return None if value is None else value.detach().cpu().tolist()

    def _gamma_curve_payload(self, model):
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

    def _write_diagnostics(self, model, seed, branches, concentration):
        alpha = model.alpha_wavelet.diagnostic_coefficients()
        alpha_payload = {
            "mode": alpha["mode"],
            "coefficient_normalized": alpha["coefficient_normalized"],
            "filter_coefficients": self._tensor_list(alpha["filter_coefficients"]),
            "mixing_weights": self._tensor_list(alpha["mixing_weights"]),
            "degree_diagnostics": None,
        }
        branch_numpy = {
            name: value.detach().cpu().numpy() for name, value in branches.items()
        }
        labels = self.labels.detach().cpu().numpy()
        evaluation_index = (
            self.evaluation_index.detach().cpu().numpy()
            if self.evaluation_index is not None else None
        )
        if "alpha" in branch_numpy:
            alpha_payload["degree_diagnostics"] = alpha_degree_diagnostics(
                labels,
                branch_numpy["alpha"],
                self.degree.detach().cpu().numpy(),
                evaluation_index,
            )
        weights = {name: float(getattr(self.options, name)) for name in ("alpha", "beta", "gamma")}
        score_only = (
            score_only_metrics(branch_numpy, weights, labels, evaluation_index)
            if set(branch_numpy) == {"alpha", "beta", "gamma"} else {}
        )
        write_json_atomic(self.options.diagnostics_json, {
            "schema_version": 1,
            "dataset": self.spec.cli_name,
            "seed": seed,
            "trainer": "large_graph_sampled",
            "alpha": alpha_payload,
            "gamma": {
                **self._gamma_curve_payload(model),
                "centered_channel_concentration": (
                    {"available": True, **concentration.as_dict()}
                    if concentration is not None
                    else {"available": False, "reason": "mode_has_single_response_or_gamma_disabled"}
                ),
            },
            "score_only_combinations": score_only,
            "notes": {
                "alpha_full_mode": "sampled random-walk analogue of P=D^-1(A-diag(A)) with target-return removal",
                "raw_volume_interpretation": "A5/A6 use fanout-sampled raw message/path-volume analogues; they are not exact full-graph A^2 volumes.",
                "streaming_diagnostics": "Gamma concentration is reduced per batch; no N-by-hidden tensor is saved.",
                "checkpoint_selection": "minimum unsupervised total training loss after the training midpoint",
            },
        })

    def _checkpoint_path(self, run_index: int) -> Path:
        if self.options.runs == 1:
            return Path.cwd() / "best_model.pth"
        return Path.cwd() / f"best_model_run{run_index}.pth"

    def _train_one(self, run_index: int) -> RunResult:
        """Train and evaluate one T-Social seed using the sampled pipeline."""

        seed = self.options.seed_offset + run_index
        set_seed(seed)
        print(f"\n# Run:{run_index} seed={seed}", flush=True)
        run_started = time.perf_counter()
        model = GAD(
            self.feature_store.size(1),
            self.options.hidden_dim,
            self.options.dropout,
            alpha_mode=self.options.alpha_mode,
            gamma_mode=self.options.gamma_mode,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.options.lr,
            weight_decay=self.options.weight_decay,
        )
        alpha_bce = nn.BCEWithLogitsLoss(reduction="none") if self.use_alpha else None
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
            branches = None
            if not self.options.disable_monitor_auc:
                branches = {}
                if self.use_alpha:
                    branches["alpha"] = torch.empty(self.num_nodes, device=self.device)
                if self.use_beta:
                    branches["beta"] = torch.empty(self.num_nodes, device=self.device)
                if self.use_gamma:
                    branches["gamma"] = torch.empty(self.num_nodes, device=self.device)
            loss_sums = {"total": 0.0, "alpha": 0.0, "beta": 0.0, "gamma": 0.0}
            seen = 0

            for input_nodes, output_nodes, blocks in self._loader(
                self.options.batch_size, True
            ):
                optimizer.zero_grad()
                reps = self._representations(model, input_nodes, blocks)
                zero = torch.zeros((), device=self.device)
                alpha_loss = beta_loss = gamma_loss = zero
                if self.use_alpha:
                    negative_filters = _negative_alpha(
                        model,
                        self.feature_store,
                        reps["h_norm"],
                        output_nodes,
                        self.num_nodes,
                        self.options.batch_fanout,
                        self.device,
                    )
                    alpha_loss, _ = alpha_filter_bce(
                        reps["alpha_filters"], negative_filters, alpha_bce
                    )
                if self.use_beta:
                    beta_score, beta_loss = embedding_compactness(reps["h"])
                if self.use_gamma:
                    gamma_score, gamma_loss = self._gamma_score_loss(
                        model,
                        reps["gamma_thick"],
                        reps["gamma_thin"],
                    )
                alpha_term = alpha_loss * self.options.alpha
                beta_term = beta_loss * self.options.beta
                gamma_term = gamma_loss * self.options.gamma
                total = alpha_term + beta_term + gamma_term
                total.backward()
                optimizer.step()

                if branches is not None:
                    index = output_nodes.to(self.device, dtype=torch.long)
                    if self.use_alpha:
                        branches["alpha"][index] = -reps["alpha"].detach()
                    if self.use_beta:
                        branches["beta"][index] = beta_score.detach()
                    if self.use_gamma:
                        branches["gamma"][index] = gamma_score.detach()
                batch_count = output_nodes.numel()
                seen += batch_count
                for name, value in (
                    ("total", total),
                    ("alpha", alpha_term),
                    ("beta", beta_term),
                    ("gamma", gamma_term),
                ):
                    loss_sums[name] += value.item() * batch_count

            averages = {
                name: value / max(seen, 1) for name, value in loss_sums.items()
            }
            monitor_auc = None
            if branches is not None:
                monitor_score = combine_torch_scores(branches, self.options)
                monitor_auc, _ = evaluate_torch(
                    self.labels, monitor_score, self.evaluation_index
                )
                if monitor_auc > best_monitor_auc:
                    best_monitor_auc = monitor_auc
                    best_monitor_epoch = epoch
            if averages["total"] < best_loss and epoch > self.options.epoch // 2:
                best_loss = averages["total"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
            print(
                "Epoch:", f"{epoch:04d}",
                "train_loss=", f"{averages['total']:.5f}",
                "loss_alpha=", f"{averages['alpha']:.5f}",
                "loss_beta=", f"{averages['beta']:.5f}",
                "loss_gamma=", f"{averages['gamma']:.5f}",
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
        diagnostic_branches = diagnostic_concentration = None
        best_test_auc = 0.0
        inference_started = time.perf_counter()
        for test_index in range(self.options.tests):
            centers = self._estimate_centers(model)
            if self.options.diagnostics_json and test_index == self.options.tests - 1:
                cross_centers = self._estimate_cross_response_centers(model)
                score, diagnostic_branches, diagnostic_concentration = self._collect_scores(
                    model,
                    centers,
                    return_branches=True,
                    diagnostic_cross_centers=cross_centers,
                )
            else:
                score = self._collect_scores(model, centers)
            test_auc, _ = evaluate_torch(
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
        final_score = torch.stack(test_scores, dim=0).mean(dim=0)
        final_auc, final_auprc = evaluate_torch(
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
                model, seed, diagnostic_branches, diagnostic_concentration
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
        """Execute all requested T-Social seeds and print aggregate metrics."""

        print(
            f"Loaded T-Social: nodes={self.num_nodes}, "
            f"features={self.feature_store.size(1)}, edges={self.graph.num_edges()}, "
            f"batch_size={self.options.batch_size}, "
            f"eval_batch_size={self.options.eval_batch_size}, "
            f"fanout={self.options.batch_fanout}, sampling_device={self.graph_device}",
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


def run_large_graph(options, spec: DatasetSpec, data: LargeGraphData):
    """Run the T-Social path selected by the CLI dispatcher."""

    return TSocialTrainer(options, spec, data).run()
