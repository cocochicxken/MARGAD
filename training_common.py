"""Shared losses, metrics, reproducibility, and result reporting."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


@dataclass(frozen=True)
class RunResult:
    auc: float
    auprc: float
    best_monitor_epoch: int
    best_monitor_auc: float
    best_selection_epoch: int
    best_selection_loss: float
    checkpoint: Path
    train_seconds: float
    inference_seconds: float
    total_seconds: float
    peak_gpu_allocated_mib: float
    peak_gpu_reserved_mib: float


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if requested.startswith("cuda"):
        print("CUDA is unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        import dgl

        dgl.seed(seed)
    except ImportError:
        pass


def _index_tensor(index, device: torch.device) -> torch.Tensor | None:
    if index is None:
        return None
    return torch.as_tensor(index, device=device)


def embedding_compactness(
    embeddings: torch.Tensor,
    evaluation_index=None,
    center: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = F.normalize(embeddings, p=2, dim=-1)
    if center is None:
        index = _index_tensor(evaluation_index, normalized.device)
        if index is None:
            center = normalized.mean(dim=0)
        elif index.dtype == torch.bool:
            center = normalized[index].mean(dim=0)
        else:
            center = normalized.index_select(0, index.long()).mean(dim=0)
    scores = (normalized - center).pow(2).sum(dim=-1)
    return scores, scores.mean()


def normalize_gamma_bands(bands: torch.Tensor) -> torch.Tensor:
    """L2-normalize every hidden vector in a Gamma band."""
    return F.normalize(bands, p=2, dim=-1)


def prepare_gamma_bands(
    bands: torch.Tensor, normalize_bands: bool = True,
) -> torch.Tensor:
    """Apply the dataset's explicit Gamma representation policy."""
    return normalize_gamma_bands(bands) if normalize_bands else bands


def gamma_centers(
    bands: torch.Tensor, evaluation_index=None,
) -> torch.Tensor:
    index = _index_tensor(evaluation_index, bands.device)
    if index is None:
        return bands.mean(dim=1)
    if index.dtype == torch.bool:
        return bands[:, index, :].mean(dim=1)
    return bands.index_select(1, index.long()).mean(dim=1)


def gamma_discrepancy(
    thick_bands: torch.Tensor,
    thin_bands: torch.Tensor,
    evaluation_index=None,
    thick_center: torch.Tensor | None = None,
    thin_center: torch.Tensor | None = None,
    normalize_bands: bool = True,
    center: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-response Gamma discrepancy with optional population centering."""
    thick = prepare_gamma_bands(thick_bands, normalize_bands)
    thin = prepare_gamma_bands(thin_bands, normalize_bands)
    if center:
        if thick_center is None:
            thick_center = gamma_centers(thick, evaluation_index)
        if thin_center is None:
            thin_center = gamma_centers(thin, evaluation_index)
        thick = thick - thick_center.unsqueeze(1)
        thin = thin - thin_center.unsqueeze(1)
    scores = torch.norm(thin - thick, p=2, dim=-1).mean(dim=0)
    return scores, scores.mean()


def gamma_component_names(mode: str) -> tuple[str, ...]:
    """Responses that need population centres for a Gamma ablation mode."""
    if mode in ("fixed_shared_wavelet", "learned_global_shared_only"):
        return ("thick",)
    if mode == "learned_channel_specific_only":
        return ("thin",)
    if mode in ("full", "dual_independent"):
        return ("thick", "thin")
    if mode == "uncentered_cross_response":
        return ()
    raise ValueError(f"Unsupported gamma mode: {mode}")


def gamma_components_need_centers(
    mode: str, gamma_centering: bool = True,
) -> tuple[str, ...]:
    """Responses requiring graph-level centers for the selected Gamma score."""
    if mode == "full" and not gamma_centering:
        return ()
    return gamma_component_names(mode)


def gamma_mode_centers(
    thick_bands: torch.Tensor | None,
    thin_bands: torch.Tensor | None,
    mode: str,
    evaluation_index=None,
    normalize_bands: bool = True,
) -> dict[str, torch.Tensor]:
    """Estimate exactly the centres used by a non-default Gamma score."""
    centers: dict[str, torch.Tensor] = {}
    if "thick" in gamma_component_names(mode):
        if thick_bands is None:
            raise ValueError(f"Gamma mode {mode} requires a shared response.")
        thick = prepare_gamma_bands(thick_bands, normalize_bands)
        centers["thick"] = gamma_centers(thick, evaluation_index)
    if "thin" in gamma_component_names(mode):
        if thin_bands is None:
            raise ValueError(f"Gamma mode {mode} requires a channel-specific response.")
        thin = prepare_gamma_bands(thin_bands, normalize_bands)
        centers["thin"] = gamma_centers(thin, evaluation_index)
    return centers


def _centered_gamma_deviation(
    bands: torch.Tensor,
    center: torch.Tensor | None,
    evaluation_index=None,
    normalize_bands: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prepared = prepare_gamma_bands(bands, normalize_bands)
    if center is None:
        center = gamma_centers(prepared, evaluation_index)
    score = torch.norm(prepared - center.unsqueeze(1), p=2, dim=-1).mean(dim=0)
    return score, score.mean(), center


def gamma_mode_score_loss(
    thick_bands: torch.Tensor | None,
    thin_bands: torch.Tensor | None,
    mode: str,
    evaluation_index=None,
    centers: dict[str, torch.Tensor] | None = None,
    normalize_bands: bool = True,
    gamma_centering: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Loss/score definitions for the five non-default Gamma mechanisms.

    ``full`` is supported for convenience, but the established trainers keep
    calling :func:`gamma_discrepancy` directly in their default path so the
    historical calculation remains byte-for-byte structurally unchanged.
    """
    centers = {} if centers is None else dict(centers)
    if mode == "full":
        if thick_bands is None or thin_bands is None:
            raise ValueError("full Gamma requires both responses")
        score, loss = gamma_discrepancy(
            thick_bands,
            thin_bands,
            evaluation_index,
            thick_center=centers.get("thick"),
            thin_center=centers.get("thin"),
            normalize_bands=normalize_bands,
            center=gamma_centering,
        )
        if gamma_centering and not centers:
            centers = gamma_mode_centers(
                thick_bands, thin_bands, mode, evaluation_index, normalize_bands
            )
        return score, loss, centers

    if mode in ("fixed_shared_wavelet", "learned_global_shared_only"):
        if thick_bands is None:
            raise ValueError(f"{mode} requires a shared response")
        score, loss, center = _centered_gamma_deviation(
            thick_bands,
            centers.get("thick"),
            evaluation_index,
            normalize_bands,
        )
        return score, loss, {"thick": center}

    if mode == "learned_channel_specific_only":
        if thin_bands is None:
            raise ValueError("learned_channel_specific_only requires a thin response")
        score, loss, center = _centered_gamma_deviation(
            thin_bands,
            centers.get("thin"),
            evaluation_index,
            normalize_bands,
        )
        return score, loss, {"thin": center}

    if mode == "dual_independent":
        if thick_bands is None or thin_bands is None:
            raise ValueError("dual_independent requires both responses")
        thick_score, _, thick_center = _centered_gamma_deviation(
            thick_bands, centers.get("thick"), evaluation_index, normalize_bands
        )
        thin_score, _, thin_center = _centered_gamma_deviation(
            thin_bands, centers.get("thin"), evaluation_index, normalize_bands
        )
        score = 0.5 * (thick_score + thin_score)
        return score, score.mean(), {"thick": thick_center, "thin": thin_center}

    if mode == "uncentered_cross_response":
        if thick_bands is None or thin_bands is None:
            raise ValueError("uncentered_cross_response requires both responses")
        thick = prepare_gamma_bands(thick_bands, normalize_bands)
        thin = prepare_gamma_bands(thin_bands, normalize_bands)
        score = torch.norm(thin - thick, p=2, dim=-1).mean(dim=0)
        return score, score.mean(), {}

    raise ValueError(f"Unsupported gamma mode: {mode}")


def alpha_filter_bce(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    loss_fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = positive_scores.size(0)
    labels = torch.cat(
        (
            torch.ones(count, device=positive_scores.device),
            torch.zeros(count, device=positive_scores.device),
        )
    )
    losses = loss_fn(
        torch.cat((positive_scores, negative_scores), dim=0),
        labels.unsqueeze(1).expand(-1, positive_scores.size(1)),
    ).mean(dim=0)
    return losses.mean(), losses


def minmax_numpy(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values)
    lower, upper = values.min(), values.max()
    return (values - lower) / max(float(upper - lower), eps)


def minmax_torch(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    lower, upper = values.min(), values.max()
    return (values - lower) / (upper - lower).clamp_min(eps)


def combine_numpy_scores(branches: dict[str, np.ndarray], options) -> np.ndarray:
    score = None
    for name in ("alpha", "beta", "gamma"):
        if name not in branches:
            continue
        term = minmax_numpy(branches[name]) * float(getattr(options, name))
        score = term if score is None else score + term
    if score is None:
        raise ValueError("At least one inference branch is required.")
    return score


def combine_torch_scores(branches: dict[str, torch.Tensor], options) -> torch.Tensor:
    score = None
    for name in ("alpha", "beta", "gamma"):
        if name not in branches:
            continue
        term = minmax_torch(branches[name]) * float(getattr(options, name))
        score = term if score is None else score + term
    if score is None:
        raise ValueError("At least one inference branch is required.")
    return score


def evaluation_subset(labels, scores, evaluation_index=None):
    if evaluation_index is None:
        return labels, scores
    return labels[evaluation_index], scores[evaluation_index]


def evaluate_numpy(
    labels: np.ndarray, scores: np.ndarray, evaluation_index=None,
) -> tuple[float, float]:
    labels, scores = evaluation_subset(labels, scores, evaluation_index)
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(roc_auc_score(labels, scores)), float(auc(recall, precision))


def evaluate_torch(
    labels: torch.Tensor, scores: torch.Tensor, evaluation_index=None,
) -> tuple[float, float]:
    labels_np = labels.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    index_np = None
    if evaluation_index is not None:
        index_np = evaluation_index.detach().cpu().numpy()
    return evaluate_numpy(labels_np, scores_np, index_np)


def print_monitor(epoch: int, auc_value: float) -> None:
    print(
        "BEST_MONITOR_EPOCH:", epoch,
        "BEST_MONITOR_AUC:", f"{auc_value * 100:.6f}",
        flush=True,
    )


def print_selection(epoch: int, loss_value: float) -> None:
    """Record the label-free checkpoint-selection criterion for search tools."""
    print(
        "BEST_SELECTION_EPOCH:", epoch,
        "BEST_SELECTION_LOSS:", f"{loss_value:.10f}",
        flush=True,
    )


def reset_peak_gpu_memory(device: torch.device) -> None:
    """Begin a run-level GPU-memory measurement, retaining static graph memory."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_gpu_memory_mib(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return float("nan"), float("nan")
    scale = 1024.0 * 1024.0
    return (
        torch.cuda.max_memory_allocated(device) / scale,
        torch.cuda.max_memory_reserved(device) / scale,
    )


def print_efficiency(
    run_index: int,
    train_seconds: float,
    inference_seconds: float,
    total_seconds: float,
    peak_allocated_mib: float,
    peak_reserved_mib: float,
) -> None:
    """Emit a stable, machine-readable per-run efficiency record."""
    print(
        "EFFICIENCY_RUN:", run_index,
        "TRAIN_SECONDS:", f"{train_seconds:.6f}",
        "INFERENCE_SECONDS:", f"{inference_seconds:.6f}",
        "TOTAL_SECONDS:", f"{total_seconds:.6f}",
        "PEAK_GPU_ALLOCATED_MIB:", f"{peak_allocated_mib:.3f}",
        "PEAK_GPU_RESERVED_MIB:", f"{peak_reserved_mib:.3f}",
        flush=True,
    )


def print_final_summary(results: list[RunResult]) -> None:
    auc_values = np.asarray([result.auc for result in results], dtype=np.float64)
    auprc_values = np.asarray([result.auprc for result in results], dtype=np.float64)
    print("\n==============================")
    print(auc_values.tolist())
    print(
        f"FINAL TESTING AUC:{auc_values.mean() * 100:.4f}",
        f"FINAL TESTING AUC std:{auc_values.std() * 100:.4f}",
    )
    print(f"{auc_values.mean() * 100:.2f} ({auc_values.std() * 100:.2f})")
    print(auprc_values.tolist())
    print(
        f"FINAL TESTING AUPRC:{auprc_values.mean() * 100:.4f}",
        f"FINAL TESTING AUPRC std:{auprc_values.std() * 100:.4f}",
    )
    print(f"{auprc_values.mean() * 100:.2f} ({auprc_values.std() * 100:.2f})")
    for label, values in (
        ("TRAIN_SECONDS", [result.train_seconds for result in results]),
        ("INFERENCE_SECONDS", [result.inference_seconds for result in results]),
        ("TOTAL_SECONDS", [result.total_seconds for result in results]),
        ("PEAK_GPU_ALLOCATED_MIB", [result.peak_gpu_allocated_mib for result in results]),
        ("PEAK_GPU_RESERVED_MIB", [result.peak_gpu_reserved_mib for result in results]),
    ):
        values_np = np.asarray(values, dtype=np.float64)
        print(
            f"FINAL EFFICIENCY {label}: mean={np.nanmean(values_np):.6f} "
            f"std={np.nanstd(values_np):.6f} max={np.nanmax(values_np):.6f}",
            flush=True,
        )
    print("==============================", flush=True)
