"""Small, serialisable diagnostics for the three-branch MARGAD ablations.

The training paths keep tensors on the accelerator.  This module consumes
only scalar arrays or channel aggregates, so it never asks T-Social to save a
node-by-hidden-dimension tensor merely for plotting diagnostics.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Convert numpy values recursively while retaining missing values as null."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact atomically after converting NumPy values."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _evaluation_arrays(labels, scores, degree, evaluation_index=None):
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    degree = np.asarray(degree, dtype=np.float64).reshape(-1)
    if not (labels.size == scores.size == degree.size):
        raise ValueError("labels, scores, and degree must have identical length")
    if evaluation_index is not None:
        index = np.asarray(evaluation_index)
        labels, scores, degree = labels[index], scores[index], degree[index]
    return labels, scores, degree


def _safe_binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or np.unique(labels).size < 2:
        return {"auc": None, "auprc": None}
    # Import lazily so a pure matrix dry-run does not need sklearn.
    from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

    precision, recall, _ = precision_recall_curve(labels, scores)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "auprc": float(auc(recall, precision)),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, implemented locally to keep diagnostics light."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        # Ranks are one-based; the tied group receives its mean rank.
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return None
    return _pearson(_average_ranks(x), _average_ranks(y))


def alpha_degree_diagnostics(
    labels,
    alpha_anomaly_score,
    degree,
    evaluation_index=None,
    buckets: int = 5,
) -> dict[str, Any]:
    """Correlation and equal-frequency degree buckets for alpha diagnostics."""
    labels, score, degree = _evaluation_arrays(
        labels, alpha_anomaly_score, degree, evaluation_index
    )
    log_degree = np.log1p(np.maximum(degree, 0.0))
    result: dict[str, Any] = {
        "count": int(score.size),
        "score_degree_pearson": _pearson(score, degree),
        "score_log_degree_pearson": _pearson(score, log_degree),
        "score_degree_spearman": _spearman(score, degree),
        "score_log_degree_spearman": _spearman(score, log_degree),
        "buckets": [],
    }
    if score.size == 0:
        return result

    # Quantiles can coincide on sparse/regular graphs.  Keeping deterministic
    # digitize edges still makes the empty and one-class buckets visible rather
    # than silently inventing an AUROC.
    edges = np.quantile(degree, np.linspace(0.0, 1.0, buckets + 1))
    assignments = np.searchsorted(edges[1:-1], degree, side="right")
    for bucket in range(buckets):
        mask = assignments == bucket
        bucket_labels, bucket_score, bucket_degree = labels[mask], score[mask], degree[mask]
        metrics = _safe_binary_metrics(bucket_labels, bucket_score)
        result["buckets"].append({
            "bucket": int(bucket + 1),
            "count": int(mask.sum()),
            "anomaly_count": int(np.count_nonzero(bucket_labels)),
            "degree_min": float(bucket_degree.min()) if bucket_degree.size else None,
            "degree_max": float(bucket_degree.max()) if bucket_degree.size else None,
            **metrics,
        })
    return result


def score_only_metrics(
    branches: dict[str, np.ndarray],
    weights: dict[str, float],
    labels,
    evaluation_index=None,
) -> dict[str, dict[str, float | None]]:
    """Evaluate all available non-empty fusion subsets with fresh min--maxes."""
    from training_common import minmax_numpy

    names = tuple(name for name in ("alpha", "beta", "gamma") if name in branches)
    results: dict[str, dict[str, float | None]] = {}
    for mask in range(1, 1 << len(names)):
        selected = tuple(names[index] for index in range(len(names)) if mask & (1 << index))
        fused = None
        for name in selected:
            term = minmax_numpy(np.asarray(branches[name])) * float(weights[name])
            fused = term if fused is None else fused + term
        metrics = _safe_binary_metrics(
            np.asarray(labels)[evaluation_index] if evaluation_index is not None else labels,
            np.asarray(fused)[evaluation_index] if evaluation_index is not None else fused,
        )
        results["+".join(selected)] = {**metrics, "weights": {name: weights[name] for name in selected}}
    return results


def channel_concentration(channel_energy) -> dict[str, Any]:
    """Summarise channel energy without retaining per-node energy maps."""
    energy = np.asarray(channel_energy, dtype=np.float64).reshape(-1)
    energy = np.maximum(energy, 0.0)
    total = float(energy.sum())
    if energy.size == 0 or total <= 0.0:
        return {
            "channels": int(energy.size),
            "total_energy": total,
            "top10_percent_energy_share": None,
            "normalized_hhi": None,
            "channel_energy": energy.tolist(),
        }
    proportion = energy / total
    top_count = max(1, int(math.ceil(0.10 * energy.size)))
    top_share = float(np.sort(proportion)[-top_count:].sum())
    # 0 = perfectly diffuse, 1 = all energy in one channel.
    hhi = float((proportion * proportion).sum())
    normalized_hhi = (
        (hhi - 1.0 / energy.size) / (1.0 - 1.0 / energy.size)
        if energy.size > 1 else 1.0
    )
    return {
        "channels": int(energy.size),
        "total_energy": total,
        "top10_percent_energy_share": top_share,
        "normalized_hhi": normalized_hhi,
        "channel_energy": energy.tolist(),
    }


class _RunningMoments:
    """Vector-free Welford moments used for node-level concentration values."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values) -> None:
        for value in np.asarray(values, dtype=np.float64).reshape(-1):
            if not np.isfinite(value):
                continue
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (value - self.mean)

    def as_dict(self) -> dict[str, float | int | None]:
        deviation = math.sqrt(self.m2 / (self.count - 1)) if self.count > 1 else None
        return {"count": self.count, "mean": self.mean if self.count else None, "std": deviation}


class GammaConcentrationAccumulator:
    """Streaming node-level and aggregate channel energy for sampled T-Social.

    Node-level moments prevent a misleading conclusion when every anomaly is
    concentrated in a different channel: aggregate channel energy may look
    diffuse even though each individual anomaly is highly channel-selective.
    """

    def __init__(self, channels: int):
        self.normal = np.zeros(int(channels), dtype=np.float64)
        self.anomaly = np.zeros(int(channels), dtype=np.float64)
        self.normal_count = 0
        self.anomaly_count = 0
        self.normal_top_share = _RunningMoments()
        self.anomaly_top_share = _RunningMoments()
        self.normal_hhi = _RunningMoments()
        self.anomaly_hhi = _RunningMoments()

    def update(self, channel_energy, labels) -> None:
        energy = np.asarray(channel_energy, dtype=np.float64)
        labels = np.asarray(labels).reshape(-1)
        if energy.ndim != 2 or energy.shape[0] != labels.size:
            raise ValueError("expected [nodes, channels] energy and matching labels")
        normal = labels == 0
        anomalous = ~normal
        nonnegative = np.maximum(energy, 0.0)
        total = nonnegative.sum(axis=1)
        valid = total > 0.0
        proportion = np.zeros_like(nonnegative)
        proportion[valid] = nonnegative[valid] / total[valid, None]
        top_count = max(1, int(math.ceil(0.10 * energy.shape[1])))
        top_share = np.sort(proportion, axis=1)[:, -top_count:].sum(axis=1)
        raw_hhi = (proportion * proportion).sum(axis=1)
        if energy.shape[1] == 1:
            normalized_hhi = np.ones_like(raw_hhi)
        else:
            normalized_hhi = (raw_hhi - 1.0 / energy.shape[1]) / (1.0 - 1.0 / energy.shape[1])
        if normal.any():
            self.normal += nonnegative[normal].sum(axis=0)
            self.normal_count += int(normal.sum())
            self.normal_top_share.update(top_share[normal & valid])
            self.normal_hhi.update(normalized_hhi[normal & valid])
        if anomalous.any():
            self.anomaly += nonnegative[anomalous].sum(axis=0)
            self.anomaly_count += int(anomalous.sum())
            self.anomaly_top_share.update(top_share[anomalous & valid])
            self.anomaly_hhi.update(normalized_hhi[anomalous & valid])

    def as_dict(self) -> dict[str, Any]:
        return {
            "normal_count": self.normal_count,
            "anomaly_count": self.anomaly_count,
            "normal": {
                "aggregate_channel_profile": channel_concentration(self.normal),
                "node_level_concentration": {
                    "top10_percent_energy_share": self.normal_top_share.as_dict(),
                    "normalized_hhi": self.normal_hhi.as_dict(),
                },
            },
            "anomaly": {
                "aggregate_channel_profile": channel_concentration(self.anomaly),
                "node_level_concentration": {
                    "top10_percent_energy_share": self.anomaly_top_share.as_dict(),
                    "normalized_hhi": self.anomaly_hhi.as_dict(),
                },
            },
        }
