"""Aggregate completed MARGAD ablation seeds into CSV and JSON summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _as_float(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(statistics.fmean(values)), float(statistics.pstdev(values))


def _completed_results(root: Path):
    results = []
    run_root = root / "runs"
    if not run_root.is_dir():
        return results
    for path in sorted(run_root.rglob("result.json")):
        completed = path.with_name("completed.json")
        if not completed.is_file():
            continue
        try:
            result = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("status") == "completed":
            results.append((path, result))
    return results


RUN_FIELDS = [
    "study", "variant", "dataset", "dataset_key", "seed", "active_losses",
    "alpha_mode", "gamma_mode", "alpha_weight", "beta_weight", "gamma_weight",
    "epoch_override", "auc", "auprc", "best_monitor_epoch", "best_monitor_auc",
    "best_selection_epoch", "best_selection_loss", "train_seconds", "inference_seconds",
    "total_seconds", "peak_gpu_allocated_mib", "peak_gpu_reserved_mib", "checkpoint",
    "result_path", "diagnostics_path",
]


def _run_row(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics", {})
    weights = result.get("weights", {})
    return {
        "study": result.get("study"),
        "variant": result.get("variant"),
        "dataset": result.get("dataset"),
        "dataset_key": result.get("dataset_key"),
        "seed": result.get("seed"),
        "active_losses": "+".join(result.get("active_losses", [])),
        "alpha_mode": result.get("alpha_mode"),
        "gamma_mode": result.get("gamma_mode"),
        "alpha_weight": weights.get("alpha"),
        "beta_weight": weights.get("beta"),
        "gamma_weight": weights.get("gamma"),
        "epoch_override": result.get("epoch_override"),
        "auc": metrics.get("auc"),
        "auprc": metrics.get("auprc"),
        "best_monitor_epoch": metrics.get("best_monitor_epoch"),
        "best_monitor_auc": metrics.get("best_monitor_auc"),
        "best_selection_epoch": metrics.get("best_selection_epoch"),
        "best_selection_loss": metrics.get("best_selection_loss"),
        "train_seconds": metrics.get("train_seconds"),
        "inference_seconds": metrics.get("inference_seconds"),
        "total_seconds": metrics.get("total_seconds"),
        "peak_gpu_allocated_mib": metrics.get("peak_gpu_allocated_mib"),
        "peak_gpu_reserved_mib": metrics.get("peak_gpu_reserved_mib"),
        "checkpoint": metrics.get("checkpoint"),
        "result_path": str(path),
        "diagnostics_path": str(path.with_name(result.get("diagnostics_file", "diagnostics.json"))),
    }


SUMMARY_FIELDS = [
    "study", "variant", "dataset", "n", "auc_mean", "auc_std", "auprc_mean", "auprc_std",
    "selection_loss_mean", "selection_loss_std", "train_seconds_mean", "total_seconds_mean",
]


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["study"], row["variant"], row["dataset"])].append(row)
    summaries = []
    for (study, variant, dataset), values in sorted(grouped.items()):
        auc_mean, auc_std = _mean_std([value for row in values if (value := _as_float(row["auc"])) is not None])
        pr_mean, pr_std = _mean_std([value for row in values if (value := _as_float(row["auprc"])) is not None])
        loss_mean, loss_std = _mean_std([value for row in values if (value := _as_float(row["best_selection_loss"])) is not None])
        train_mean, _ = _mean_std([value for row in values if (value := _as_float(row["train_seconds"])) is not None])
        total_mean, _ = _mean_std([value for row in values if (value := _as_float(row["total_seconds"])) is not None])
        summaries.append({
            "study": study, "variant": variant, "dataset": dataset, "n": len(values),
            "auc_mean": auc_mean, "auc_std": auc_std,
            "auprc_mean": pr_mean, "auprc_std": pr_std,
            "selection_loss_mean": loss_mean, "selection_loss_std": loss_std,
            "train_seconds_mean": train_mean, "total_seconds_mean": total_mean,
        })
    return summaries


SCORE_FIELDS = [
    "study", "variant", "dataset", "seed", "combination", "auc", "auprc", "weights", "result_path",
]
SCORE_SUMMARY_FIELDS = [
    "study", "variant", "dataset", "combination", "n", "auc_mean", "auc_std", "auprc_mean", "auprc_std",
]


def _score_rows(path: Path, result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for combination, metrics in result.get("score_only_combinations", {}).items():
        rows.append({
            "study": result.get("study"), "variant": result.get("variant"),
            "dataset": result.get("dataset"), "seed": result.get("seed"),
            "combination": combination, "auc": metrics.get("auc"), "auprc": metrics.get("auprc"),
            "weights": json.dumps(metrics.get("weights", {}), ensure_ascii=False, sort_keys=True),
            "result_path": str(path),
        })
    return rows


def _score_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["study"], row["variant"], row["dataset"], row["combination"])].append(row)
    result = []
    for key, values in sorted(grouped.items()):
        auc_mean, auc_std = _mean_std([value for row in values if (value := _as_float(row["auc"])) is not None])
        pr_mean, pr_std = _mean_std([value for row in values if (value := _as_float(row["auprc"])) is not None])
        result.append({
            "study": key[0], "variant": key[1], "dataset": key[2], "combination": key[3],
            "n": len(values), "auc_mean": auc_mean, "auc_std": auc_std,
            "auprc_mean": pr_mean, "auprc_std": pr_std,
        })
    return result


DIAGNOSTIC_FIELDS = [
    "study", "variant", "dataset", "seed", "alpha_degree_pearson", "alpha_log_degree_pearson",
    "alpha_degree_spearman", "alpha_log_degree_spearman", "gamma_normal_top10_mean",
    "gamma_anomaly_top10_mean", "gamma_normal_hhi_mean", "gamma_anomaly_hhi_mean", "diagnostics_path",
]

DEGREE_BUCKET_FIELDS = [
    "study", "variant", "dataset", "seed", "bucket", "count", "anomaly_count",
    "degree_min", "degree_max", "auc", "auprc", "diagnostics_path",
]
DEGREE_BUCKET_SUMMARY_FIELDS = [
    "study", "variant", "dataset", "bucket", "n", "count_mean", "anomaly_count_mean",
    "auc_mean", "auc_std", "auprc_mean", "auprc_std",
]


def _nested(value: dict[str, Any], *keys, default=None):
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return value if value is not None else default


def _diagnostic_row(path: Path, result: dict[str, Any]) -> dict[str, Any] | None:
    diagnostic_path = path.with_name(result.get("diagnostics_file", "diagnostics.json"))
    if not diagnostic_path.is_file():
        return None
    try:
        data = _read_json(diagnostic_path)
    except (OSError, json.JSONDecodeError):
        return None
    degree = _nested(data, "alpha", "degree_diagnostics", default={})
    concentration = _nested(data, "gamma", "centered_channel_concentration", default={})
    return {
        "study": result.get("study"), "variant": result.get("variant"),
        "dataset": result.get("dataset"), "seed": result.get("seed"),
        "alpha_degree_pearson": _nested(degree, "score_degree_pearson"),
        "alpha_log_degree_pearson": _nested(degree, "score_log_degree_pearson"),
        "alpha_degree_spearman": _nested(degree, "score_degree_spearman"),
        "alpha_log_degree_spearman": _nested(degree, "score_log_degree_spearman"),
        "gamma_normal_top10_mean": _nested(concentration, "normal", "node_level_concentration", "top10_percent_energy_share", "mean"),
        "gamma_anomaly_top10_mean": _nested(concentration, "anomaly", "node_level_concentration", "top10_percent_energy_share", "mean"),
        "gamma_normal_hhi_mean": _nested(concentration, "normal", "node_level_concentration", "normalized_hhi", "mean"),
        "gamma_anomaly_hhi_mean": _nested(concentration, "anomaly", "node_level_concentration", "normalized_hhi", "mean"),
        "diagnostics_path": str(diagnostic_path),
    }


def _degree_bucket_rows(path: Path, result: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostic_path = path.with_name(result.get("diagnostics_file", "diagnostics.json"))
    if not diagnostic_path.is_file():
        return []
    try:
        data = _read_json(diagnostic_path)
    except (OSError, json.JSONDecodeError):
        return []
    buckets = _nested(data, "alpha", "degree_diagnostics", "buckets", default=[])
    if not isinstance(buckets, list):
        return []
    rows = []
    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue
        rows.append({
            "study": result.get("study"), "variant": result.get("variant"),
            "dataset": result.get("dataset"), "seed": result.get("seed"),
            "bucket": bucket.get("bucket"), "count": bucket.get("count"),
            "anomaly_count": bucket.get("anomaly_count"),
            "degree_min": bucket.get("degree_min"), "degree_max": bucket.get("degree_max"),
            "auc": bucket.get("auc"), "auprc": bucket.get("auprc"),
            "diagnostics_path": str(diagnostic_path),
        })
    return rows


def _degree_bucket_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["study"], row["variant"], row["dataset"], row["bucket"])].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        count_mean, _ = _mean_std([value for row in group if (value := _as_float(row["count"])) is not None])
        anomalies_mean, _ = _mean_std([value for row in group if (value := _as_float(row["anomaly_count"])) is not None])
        auc_mean, auc_std = _mean_std([value for row in group if (value := _as_float(row["auc"])) is not None])
        pr_mean, pr_std = _mean_std([value for row in group if (value := _as_float(row["auprc"])) is not None])
        output.append({
            "study": key[0], "variant": key[1], "dataset": key[2], "bucket": key[3],
            "n": len(group), "count_mean": count_mean, "anomaly_count_mean": anomalies_mean,
            "auc_mean": auc_mean, "auc_std": auc_std, "auprc_mean": pr_mean, "auprc_std": pr_std,
        })
    return output


def _diagnostic_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["study"], row["variant"], row["dataset"])].append(row)
    output = []
    fields = [field for field in DIAGNOSTIC_FIELDS if field not in {"study", "variant", "dataset", "seed", "diagnostics_path"}]
    for key, group in sorted(grouped.items()):
        row = {"study": key[0], "variant": key[1], "dataset": key[2], "n": len(group)}
        for field in fields:
            mean, std = _mean_std([value for item in group if (value := _as_float(item.get(field))) is not None])
            row[f"{field}_mean"] = mean
            row[f"{field}_std"] = std
        output.append(row)
    return output


def aggregate(root: str | Path) -> dict[str, int]:
    """Validate and aggregate every completed seed below one output root."""

    root = Path(root).resolve()
    completed = _completed_results(root)
    run_rows = [_run_row(path, result) for path, result in completed]
    score_rows = [row for path, result in completed for row in _score_rows(path, result)]
    diagnostic_rows = [row for path, result in completed if (row := _diagnostic_row(path, result)) is not None]
    degree_bucket_rows = [row for path, result in completed for row in _degree_bucket_rows(path, result)]
    _atomic_csv(root / "runs.csv", RUN_FIELDS, run_rows)
    _atomic_csv(root / "summary.csv", SUMMARY_FIELDS, _summary_rows(run_rows))
    _atomic_csv(root / "score_only_runs.csv", SCORE_FIELDS, score_rows)
    _atomic_csv(root / "score_only_summary.csv", SCORE_SUMMARY_FIELDS, _score_summary(score_rows))
    _atomic_csv(root / "diagnostics_runs.csv", DIAGNOSTIC_FIELDS, diagnostic_rows)
    diagnostic_summary = _diagnostic_summary(diagnostic_rows)
    diagnostic_fields = sorted({key for row in diagnostic_summary for key in row})
    _atomic_csv(root / "diagnostics_summary.csv", diagnostic_fields, diagnostic_summary)
    _atomic_csv(root / "degree_bucket_runs.csv", DEGREE_BUCKET_FIELDS, degree_bucket_rows)
    _atomic_csv(
        root / "degree_bucket_summary.csv",
        DEGREE_BUCKET_SUMMARY_FIELDS,
        _degree_bucket_summary(degree_bucket_rows),
    )
    manifest = {
        "schema_version": 1,
        "completed_seed_runs": len(run_rows),
        "summary_rows": len(_summary_rows(run_rows)),
        "score_only_seed_rows": len(score_rows),
        "diagnostic_seed_rows": len(diagnostic_rows),
        "degree_bucket_seed_rows": len(degree_bucket_rows),
    }
    _atomic_json(root / "aggregate_manifest.json", manifest)
    print("Aggregation: " + ", ".join(f"{name}={value}" for name, value in manifest.items() if name != "schema_version"), flush=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Parse the aggregation output directory and build summary artifacts."""

    parser = argparse.ArgumentParser(description="Aggregate three-loss ablation result JSON files.")
    parser.add_argument("--output_dir", default="three_loss_ablation_results")
    args = parser.parse_args(argv)
    aggregate(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
