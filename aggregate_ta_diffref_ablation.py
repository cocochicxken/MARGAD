"""Aggregate TA-DiffRef A0--A6 runs into plotting- and table-ready files."""

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

from ta_diffref_ablation_config import (
    EXPERIMENT_VERSION,
    PAIRINGS,
    parse_seeds,
    select_datasets,
    select_variants,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})
    os.replace(temporary, path)


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(statistics.fmean(values)), float(statistics.pstdev(values))


def _latest_failure(run_dir: Path) -> dict[str, Any] | None:
    candidates = sorted(run_dir.glob("failure_attempt_*.json"))
    if not candidates:
        return None
    try:
        return _read_json(candidates[-1])
    except (OSError, json.JSONDecodeError):
        return {"status": "unreadable_failure_record"}


def _run_row(root: Path, spec, variant, seed: int) -> dict[str, Any]:
    run_dir = root / "runs" / spec.key / variant.code / f"seed_{seed:02d}"
    result_path = run_dir / "result.json"
    completed_path = run_dir / "completed.json"
    row = {
        "experiment_version": EXPERIMENT_VERSION,
        "dataset": spec.cli_name,
        "dataset_key": spec.key,
        "variant": variant.code,
        "alpha_mode": variant.alpha_mode,
        "seed": seed,
        "status": "missing",
        "failure_reason": None,
        "auc": None,
        "auprc": None,
        "auc_percent": None,
        "auprc_percent": None,
        "best_selection_epoch": None,
        "best_selection_loss": None,
        "train_seconds": None,
        "inference_seconds": None,
        "total_seconds": None,
        "peak_gpu_allocated_mib": None,
        "checkpoint": None,
        "result_path": str(result_path),
    }
    if completed_path.is_file() and result_path.is_file():
        try:
            result = _read_json(result_path)
            metrics = result.get("metrics", {})
            auc = _finite(metrics.get("auc"))
            auprc = _finite(metrics.get("auprc"))
            if (
                result.get("experiment_version") == EXPERIMENT_VERSION
                and result.get("dataset_key") == spec.key
                and result.get("variant") == variant.code
                and result.get("seed") == seed
                and auc is not None
                and auprc is not None
            ):
                row.update({
                    "status": "completed",
                    "auc": auc,
                    "auprc": auprc,
                    "auc_percent": 100.0 * auc,
                    "auprc_percent": 100.0 * auprc,
                    "best_selection_epoch": metrics.get("best_selection_epoch"),
                    "best_selection_loss": metrics.get("best_selection_loss"),
                    "train_seconds": metrics.get("train_seconds"),
                    "inference_seconds": metrics.get("inference_seconds"),
                    "total_seconds": metrics.get("total_seconds"),
                    "peak_gpu_allocated_mib": metrics.get("peak_gpu_allocated_mib"),
                    "checkpoint": metrics.get("checkpoint"),
                })
                return row
            row["status"] = "invalid"
            row["failure_reason"] = "result metadata mismatch or non-finite metric"
            return row
        except (OSError, json.JSONDecodeError, TypeError) as error:
            row["status"] = "invalid"
            row["failure_reason"] = str(error)
            return row
    failure = _latest_failure(run_dir)
    if failure is not None:
        row["status"] = str(failure.get("status", "failed"))
        row["failure_reason"] = failure.get("error") or failure.get("status")
    return row


RUN_FIELDS = [
    "experiment_version", "dataset", "dataset_key", "variant", "alpha_mode", "seed",
    "status", "failure_reason", "auc", "auprc", "auc_percent", "auprc_percent",
    "best_selection_epoch", "best_selection_loss", "train_seconds", "inference_seconds",
    "total_seconds", "peak_gpu_allocated_mib", "checkpoint", "result_path",
]


SUMMARY_FIELDS = [
    "dataset", "dataset_key", "variant", "alpha_mode", "valid_n", "failed_n", "missing_n",
    "auc_mean_percent", "auc_std_percent", "auprc_mean_percent", "auprc_std_percent",
    "selection_loss_mean", "train_seconds_mean", "total_seconds_mean",
]


def _summary_rows(rows: list[dict[str, Any]], datasets, variants) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset_key"], row["variant"])].append(row)
    summaries = []
    for spec in datasets:
        for variant in variants:
            values = grouped[(spec.key, variant.code)]
            valid = [row for row in values if row["status"] == "completed"]
            failed = [row for row in values if row["status"] not in ("completed", "missing")]
            missing = [row for row in values if row["status"] == "missing"]
            auc_mean, auc_std = _mean_std([float(row["auc_percent"]) for row in valid])
            pr_mean, pr_std = _mean_std([float(row["auprc_percent"]) for row in valid])
            selection_mean, _ = _mean_std([
                value for row in valid if (value := _finite(row["best_selection_loss"])) is not None
            ])
            train_mean, _ = _mean_std([
                value for row in valid if (value := _finite(row["train_seconds"])) is not None
            ])
            total_mean, _ = _mean_std([
                value for row in valid if (value := _finite(row["total_seconds"])) is not None
            ])
            summaries.append({
                "dataset": spec.cli_name,
                "dataset_key": spec.key,
                "variant": variant.code,
                "alpha_mode": variant.alpha_mode,
                "valid_n": len(valid),
                "failed_n": len(failed),
                "missing_n": len(missing),
                "auc_mean_percent": auc_mean,
                "auc_std_percent": auc_std,
                "auprc_mean_percent": pr_mean,
                "auprc_std_percent": pr_std,
                "selection_loss_mean": selection_mean,
                "train_seconds_mean": train_mean,
                "total_seconds_mean": total_mean,
            })
    return summaries


PAIR_FIELDS = [
    "dataset", "from_variant", "to_variant", "comparison", "metric", "paired_n",
    "mean_difference_percent_points", "std_difference_percent_points",
    "positive_seed_count", "zero_seed_count", "negative_seed_count",
]


PER_SEED_PAIR_FIELDS = [
    "dataset", "from_variant", "to_variant", "comparison", "metric", "seed",
    "from_value_percent", "to_value_percent", "difference_percent_points",
]


def _paired_rows(rows: list[dict[str, Any]], datasets, variants):
    allowed = {variant.code for variant in variants}
    lookup = {
        (row["dataset_key"], row["variant"], row["seed"]): row
        for row in rows if row["status"] == "completed"
    }
    summary_rows, seed_rows = [], []
    for spec in datasets:
        for source, target, description in PAIRINGS:
            if source not in allowed or target not in allowed:
                continue
            for metric in ("auc_percent", "auprc_percent"):
                differences = []
                for seed in sorted({key[2] for key in lookup if key[0] == spec.key}):
                    left = lookup.get((spec.key, source, seed))
                    right = lookup.get((spec.key, target, seed))
                    if left is None or right is None:
                        continue
                    source_value = float(left[metric])
                    target_value = float(right[metric])
                    difference = target_value - source_value
                    differences.append(difference)
                    seed_rows.append({
                        "dataset": spec.cli_name,
                        "from_variant": source,
                        "to_variant": target,
                        "comparison": description,
                        "metric": "AUROC" if metric == "auc_percent" else "AUPRC",
                        "seed": seed,
                        "from_value_percent": source_value,
                        "to_value_percent": target_value,
                        "difference_percent_points": difference,
                    })
                mean, std = _mean_std(differences)
                summary_rows.append({
                    "dataset": spec.cli_name,
                    "from_variant": source,
                    "to_variant": target,
                    "comparison": description,
                    "metric": "AUROC" if metric == "auc_percent" else "AUPRC",
                    "paired_n": len(differences),
                    "mean_difference_percent_points": mean,
                    "std_difference_percent_points": std,
                    "positive_seed_count": sum(value > 0 for value in differences),
                    "zero_seed_count": sum(value == 0 for value in differences),
                    "negative_seed_count": sum(value < 0 for value in differences),
                })
    return summary_rows, seed_rows


def _format_metric(mean, std, n: int) -> str:
    if mean is None or std is None or n == 0:
        return "NA"
    return f"{mean:.2f} ± {std:.2f} (n={n})"


def _write_markdown(path: Path, summaries: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> None:
    lines = [
        "# TA-DiffRef A0–A6 final-configuration ablation results",
        "",
        "Values are percentages and population standard deviations across valid seeds.",
        "",
        "| Dataset | Variant | Valid n | AUROC | AUPRC | Failed | Missing |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['valid_n']} | "
            f"{_format_metric(row['auc_mean_percent'], row['auc_std_percent'], row['valid_n'])} | "
            f"{_format_metric(row['auprc_mean_percent'], row['auprc_std_percent'], row['valid_n'])} | "
            f"{row['failed_n']} | {row['missing_n']} |"
        )
    lines.extend((
        "",
        "## Predeclared paired differences",
        "",
        "Differences are `to − from` in percentage points using common valid seeds.",
        "",
        "| Dataset | Pair | Metric | Paired n | Mean ± population std | Positive / zero / negative |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ))
    for row in pairs:
        mean = row["mean_difference_percent_points"]
        std = row["std_difference_percent_points"]
        value = "NA" if mean is None or std is None else f"{mean:.2f} ± {std:.2f}"
        lines.append(
            f"| {row['dataset']} | {row['from_variant']}→{row['to_variant']} | {row['metric']} | "
            f"{row['paired_n']} | {value} | {row['positive_seed_count']} / "
            f"{row['zero_seed_count']} / {row['negative_seed_count']} |"
        )
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="return/ta_diffref_mechanism_ablation")
    parser.add_argument("--datasets", nargs="*", default=("all",))
    parser.add_argument("--variants", nargs="*", default=("all",))
    parser.add_argument("--seeds", nargs="*", default=("0-4",))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.results_dir).resolve()
    datasets = select_datasets(args.datasets)
    variants = select_variants(args.variants)
    seeds = parse_seeds(args.seeds)
    rows = [
        _run_row(root, spec, variant, seed)
        for spec in datasets for variant in variants for seed in seeds
    ]
    summaries = _summary_rows(rows, datasets, variants)
    paired, paired_by_seed = _paired_rows(rows, datasets, variants)
    summary_dir = root / "summary"
    _write_csv(summary_dir / "runs.csv", RUN_FIELDS, rows)
    _write_csv(summary_dir / "summary.csv", SUMMARY_FIELDS, summaries)
    _write_csv(summary_dir / "paired_differences.csv", PAIR_FIELDS, paired)
    _write_csv(summary_dir / "paired_differences_by_seed.csv", PER_SEED_PAIR_FIELDS, paired_by_seed)
    _write_json(summary_dir / "plot_data.json", {
        "schema_version": 1,
        "experiment_version": EXPERIMENT_VERSION,
        "metrics_unit": "percent",
        "runs": rows,
        "summary": summaries,
        "paired_differences": paired,
        "paired_differences_by_seed": paired_by_seed,
    })
    _write_markdown(summary_dir / "complete_results.md", summaries, paired)
    completed = sum(row["status"] == "completed" for row in rows)
    failed = sum(row["status"] not in ("completed", "missing") for row in rows)
    missing = sum(row["status"] == "missing" for row in rows)
    _write_json(summary_dir / "completion_status.json", {
        "experiment_version": EXPERIMENT_VERSION,
        "expected_runs": len(rows),
        "completed_runs": completed,
        "failed_runs": failed,
        "missing_runs": missing,
        "complete": completed == len(rows),
    })
    print(
        f"Aggregated expected={len(rows)} completed={completed} failed={failed} "
        f"missing={missing} -> {summary_dir}",
        flush=True,
    )
    return 0 if completed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
