"""Summarize independently trained F0--F5 format runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path

from ta_diffref_format_config import (
    EXPERIMENT_VERSION, parse_seeds, select_datasets,
    select_run_variants,
)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.fmean(values), statistics.pstdev(values)


def _run_row(root: Path, spec, variant, seed: int) -> dict:
    run_dir = root / "runs" / spec.key / variant.code / f"seed_{seed:02d}"
    result_path = run_dir / "result.json"
    row = {
        "dataset": spec.cli_name, "dataset_key": spec.key,
        "variant": variant.code, "alpha_mode": variant.alpha_mode,
        "seed": seed, "status": "missing", "auroc_percent": "",
        "auprc_percent": "", "train_seconds": "", "checkpoint": "",
        "result_path": str(result_path),
    }
    if (run_dir / "completed.json").is_file() and result_path.is_file():
        try:
            result = _read_json(result_path)
            metrics = result["metrics"]
            auc, auprc = float(metrics["auc"]), float(metrics["auprc"])
            if not (
                result["experiment_version"] == EXPERIMENT_VERSION
                and result["dataset_key"] == spec.key
                and result["variant"] == variant.code
                and result["seed"] == seed
                and math.isfinite(auc) and math.isfinite(auprc)
            ):
                raise ValueError("Result metadata or metric mismatch")
            row.update({
                "status": "completed", "auroc_percent": 100.0 * auc,
                "auprc_percent": 100.0 * auprc,
                "train_seconds": metrics.get("train_seconds", ""),
                "checkpoint": metrics.get("checkpoint", ""),
            })
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            row["status"] = "invalid"
    elif list(run_dir.glob("failure_attempt_*.json")):
        row["status"] = "failed"
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="return/ta_diffref_format_final_f0_f5_10seed")
    parser.add_argument("--datasets", nargs="*", default=("all",))
    parser.add_argument("--variants", nargs="*", default=("all",))
    parser.add_argument("--seeds", nargs="*", default=("0-9",))
    args = parser.parse_args(argv)
    root = Path(args.results_dir).resolve()
    datasets = select_datasets(args.datasets)
    variants = select_run_variants(args.variants)
    seeds = parse_seeds(args.seeds)
    run_rows = [
        _run_row(root, spec, variant, seed)
        for spec in datasets for variant in variants for seed in seeds
    ]
    summary_rows = []
    for spec in datasets:
        for variant in variants:
            group = [
                row for row in run_rows
                if row["dataset_key"] == spec.key and row["variant"] == variant.code
            ]
            valid = [row for row in group if row["status"] == "completed"]
            auc_mean, auc_std = _mean_std([float(row["auroc_percent"]) for row in valid])
            ap_mean, ap_std = _mean_std([float(row["auprc_percent"]) for row in valid])
            summary_rows.append({
                "dataset": spec.cli_name, "dataset_key": spec.key,
                "variant": variant.code, "alpha_mode": variant.alpha_mode,
                "source_kind": "new_10seed_experiment", "valid_n": len(valid),
                "failed_n": sum(row["status"] in ("failed", "invalid") for row in group),
                "missing_n": sum(row["status"] == "missing" for row in group),
                "auroc_mean_percent": auc_mean if auc_mean is not None else "",
                "auroc_std_percent": auc_std if auc_std is not None else "",
                "auprc_mean_percent": ap_mean if ap_mean is not None else "",
                "auprc_std_percent": ap_std if ap_std is not None else "",
                "source": str(root / "runs" / spec.key / variant.code),
            })
    lookup = {
        (row["dataset_key"], row["variant"], row["seed"]): row
        for row in run_rows if row["status"] == "completed"
    }
    paired = []
    for spec in datasets:
        for before, after, contrast in (
            ("F2", "F5", "calibration; target return retained"),
            ("F2", "F4", "target removal; raw propagation"),
            ("F3", "F5", "target return retained after calibration"),
        ):
            if not {before, after}.issubset({variant.code for variant in variants}):
                continue
            for metric in ("auroc_percent", "auprc_percent"):
                differences = [
                    float(lookup[(spec.key, after, seed)][metric])
                    - float(lookup[(spec.key, before, seed)][metric])
                    for seed in seeds
                    if (spec.key, before, seed) in lookup
                    and (spec.key, after, seed) in lookup
                ]
                mean, std = _mean_std(differences)
                paired.append({
                    "dataset": spec.cli_name, "from_variant": before,
                    "to_variant": after, "contrast": contrast,
                    "metric": "AUROC" if metric == "auroc_percent" else "AUPRC",
                    "paired_n": len(differences),
                    "mean_difference_percent_points": mean if mean is not None else "",
                    "std_difference_percent_points": std if std is not None else "",
                    "positive_seeds": sum(x > 0 for x in differences),
                    "negative_seeds": sum(x < 0 for x in differences),
                })

    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(summary_dir / "runs.csv", list(run_rows[0]), run_rows)
    _write_csv(summary_dir / "summary.csv", list(summary_rows[0]), summary_rows)
    if paired:
        _write_csv(summary_dir / "paired_differences.csv", list(paired[0]), paired)
    completed = sum(row["status"] == "completed" for row in run_rows)
    failed = sum(row["status"] in ("failed", "invalid") for row in run_rows)
    missing = sum(row["status"] == "missing" for row in run_rows)
    status = {
        "experiment_version": EXPERIMENT_VERSION,
        "expected_new_runs": len(run_rows), "completed_new_runs": completed,
        "failed_new_runs": failed, "missing_new_runs": missing,
        "complete": completed == len(run_rows),
    }
    _write_json(summary_dir / "completion_status.json", status)
    lines = [
        "# TA-DiffRef F0--F5 format study",
        "",
        "F0--F5: independently trained with the selected common seeds.",
        "Values are mean ± population standard deviation in percent.",
        "",
        "| Dataset | Format | Source | n | AUROC (%) | AUPRC (%) | Failed | Missing |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        def formatted(metric: str) -> str:
            mean, std = row[f"{metric}_mean_percent"], row[f"{metric}_std_percent"]
            return "NA" if mean == "" else f"{float(mean):.2f} ± {float(std):.2f}"
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['source_kind']} | "
            f"{row['valid_n']} | {formatted('auroc')} | {formatted('auprc')} | "
            f"{row['failed_n']} | {row['missing_n']} |"
        )
    lines.extend((
        "", "Paired seed differences in `paired_differences.csv` cover F2→F5,",
        "F2→F4, and F3→F5 where both variants have completed common seeds.", "",
    ))
    (summary_dir / "complete_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"Aggregated new runs: expected={len(run_rows)} completed={completed} "
        f"failed={failed} missing={missing} -> {summary_dir}",
        flush=True,
    )
    return 0 if status["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
