"""Label-free MARGAD epoch-budget search for non-T-Social datasets.

Each trial fixes the final README alpha/beta/gamma values and runs one seed.
The checkpoint is selected by the unsupervised total loss already used by the
trainer; final AUROC/AUPRC are recorded only after training for analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from dataset_config import resolve_dataset


ROOT = Path(__file__).resolve().parent
EPOCH_VALUES = tuple(range(50, 121, 5))
SEARCH_DATASETS = ("Facebook", "Reddit", "Amazon", "YelpChi", "elliptic", "tfinance")
DEFAULT_SEARCH_ROOT = "epoch_budget_search_h64_e50to120_step5_results"


@dataclass(frozen=True)
class FinalWeights:
    """Hold the fixed model settings used while varying epoch budgets."""

    hidden_dim: int
    lr: float
    alpha: float
    beta: float
    gamma: float


# Values transcribed from the final commands in README.md.  T-Social is
# deliberately excluded because GPU 0 remains reserved for its active search.
FINAL_WEIGHTS = {
    "facebook": FinalWeights(64, 3e-3, 1.0, 0.15, 0.50),
    "reddit": FinalWeights(64, 3e-3, 1.0, 0.35, 1.45),
    "amazon": FinalWeights(64, 3e-3, 1.0, 0.40, 0.15),
    "yelpchi": FinalWeights(64, 3e-3, 1.0, 0.15, 1.05),
    "elliptic": FinalWeights(64, 1e-3, 1.0, 0.30, 1.00),
    "tfinance": FinalWeights(64, 3e-3, 1.0, 1.00, 0.05),
}

RESULT_FIELDS = (
    "dataset", "epoch_budget", "alpha", "beta", "gamma", "hidden_dim", "lr",
    "selection_epoch", "selection_loss", "final_auc", "final_auprc",
    "gamma_normalize_bands", "gamma_input_tanh", "gamma_coefficient_tanh",
    "status", "return_code", "started_at", "finished_at", "duration_seconds",
    "device", "cuda_visible_devices", "log_path", "working_dir", "command",
)
SELECTION_FIELDS = (
    "dataset", "selection", "epoch_budget", "selection_epoch", "selection_loss",
    "alpha", "beta", "gamma", "hidden_dim", "lr", "final_auc", "final_auprc",
    "gamma_normalize_bands", "gamma_input_tanh", "gamma_coefficient_tanh",
    "log_path", "command",
)

FINAL_AUC_RE = re.compile(r"FINAL TESTING AUC:\s*([-+]?\d+(?:\.\d+)?)")
FINAL_AUPRC_RE = re.compile(r"FINAL TESTING AUPRC:\s*([-+]?\d+(?:\.\d+)?)")
SELECTION_RE = re.compile(
    r"BEST_SELECTION_EPOCH:\s*(\d+)\s+BEST_SELECTION_LOSS:\s*"
    r"([-+]?\d+(?:\.\d+)?)"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def root_path(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def decimal(value: float) -> str:
    return f"{value:.2f}"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Iterable[dict], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def numeric(row: dict, field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return float("inf")


def trial_dir(dataset_dir: Path, epoch_budget: int) -> Path:
    return dataset_dir / "trials" / f"epoch_{epoch_budget:03d}"


def load_results(dataset_dir: Path) -> list[dict]:
    rows = []
    for path in dataset_dir.glob("trials/epoch_*/result.json"):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            print(f"[warning] ignored unreadable result {path}: {error}", flush=True)
    return sorted(rows, key=lambda row: int(row.get("epoch_budget", -1)))


def parse_metrics(log: str) -> dict[str, str]:
    final_auc = FINAL_AUC_RE.findall(log)
    final_auprc = FINAL_AUPRC_RE.findall(log)
    selection = SELECTION_RE.findall(log)
    return {
        "final_auc": final_auc[-1] if final_auc else "",
        "final_auprc": final_auprc[-1] if final_auprc else "",
        "selection_epoch": selection[-1][0] if selection else "",
        "selection_loss": selection[-1][1] if selection else "",
    }


def refresh_reports(dataset_dir: Path) -> None:
    write_csv(dataset_dir / "trials.csv", load_results(dataset_dir), RESULT_FIELDS)


def run_trial(
    dataset: str,
    epoch_budget: int,
    *,
    data_dir: Path,
    search_root: Path,
    device: str,
    resume: bool,
) -> dict:
    spec = resolve_dataset(dataset)
    weights = FINAL_WEIGHTS[spec.key]
    dataset_dir = search_root / safe_name(spec.cli_name)
    workdir = trial_dir(dataset_dir, epoch_budget)
    artifact_path = workdir / "result.json"
    if resume and artifact_path.exists():
        try:
            prior = json.loads(artifact_path.read_text(encoding="utf-8"))
            if prior.get("status") == "completed":
                print(f"[resume] {spec.cli_name} epoch={epoch_budget}", flush=True)
                return prior
        except (OSError, json.JSONDecodeError):
            pass

    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "run.log"
    command = [
        sys.executable,
        str(ROOT / "run.py"),
        "--dataset", spec.cli_name,
        "--data_dir", str(data_dir),
        "--hidden_dim", str(weights.hidden_dim),
        "--lr", str(weights.lr),
        "--epoch", str(epoch_budget),
        "--alpha", decimal(weights.alpha),
        "--beta", decimal(weights.beta),
        "--gamma", decimal(weights.gamma),
        "--runs", "1",
        "--tests", "1",
        "--device", device,
        "--disable_monitor_auc",
    ]
    command_text = subprocess.list2cmdline(command)
    started_at = now()
    started_clock = datetime.now(timezone.utc)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return_code = -1
    print(f"[start] {spec.cli_name} epoch={epoch_budget}", flush=True)
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as handle:
            handle.write(
                f"Started UTC: {started_at}\n"
                f"CUDA_VISIBLE_DEVICES: {cuda_visible}\n"
                f"Device argument: {device}\n"
                "Selection: label-free checkpoint loss; epoch AUC monitoring disabled.\n"
                f"Command:\n{command_text}\n\n"
            )
            process = subprocess.run(
                command,
                cwd=workdir,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            return_code = process.returncode
            handle.write(f"\nProcess exit code: {return_code}\nFinished UTC: {now()}\n")
    except OSError as error:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\nLauncher error: {error!r}\n")

    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    metrics = parse_metrics(log)
    duration = (datetime.now(timezone.utc) - started_clock).total_seconds()
    completed = (
        return_code == 0
        and bool(metrics["final_auc"])
        and bool(metrics["final_auprc"])
        and bool(metrics["selection_epoch"])
        and bool(metrics["selection_loss"])
    )
    result = {
        "dataset": spec.cli_name,
        "epoch_budget": str(epoch_budget),
        "alpha": decimal(weights.alpha),
        "beta": decimal(weights.beta),
        "gamma": decimal(weights.gamma),
        "hidden_dim": str(weights.hidden_dim),
        "lr": str(weights.lr),
        "gamma_normalize_bands": str(spec.gamma_normalize_bands),
        "gamma_input_tanh": str(spec.gamma_input_tanh),
        "gamma_coefficient_tanh": str(spec.gamma_coefficient_tanh),
        "status": "completed" if completed else "failed",
        "return_code": str(return_code),
        "started_at": started_at,
        "finished_at": now(),
        "duration_seconds": f"{duration:.3f}",
        "device": device,
        "cuda_visible_devices": cuda_visible,
        "log_path": str(log_path.relative_to(dataset_dir)),
        "working_dir": str(workdir.relative_to(dataset_dir)),
        "command": command_text,
        **metrics,
    }
    atomic_json(artifact_path, result)
    refresh_reports(dataset_dir)
    print(
        f"[done] {spec.cli_name} epoch={epoch_budget} status={result['status']} "
        f"selection_loss={result['selection_loss'] or 'NA'}",
        flush=True,
    )
    return result


def select_epoch(dataset: str, rows: list[dict]) -> dict:
    spec = resolve_dataset(dataset)
    completed = [
        row for row in rows
        if row.get("status") == "completed" and math.isfinite(numeric(row, "selection_loss"))
    ]
    if len(completed) != len(EPOCH_VALUES):
        raise RuntimeError(
            f"{spec.cli_name}: expected {len(EPOCH_VALUES)} completed trials, "
            f"found {len(completed)}. Inspect failed run.log files and resume."
        )
    best = min(
        completed,
        key=lambda row: (
            numeric(row, "selection_loss"),
            int(row["selection_epoch"]),
            int(row["epoch_budget"]),
        ),
    )
    return {"dataset": spec.cli_name, "selection": "min_unsupervised_checkpoint_loss", **best}


def run_dataset(
    dataset: str,
    *,
    data_dir: Path,
    search_root: Path,
    device: str,
    resume: bool,
) -> None:
    spec = resolve_dataset(dataset)
    for epoch_budget in EPOCH_VALUES:
        run_trial(
            spec.cli_name,
            epoch_budget,
            data_dir=data_dir,
            search_root=search_root,
            device=device,
            resume=resume,
        )
    dataset_dir = search_root / safe_name(spec.cli_name)
    selection = select_epoch(spec.cli_name, load_results(dataset_dir))
    atomic_json(dataset_dir / "best_epoch_by_unsupervised_loss.json", selection)
    write_csv(dataset_dir / "best_epoch_by_unsupervised_loss.csv", [selection], SELECTION_FIELDS)
    print(
        f"[selected] {spec.cli_name}: epoch_budget={selection['epoch_budget']} "
        f"checkpoint_epoch={selection['selection_epoch']} "
        f"selection_loss={selection['selection_loss']}",
        flush=True,
    )


def aggregate(search_root: Path, datasets: Iterable[str]) -> None:
    all_rows: list[dict] = []
    selected_rows: list[dict] = []
    for dataset in datasets:
        spec = resolve_dataset(dataset)
        dataset_dir = search_root / safe_name(spec.cli_name)
        all_rows.extend(load_results(dataset_dir))
        selection_path = dataset_dir / "best_epoch_by_unsupervised_loss.json"
        if selection_path.exists():
            try:
                selected_rows.append(json.loads(selection_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as error:
                print(f"[warning] ignored unreadable selection {selection_path}: {error}", flush=True)
    write_csv(search_root / "all_trials.csv", all_rows, RESULT_FIELDS)
    write_csv(search_root / "selected_epochs.csv", selected_rows, SELECTION_FIELDS)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=SEARCH_DATASETS)
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--search_root", default=DEFAULT_SEARCH_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--skip_aggregate", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the requested epoch-budget trials and aggregate their results."""

    options = parse_args(argv)
    datasets = tuple(resolve_dataset(dataset).cli_name for dataset in options.datasets)
    unsupported = [dataset for dataset in datasets if resolve_dataset(dataset).key == "tsocial"]
    if unsupported:
        raise ValueError("T-Social is intentionally excluded: GPU 0 is reserved for its active search.")
    search_root = root_path(options.search_root)
    if options.dry_run:
        print(f"epoch values={EPOCH_VALUES[0]}..{EPOCH_VALUES[-1]} step=5 ({len(EPOCH_VALUES)} values)")
        for dataset in datasets:
            weights = FINAL_WEIGHTS[resolve_dataset(dataset).key]
            print(
                f"  {dataset}: trials={len(EPOCH_VALUES)}, hidden={weights.hidden_dim}, "
                f"lr={weights.lr}, alpha={weights.alpha}, beta={weights.beta}, gamma={weights.gamma}",
            )
        print(f"total trials={len(datasets) * len(EPOCH_VALUES)}")
        return
    if options.aggregate_only:
        aggregate(search_root, datasets)
        return
    for dataset in datasets:
        run_dataset(
            dataset,
            data_dir=root_path(options.data_dir),
            search_root=search_root,
            device=options.device,
            resume=not options.no_resume,
        )
    if not options.skip_aggregate:
        aggregate(search_root, datasets)


if __name__ == "__main__":
    main()
