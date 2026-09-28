"""Resumable staged beta/gamma search for the seven supported datasets.

Every dataset first searches 20 beta values with gamma=0, retains the three
highest final-AUC betas, and searches all 30 gamma values for each. Every
trial owns an isolated directory containing its command, complete log,
checkpoint, and durable result JSON.
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
from decimal import Decimal
from pathlib import Path
from typing import Iterable

from dataset_config import SUPPORTED_DATASETS, resolve_dataset


ROOT = Path(__file__).resolve().parent
DATASETS = SUPPORTED_DATASETS
STAGED_DATASET = "tsocial"
NON_TSOCIAL_TOP_BETAS = 3
TSOCIAL_TOP_BETAS = 3
DEFAULT_SEARCH_ROOT = "beta_gamma_staged_top3_h64_e100_tsocial10_results"


@dataclass(frozen=True)
class SearchConfig:
    hidden_dim: int
    lr: float
    epoch: int
    extra: tuple[str, ...] = ()


def config_for(dataset: str) -> SearchConfig:
    spec = resolve_dataset(dataset)
    defaults = spec.defaults
    extra: tuple[str, ...] = ()
    if spec.key == STAGED_DATASET:
        extra = (
            "--batch_size", str(defaults.batch_size),
            "--eval_batch_size", str(defaults.eval_batch_size),
            "--batch_fanout", str(defaults.batch_fanout),
            "--num_workers", str(defaults.num_workers),
            "--dgl_graph_on_gpu", str(defaults.dgl_graph_on_gpu),
        )
    return SearchConfig(
        hidden_dim=64,
        lr=1e-3 if spec.key == "elliptic" else 3e-3,
        epoch=10 if spec.key == STAGED_DATASET else 100,
        extra=extra,
    )


RESULT_FIELDS = (
    "dataset", "stage", "beta", "gamma", "gamma_centering", "alpha", "hidden_dim", "lr", "epoch",
    "final_auc", "final_auprc", "best_monitor_auc", "best_monitor_epoch",
    "gamma_normalize_bands", "gamma_input_tanh", "gamma_coefficient_tanh",
    "status", "return_code", "started_at", "finished_at", "duration_seconds",
    "device", "cuda_visible_devices", "log_path", "working_dir", "command",
)
BEST_FIELDS = (
    "dataset", "selection", "stage", "beta", "gamma", "gamma_centering", "alpha", "hidden_dim", "lr",
    "epoch", "final_auc", "final_auprc", "best_monitor_auc", "best_monitor_epoch",
    "gamma_normalize_bands", "gamma_input_tanh", "gamma_coefficient_tanh",
    "log_path", "command",
)

FINAL_AUC_RE = re.compile(r"FINAL TESTING AUC:\s*([-+]?\d+(?:\.\d+)?)")
FINAL_AUPRC_RE = re.compile(r"FINAL TESTING AUPRC:\s*([-+]?\d+(?:\.\d+)?)")
MONITOR_RE = re.compile(
    r"BEST_MONITOR_EPOCH:\s*(\d+)\s+BEST_MONITOR_AUC:\s*([-+]?\d+(?:\.\d+)?)"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def values(start: str, end: str, step: str) -> list[float]:
    current, stop, interval = Decimal(start), Decimal(end), Decimal(step)
    answer = []
    while current <= stop:
        answer.append(float(current))
        current += interval
    return answer


BETA_VALUES = values("0.05", "1.00", "0.05")
GAMMA_VALUES = values("0.05", "1.50", "0.05")


def weight(value: float) -> str:
    return f"{value:.2f}"


def root_path(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def numeric(row: dict, field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return float("-inf")


def parse_metrics(log: str) -> dict[str, str]:
    final_auc = FINAL_AUC_RE.findall(log)
    final_auprc = FINAL_AUPRC_RE.findall(log)
    monitor = MONITOR_RE.findall(log)
    return {
        "final_auc": final_auc[-1] if final_auc else "",
        "final_auprc": final_auprc[-1] if final_auprc else "",
        "best_monitor_auc": monitor[-1][1] if monitor else "",
        "best_monitor_epoch": monitor[-1][0] if monitor else "",
    }


def trial_name(beta: float, gamma: float) -> str:
    return f"b{weight(beta)}_g{weight(gamma)}"


def result_path(dataset_dir: Path, stage: str, beta: float, gamma: float) -> Path:
    return dataset_dir / "trials" / stage / trial_name(beta, gamma) / "result.json"


def load_results(dataset_dir: Path) -> list[dict]:
    rows = []
    for path in dataset_dir.glob("trials/*/*/result.json"):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            print(f"[warning] ignored unreadable result {path}: {error}", flush=True)
    return sorted(
        rows,
        key=lambda row: (row.get("stage", ""), row.get("beta", ""), row.get("gamma", "")),
    )


def write_csv(path: Path, rows: Iterable[dict], fields: Iterable[str]) -> bool:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
        return True
    except PermissionError:
        print(
            f"[warning] CSV is locked: {path}. JSON/log artifacts are safe; "
            "close the spreadsheet and run --aggregate_only.",
            flush=True,
        )
        temporary.unlink(missing_ok=True)
        return False


def choose(rows: Iterable[dict], metric: str) -> dict | None:
    ranked = rank_rows(rows, metric, limit=1)
    return ranked[0] if ranked else None


def rank_rows(rows: Iterable[dict], metric: str, limit: int) -> list[dict]:
    """Rank completed trials by a metric, breaking ties by smaller weights."""
    completed = [
        row for row in rows
        if row.get("status") == "completed" and numeric(row, metric) > -math.inf
    ]
    return sorted(
        completed,
        key=lambda row: (
            -numeric(row, metric),
            float(row["beta"]),
            float(row["gamma"]),
        ),
    )[:limit]


def selected_configurations(dataset: str, rows: Iterable[dict]) -> list[dict]:
    """Build stage-aware selections from final AUC without mixing phases."""
    rows = list(rows)
    spec = resolve_dataset(dataset)
    selected = []
    top_beta_count = (
        TSOCIAL_TOP_BETAS if spec.key == STAGED_DATASET else NON_TSOCIAL_TOP_BETAS
    )
    beta_rows = (row for row in rows if row.get("stage") == "beta")
    for rank, best in enumerate(rank_rows(beta_rows, "final_auc", top_beta_count), start=1):
        selected.append(
            {
                "dataset": spec.cli_name,
                "selection": f"selected_beta_rank_{rank}_by_final_auc",
                **best,
            }
        )
    best_gamma = choose(
        (row for row in rows if row.get("stage") == "gamma"), "final_auc"
    )
    if best_gamma is not None:
        selected.append(
            {
                "dataset": spec.cli_name,
                "selection": "selected_gamma_by_final_auc",
                **best_gamma,
            }
        )
    best_monitor = choose(rows, "best_monitor_auc")
    if best_monitor is not None:
        selected.append(
            {
                "dataset": spec.cli_name,
                "selection": "best_monitor_auc",
                **best_monitor,
            }
        )
    return selected


def refresh_reports(dataset_dir: Path) -> None:
    rows = load_results(dataset_dir)
    write_csv(dataset_dir / "trials.csv", rows, RESULT_FIELDS)
    if not rows:
        return
    dataset = rows[0]["dataset"]
    selected = selected_configurations(dataset, rows)
    write_csv(
        dataset_dir / f"best_configurations_{safe_name(dataset)}.csv",
        selected,
        BEST_FIELDS,
    )


def run_trial(
    dataset: str,
    stage: str,
    beta: float,
    gamma: float,
    *,
    data_dir: Path,
    search_root: Path,
    device: str,
    resume: bool,
    gamma_centering: int | None = None,
) -> dict:
    spec = resolve_dataset(dataset)
    config = config_for(dataset)
    if gamma_centering is None:
        gamma_centering = 0 if spec.key == STAGED_DATASET else 1
    if gamma_centering not in (0, 1):
        raise ValueError("gamma_centering must be 0 or 1.")
    dataset_dir = search_root / safe_name(spec.cli_name)
    artifact_path = result_path(dataset_dir, stage, beta, gamma)
    if resume and artifact_path.exists():
        try:
            prior = json.loads(artifact_path.read_text(encoding="utf-8"))
            if (
                prior.get("status") == "completed"
                and prior.get("gamma_centering") == str(gamma_centering)
            ):
                print(f"[resume] {spec.cli_name} {stage} {trial_name(beta, gamma)}", flush=True)
                return prior
        except (OSError, json.JSONDecodeError):
            pass

    workdir = artifact_path.parent
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "run.log"
    command = [
        sys.executable,
        str(ROOT / "run.py"),
        "--dataset", spec.cli_name,
        "--data_dir", str(data_dir),
        "--hidden_dim", str(config.hidden_dim),
        "--lr", str(config.lr),
        "--epoch", str(config.epoch),
        "--alpha", "1",
        "--beta", weight(beta),
        "--gamma", weight(gamma),
        "--gamma_centering", str(gamma_centering),
        "--runs", "1",
        "--tests", "1",
        "--device", device,
        *config.extra,
    ]
    command_text = subprocess.list2cmdline(command)
    started_at = now()
    started_clock = datetime.now(timezone.utc)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"[start] {spec.cli_name} {stage} {trial_name(beta, gamma)}", flush=True)
    return_code = -1
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as handle:
            handle.write(
                f"Started UTC: {started_at}\n"
                f"CUDA_VISIBLE_DEVICES: {cuda_visible}\n"
                f"Device argument: {device}\n"
                f"Stage: {stage}\n"
                f"Gamma centering: {gamma_centering}\n"
                f"Command:\n{command_text}\n\n"
            )
            handle.flush()
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
    result = {
        "dataset": spec.cli_name,
        "stage": stage,
        "beta": weight(beta),
        "gamma": weight(gamma),
        "gamma_centering": str(gamma_centering),
        "alpha": "1.00",
        "hidden_dim": str(config.hidden_dim),
        "lr": str(config.lr),
        "epoch": str(config.epoch),
        "gamma_normalize_bands": str(spec.gamma_normalize_bands),
        "gamma_input_tanh": str(spec.gamma_input_tanh),
        "gamma_coefficient_tanh": str(spec.gamma_coefficient_tanh),
        "status": "completed" if return_code == 0 and metrics["final_auc"] else "failed",
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
        f"[done] {spec.cli_name} {stage} {trial_name(beta, gamma)} "
        f"status={result['status']} final_auc={result['final_auc'] or 'NA'}",
        flush=True,
    )
    return result


def require_completed(rows: list[dict], expected: int, dataset: str, stage: str) -> None:
    completed = sum(row.get("status") == "completed" for row in rows)
    if completed != expected:
        raise RuntimeError(
            f"{dataset} {stage} incomplete ({completed}/{expected}). "
            "Inspect failed run.log files and rerun; completed trials will resume."
        )


def run_dataset(
    dataset: str,
    *,
    data_dir: Path,
    search_root: Path,
    device: str,
    resume: bool,
) -> None:
    spec = resolve_dataset(dataset)
    config = config_for(dataset)
    dataset_dir = search_root / safe_name(spec.cli_name)
    print(
        f"\n=== {spec.cli_name}: hidden={config.hidden_dim}, lr={config.lr}, "
        f"epoch={config.epoch}, device={device} ===",
        flush=True,
    )
    top_beta_count = (
        TSOCIAL_TOP_BETAS if spec.key == STAGED_DATASET else NON_TSOCIAL_TOP_BETAS
    )
    beta_rows = [
        run_trial(
            spec.cli_name, "beta", beta, 0.0,
            data_dir=data_dir, search_root=search_root,
            device=device, resume=resume,
        )
        for beta in BETA_VALUES
    ]
    require_completed(beta_rows, len(BETA_VALUES), spec.cli_name, "beta")
    selected_betas = rank_rows(beta_rows, "final_auc", top_beta_count)
    if len(selected_betas) != top_beta_count:
        raise RuntimeError(
            f"{spec.cli_name} selected only {len(selected_betas)}/{top_beta_count} "
            "completed beta trials."
        )
    atomic_json(
        dataset_dir / "beta_selection.json",
        {
            "selected_by": "final_auc",
            "top_beta_count": top_beta_count,
            "selected_betas": selected_betas,
        },
    )
    for rank, beta_row in enumerate(selected_betas, start=1):
        print(
            f"[select] {spec.cli_name} beta rank={rank} "
            f"value={beta_row['beta']} final_auc={beta_row['final_auc']}",
            flush=True,
        )

    gamma_rows = [
        run_trial(
            spec.cli_name, "gamma", float(beta_row["beta"]), gamma,
            data_dir=data_dir, search_root=search_root,
            device=device, resume=resume,
        )
        for beta_row in selected_betas
        for gamma in GAMMA_VALUES
    ]
    expected_gamma_trials = top_beta_count * len(GAMMA_VALUES)
    require_completed(gamma_rows, expected_gamma_trials, spec.cli_name, "gamma")
    best_gamma = choose(gamma_rows, "final_auc")
    assert best_gamma is not None
    atomic_json(
        dataset_dir / "staged_selection.json",
        {
            "selected_by": "final_auc",
            "top_beta_count": top_beta_count,
            "selected_betas": selected_betas,
            "best_gamma": best_gamma,
        },
    )
    print(
        f"[select] {spec.cli_name} final beta={best_gamma['beta']} "
        f"gamma={best_gamma['gamma']} final_auc={best_gamma['final_auc']}",
        flush=True,
    )
    refresh_reports(dataset_dir)


def aggregate(search_root: Path, datasets: Iterable[str]) -> None:
    all_rows, best_rows = [], []
    for dataset in datasets:
        spec = resolve_dataset(dataset)
        directory = search_root / safe_name(spec.cli_name)
        if not directory.exists():
            continue
        refresh_reports(directory)
        rows = load_results(directory)
        all_rows.extend(rows)
        best_rows.extend(selected_configurations(spec.cli_name, rows))
    write_csv(search_root / "all_trials.csv", all_rows, RESULT_FIELDS)
    write_csv(search_root / "all_best_configurations.csv", best_rows, BEST_FIELDS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable seven-dataset beta/gamma search.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--search_root", default=DEFAULT_SEARCH_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def dry_run(datasets: Iterable[str]) -> None:
    total = 0
    print(f"beta values={len(BETA_VALUES)}, gamma values={len(GAMMA_VALUES)}")
    for dataset in datasets:
        spec = resolve_dataset(dataset)
        config = config_for(dataset)
        top_beta_count = (
            TSOCIAL_TOP_BETAS
            if spec.key == STAGED_DATASET
            else NON_TSOCIAL_TOP_BETAS
        )
        count = len(BETA_VALUES) + top_beta_count * len(GAMMA_VALUES)
        total += count
        print(
            f"  {spec.cli_name}: trials={count}, hidden={config.hidden_dim}, "
            f"lr={config.lr}, epoch={config.epoch}, top_betas={top_beta_count}, "
            f"extra={config.extra}"
        )
    print(f"total trials={total}")


def main(argv: list[str] | None = None) -> None:
    options = parse_args(argv)
    search_root = root_path(options.search_root).resolve()
    if options.dry_run:
        dry_run(options.datasets)
        return
    if options.aggregate_only:
        aggregate(search_root, options.datasets)
        return
    data_dir = root_path(options.data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    for dataset in options.datasets:
        run_dataset(
            dataset,
            data_dir=data_dir,
            search_root=search_root,
            device=options.device,
            resume=not options.no_resume,
        )
    aggregate(search_root, options.datasets)


if __name__ == "__main__":
    main()
