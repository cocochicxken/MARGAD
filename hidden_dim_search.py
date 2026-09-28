"""Resumable hidden-dimension search using the final README configurations.

All parameters are fixed per dataset except hidden_dim.  Training-time AUC is
disabled so checkpoint selection remains label-free; final AUROC/AUPRC are
recorded after training for sensitivity reporting.  Each configuration uses
five deterministic seeds.  CUDA OOM trials are durable terminal results rather
than generic failures.
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
HIDDEN_DIMS = (2, 4, 8, 16, 32, 64, 128, 256)
RUNS = 5
SEARCH_DATASETS = ("Facebook", "Reddit", "Amazon", "YelpChi", "elliptic", "tfinance", "tsocial")
DEFAULT_SEARCH_ROOT = "hidden_dim_search_readme_final_h2to256_results"


@dataclass(frozen=True)
class FinalConfiguration:
    lr: float
    epoch: int
    alpha: float
    beta: float
    gamma: float
    extra: tuple[str, ...] = ()


# Transcribed from the final commands in README.md.  hidden_dim is intentionally
# absent because it is the only searched parameter.
FINAL_CONFIGURATIONS = {
    "facebook": FinalConfiguration(3e-3, 70, 1.0, 0.15, 0.50),
    "reddit": FinalConfiguration(3e-3, 110, 1.0, 0.35, 1.45),
    "amazon": FinalConfiguration(3e-3, 70, 1.0, 0.40, 0.15),
    "yelpchi": FinalConfiguration(3e-3, 65, 1.0, 0.15, 1.05),
    "elliptic": FinalConfiguration(1e-3, 70, 1.0, 0.30, 1.00),
    "tfinance": FinalConfiguration(3e-3, 85, 1.0, 1.00, 0.05),
    "tsocial": FinalConfiguration(
        3e-3, 10, 1.0, 0.85, 0.75,
        (
            "--batch_size", "51200",
            "--eval_batch_size", "51200",
            "--batch_fanout", "8",
            "--num_workers", "0",
            "--dgl_graph_on_gpu", "1",
        ),
    ),
}

RESULT_FIELDS = (
    "dataset", "hidden_dim", "runs", "alpha", "beta", "gamma", "lr", "epoch",
    "selection_epoch", "selection_loss", "selection_loss_std", "selection_epochs",
    "selection_losses", "final_auc", "final_auc_std", "final_auprc", "final_auprc_std",
    "gamma_normalize_bands", "gamma_input_tanh", "gamma_coefficient_tanh",
    "status", "oom_detected", "return_code", "started_at", "finished_at",
    "duration_seconds", "device", "cuda_visible_devices", "log_path",
    "working_dir", "command",
)
SELECTION_FIELDS = (
    "dataset", "selection", "hidden_dim", "runs", "selection_epoch", "selection_loss",
    "selection_loss_std", "selection_epochs", "selection_losses", "alpha", "beta", "gamma",
    "lr", "epoch", "final_auc", "final_auc_std", "final_auprc", "final_auprc_std",
    "gamma_normalize_bands", "gamma_input_tanh", "gamma_coefficient_tanh",
    "log_path", "command",
)

FINAL_AUC_RE = re.compile(r"FINAL TESTING AUC:\s*([-+]?\d+(?:\.\d+)?)")
FINAL_AUPRC_RE = re.compile(r"FINAL TESTING AUPRC:\s*([-+]?\d+(?:\.\d+)?)")
FINAL_AUC_STD_RE = re.compile(r"FINAL TESTING AUC std:\s*([-+]?\d+(?:\.\d+)?)")
FINAL_AUPRC_STD_RE = re.compile(r"FINAL TESTING AUPRC std:\s*([-+]?\d+(?:\.\d+)?)")
SELECTION_RE = re.compile(
    r"BEST_SELECTION_EPOCH:\s*(\d+)\s+BEST_SELECTION_LOSS:\s*"
    r"([-+]?\d+(?:\.\d+)?)"
)
OOM_RE = re.compile(
    r"(?:cuda|cudnn|cublas|torch\.cuda).{0,100}out of memory|"
    r"out of memory.{0,100}(?:cuda|cudnn|cublas|torch)|"
    r"torch\.cuda\.outofmemoryerror",
    re.IGNORECASE,
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


def as_float(row: dict, field: str, fallback: float) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return fallback


def trial_dir(dataset_dir: Path, hidden_dim: int) -> Path:
    return dataset_dir / "trials" / f"hidden_{hidden_dim:03d}"


def load_results(dataset_dir: Path) -> list[dict]:
    rows = []
    for path in dataset_dir.glob("trials/hidden_*/result.json"):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            print(f"[warning] ignored unreadable result {path}: {error}", flush=True)
    return sorted(rows, key=lambda row: int(row.get("hidden_dim", -1)))


def parse_metrics(log: str) -> dict[str, str]:
    final_auc = FINAL_AUC_RE.findall(log)
    final_auprc = FINAL_AUPRC_RE.findall(log)
    final_auc_std = FINAL_AUC_STD_RE.findall(log)
    final_auprc_std = FINAL_AUPRC_STD_RE.findall(log)
    selection = SELECTION_RE.findall(log)
    selection_epochs = [int(epoch) for epoch, _ in selection]
    selection_losses = [float(loss) for _, loss in selection]
    selection_count = len(selection_losses)
    selection_epoch = (
        f"{sum(selection_epochs) / selection_count:.4f}" if selection_count else ""
    )
    selection_loss = (
        f"{sum(selection_losses) / selection_count:.10f}" if selection_count else ""
    )
    selection_loss_std = (
        f"{math.sqrt(sum((loss - sum(selection_losses) / selection_count) ** 2 for loss in selection_losses) / selection_count):.10f}"
        if selection_count else ""
    )
    return {
        "final_auc": final_auc[-1] if final_auc else "",
        "final_auc_std": final_auc_std[-1] if final_auc_std else "",
        "final_auprc": final_auprc[-1] if final_auprc else "",
        "final_auprc_std": final_auprc_std[-1] if final_auprc_std else "",
        "selection_epoch": selection_epoch,
        "selection_loss": selection_loss,
        "selection_loss_std": selection_loss_std,
        "selection_epochs": ",".join(str(epoch) for epoch in selection_epochs),
        "selection_losses": ",".join(f"{loss:.10f}" for loss in selection_losses),
        "selection_count": str(selection_count),
    }


def refresh_reports(dataset_dir: Path) -> None:
    rows = load_results(dataset_dir)
    write_csv(dataset_dir / "trials.csv", rows, RESULT_FIELDS)
    write_csv(
        dataset_dir / "trials_ranked_by_final_auc.csv",
        sorted(rows, key=lambda row: as_float(row, "final_auc", float("-inf")), reverse=True),
        RESULT_FIELDS,
    )


def run_trial(
    dataset: str,
    hidden_dim: int,
    *,
    data_dir: Path,
    search_root: Path,
    device: str,
    resume: bool,
    retry_oom: bool,
    stream_logs: bool,
    refresh_dataset_reports: bool,
) -> dict:
    spec = resolve_dataset(dataset)
    config = FINAL_CONFIGURATIONS[spec.key]
    dataset_dir = search_root / safe_name(spec.cli_name)
    workdir = trial_dir(dataset_dir, hidden_dim)
    artifact_path = workdir / "result.json"
    if resume and artifact_path.exists():
        try:
            prior = json.loads(artifact_path.read_text(encoding="utf-8"))
            completed_matches_current_config = (
                prior.get("status") == "completed"
                and prior.get("runs") == str(RUNS)
                and prior.get("hidden_dim") == str(hidden_dim)
                and prior.get("alpha") == decimal(config.alpha)
                and prior.get("beta") == decimal(config.beta)
                and prior.get("gamma") == decimal(config.gamma)
                and prior.get("lr") == str(config.lr)
                and prior.get("epoch") == str(config.epoch)
            )
            terminal = completed_matches_current_config or (
                prior.get("status") == "oom" and not retry_oom
            )
            if terminal:
                print(f"[resume] {spec.cli_name} hidden_dim={hidden_dim} status={prior['status']}", flush=True)
                return prior
            if prior.get("status") == "completed":
                print(
                    f"[rerun-config-change] {spec.cli_name} hidden_dim={hidden_dim} "
                    "stored result does not match the current configuration",
                    flush=True,
                )
        except (OSError, json.JSONDecodeError):
            pass

    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "run.log"
    command = [
        sys.executable,
        str(ROOT / "run.py"),
        "--dataset", spec.cli_name,
        "--data_dir", str(data_dir),
        "--hidden_dim", str(hidden_dim),
        "--lr", str(config.lr),
        "--epoch", str(config.epoch),
        "--alpha", decimal(config.alpha),
        "--beta", decimal(config.beta),
        "--gamma", decimal(config.gamma),
        "--runs", str(RUNS),
        "--tests", "1",
        "--device", device,
        "--disable_monitor_auc",
        *config.extra,
    ]
    command_text = subprocess.list2cmdline(command)
    started_at = now()
    started_clock = datetime.now(timezone.utc)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return_code = -1
    print(f"[start] {spec.cli_name} hidden_dim={hidden_dim}", flush=True)
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as handle:
            handle.write(
                f"Started UTC: {started_at}\n"
                f"CUDA_VISIBLE_DEVICES: {cuda_visible}\n"
                f"Device argument: {device}\n"
                "Selection: label-free checkpoint loss; epoch AUC monitoring disabled.\n"
                f"Command:\n{command_text}\n\n"
            )
            environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
            if stream_logs:
                process = subprocess.Popen(
                    command,
                    cwd=workdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=environment,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    handle.write(line)
                    print(f"[{spec.cli_name} h={hidden_dim}] {line}", end="", flush=True)
                return_code = process.wait()
            else:
                process = subprocess.run(
                    command,
                    cwd=workdir,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                    env=environment,
                )
                return_code = process.returncode
            handle.write(f"\nProcess exit code: {return_code}\nFinished UTC: {now()}\n")
    except OSError as error:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\nLauncher error: {error!r}\n")

    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    metrics = parse_metrics(log)
    oom_detected = bool(OOM_RE.search(log))
    duration = (datetime.now(timezone.utc) - started_clock).total_seconds()
    completed = (
        return_code == 0
        and bool(metrics["final_auc"])
        and bool(metrics["final_auprc"])
        and bool(metrics["selection_epoch"])
        and bool(metrics["selection_loss"])
        and metrics["selection_count"] == str(RUNS)
    )
    status = "completed" if completed else "oom" if oom_detected else "failed"
    result = {
        "dataset": spec.cli_name,
        "hidden_dim": str(hidden_dim),
        "runs": str(RUNS),
        "alpha": decimal(config.alpha),
        "beta": decimal(config.beta),
        "gamma": decimal(config.gamma),
        "lr": str(config.lr),
        "epoch": str(config.epoch),
        "gamma_normalize_bands": str(spec.gamma_normalize_bands),
        "gamma_input_tanh": str(spec.gamma_input_tanh),
        "gamma_coefficient_tanh": str(spec.gamma_coefficient_tanh),
        "status": status,
        "oom_detected": str(oom_detected),
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
    if refresh_dataset_reports:
        refresh_reports(dataset_dir)
    print(
        f"[done] {spec.cli_name} hidden_dim={hidden_dim} status={status} "
        f"selection_loss_mean={result['selection_loss'] or 'NA'}",
        flush=True,
    )
    return result


def select_hidden_dim(dataset: str, rows: list[dict]) -> dict:
    spec = resolve_dataset(dataset)
    completed = [
        row for row in rows
        if row.get("status") == "completed"
        and math.isfinite(as_float(row, "selection_loss", float("inf")))
    ]
    if not completed:
        return {
            "dataset": spec.cli_name,
            "selection": "no_completed_trial",
            "status": "all_trials_oom_or_failed",
        }
    best = min(
        completed,
        key=lambda row: (
            as_float(row, "selection_loss", float("inf")),
            int(row["hidden_dim"]),
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
    retry_oom: bool,
    stream_logs: bool,
    hidden_dims: tuple[int, ...],
    write_dataset_reports: bool,
) -> None:
    spec = resolve_dataset(dataset)
    for hidden_dim in hidden_dims:
        run_trial(
            spec.cli_name,
            hidden_dim,
            data_dir=data_dir,
            search_root=search_root,
            device=device,
            resume=resume,
            retry_oom=retry_oom,
            stream_logs=stream_logs,
            refresh_dataset_reports=write_dataset_reports,
        )
    dataset_dir = search_root / safe_name(spec.cli_name)
    rows = load_results(dataset_dir)
    requested_dimensions = {str(hidden_dim) for hidden_dim in hidden_dims}
    failures = [
        row for row in rows
        if row.get("hidden_dim") in requested_dimensions and row.get("status") == "failed"
    ]
    if failures:
        raise RuntimeError(
            f"{spec.cli_name}: {len(failures)} non-OOM trial(s) failed. "
            "Inspect run.log and rerun without --no_resume."
        )
    if not write_dataset_reports:
        return
    refresh_reports(dataset_dir)
    selection = select_hidden_dim(spec.cli_name, rows)
    atomic_json(dataset_dir / "best_hidden_dim_by_unsupervised_loss.json", selection)
    write_csv(dataset_dir / "best_hidden_dim_by_unsupervised_loss.csv", [selection], SELECTION_FIELDS)
    print(
        f"[selected] {spec.cli_name}: hidden_dim={selection.get('hidden_dim', 'NA')} "
        f"selection_loss_mean={selection.get('selection_loss', 'NA')}",
        flush=True,
    )


def aggregate(search_root: Path, datasets: Iterable[str]) -> None:
    all_rows: list[dict] = []
    selected_rows: list[dict] = []
    for dataset in datasets:
        spec = resolve_dataset(dataset)
        dataset_dir = search_root / safe_name(spec.cli_name)
        rows = load_results(dataset_dir)
        all_rows.extend(rows)
        refresh_reports(dataset_dir)
        selection = select_hidden_dim(spec.cli_name, rows)
        atomic_json(dataset_dir / "best_hidden_dim_by_unsupervised_loss.json", selection)
        write_csv(dataset_dir / "best_hidden_dim_by_unsupervised_loss.csv", [selection], SELECTION_FIELDS)
        selected_rows.append(selection)
    write_csv(search_root / "all_trials.csv", all_rows, RESULT_FIELDS)
    write_csv(
        search_root / "all_trials_ranked_by_final_auc.csv",
        sorted(all_rows, key=lambda row: as_float(row, "final_auc", float("-inf")), reverse=True),
        RESULT_FIELDS,
    )
    write_csv(search_root / "selected_hidden_dims.csv", selected_rows, SELECTION_FIELDS)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=SEARCH_DATASETS)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=list(HIDDEN_DIMS))
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--search_root", default=DEFAULT_SEARCH_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--retry_oom", action="store_true")
    parser.add_argument(
        "--stream_logs",
        action="store_true",
        help="Mirror each training subprocess line to this terminal while preserving run.log.",
    )
    parser.add_argument("--skip_aggregate", action="store_true")
    parser.add_argument(
        "--skip_dataset_reports",
        action="store_true",
        help="For disjoint parallel hidden-dimension shards; final aggregation rebuilds reports.",
    )
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    options = parse_args(argv)
    datasets = tuple(resolve_dataset(dataset).cli_name for dataset in options.datasets)
    hidden_dims = tuple(options.hidden_dims)
    if not hidden_dims or any(hidden_dim not in HIDDEN_DIMS for hidden_dim in hidden_dims):
        raise ValueError(f"--hidden_dims must be a non-empty subset of {HIDDEN_DIMS}.")
    if len(set(hidden_dims)) != len(hidden_dims):
        raise ValueError("--hidden_dims must not contain duplicates.")
    if options.dry_run:
        print(f"hidden dimensions={hidden_dims} ({len(hidden_dims)} values), runs={RUNS}")
        for dataset in datasets:
            spec = resolve_dataset(dataset)
            config = FINAL_CONFIGURATIONS[spec.key]
            print(
                f"  {spec.cli_name}: trials={len(hidden_dims)}, lr={config.lr}, "
                f"epoch={config.epoch}, alpha={config.alpha}, beta={config.beta}, "
                f"gamma={config.gamma}, extra={config.extra}",
            )
        configurations = len(datasets) * len(hidden_dims)
        print(f"total configurations={configurations}, total training runs={configurations * RUNS}")
        return
    search_root = root_path(options.search_root)
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
            retry_oom=options.retry_oom,
            stream_logs=options.stream_logs,
            hidden_dims=hidden_dims,
            write_dataset_reports=not options.skip_dataset_reports,
        )
    if not options.skip_aggregate:
        aggregate(search_root, datasets)


if __name__ == "__main__":
    main()
