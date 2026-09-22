"""Run final MARGAD configurations ten times and collect efficiency data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from dataset_config import resolve_dataset


ROOT = Path(__file__).resolve().parent
RUNS = 10
DATASETS = ("Facebook", "Reddit", "Amazon", "YelpChi", "elliptic", "tfinance", "tsocial")
DEFAULT_RESULTS_ROOT = "final_readme_10runs_efficiency_results"


@dataclass(frozen=True)
class FinalConfiguration:
    """Store one dataset's final command-line configuration."""

    hidden_dim: int
    lr: float
    epoch: int
    alpha: float
    beta: float
    gamma: float
    gamma_centering: int = 1
    extra: tuple[str, ...] = ()


# Transcribed from the final commands in README.md.
FINAL_CONFIGURATIONS = {
    "facebook": FinalConfiguration(64, 3e-3, 70, 1.0, 0.15, 0.50),
    "reddit": FinalConfiguration(128, 3e-3, 110, 1.0, 0.35, 1.45),
    "amazon": FinalConfiguration(64, 3e-3, 70, 1.0, 0.40, 0.15),
    "yelpchi": FinalConfiguration(64, 3e-3, 65, 1.0, 0.15, 1.05),
    "elliptic": FinalConfiguration(64, 1e-3, 70, 1.0, 0.30, 1.00),
    "tfinance": FinalConfiguration(64, 3e-3, 85, 1.0, 1.00, 0.05),
    "tsocial": FinalConfiguration(
        64, 3e-3, 10, 1.0, 0.85, 0.75, 0,
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
    "dataset", "runs", "hidden_dim", "lr", "epoch", "alpha", "beta", "gamma", "gamma_centering",
    "final_auc_mean", "final_auc_std", "final_auprc_mean", "final_auprc_std",
    "train_seconds_mean", "train_seconds_std", "train_seconds_max",
    "inference_seconds_mean", "inference_seconds_std", "inference_seconds_max",
    "total_seconds_mean", "total_seconds_std", "total_seconds_max",
    "peak_gpu_allocated_mib_mean", "peak_gpu_allocated_mib_std", "peak_gpu_allocated_mib_max",
    "peak_gpu_reserved_mib_mean", "peak_gpu_reserved_mib_std", "peak_gpu_reserved_mib_max",
    "wall_seconds", "physical_gpu", "gpu_description", "cuda_visible_devices",
    "gamma_normalize_bands", "gamma_input_tanh", "gamma_coefficient_tanh",
    "status", "oom_detected", "return_code", "started_at", "finished_at",
    "log_path", "working_dir", "command",
)
RUN_EFFICIENCY_FIELDS = (
    "dataset", "run_index", "train_seconds", "inference_seconds", "total_seconds",
    "peak_gpu_allocated_mib", "peak_gpu_reserved_mib", "physical_gpu", "gpu_description",
)

NUMBER = r"([-+]?\d+(?:\.\d+)?)"
FINAL_AUC_RE = re.compile(r"FINAL TESTING AUC:\s*" + NUMBER)
FINAL_AUC_STD_RE = re.compile(r"FINAL TESTING AUC std:\s*" + NUMBER)
FINAL_AUPRC_RE = re.compile(r"FINAL TESTING AUPRC:\s*" + NUMBER)
FINAL_AUPRC_STD_RE = re.compile(r"FINAL TESTING AUPRC std:\s*" + NUMBER)
EFFICIENCY_RE = re.compile(
    r"EFFICIENCY_RUN:\s*(\d+)\s+"
    r"TRAIN_SECONDS:\s*" + NUMBER + r"\s+"
    r"INFERENCE_SECONDS:\s*" + NUMBER + r"\s+"
    r"TOTAL_SECONDS:\s*" + NUMBER + r"\s+"
    r"PEAK_GPU_ALLOCATED_MIB:\s*" + NUMBER + r"\s+"
    r"PEAK_GPU_RESERVED_MIB:\s*" + NUMBER
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


def final_match(regex: re.Pattern[str], log: str) -> str:
    matches = regex.findall(log)
    return matches[-1] if matches else ""


def summarize(values: list[float], name: str) -> dict[str, str]:
    if not values:
        return {f"{name}_mean": "", f"{name}_std": "", f"{name}_max": ""}
    return {
        f"{name}_mean": f"{statistics.fmean(values):.6f}",
        f"{name}_std": f"{statistics.pstdev(values):.6f}",
        f"{name}_max": f"{max(values):.6f}",
    }


def parse_log(log: str) -> tuple[dict[str, str], list[dict[str, str]]]:
    efficiency_runs = []
    for match in EFFICIENCY_RE.finditer(log):
        run_index, train, inference, total, allocated, reserved = match.groups()
        efficiency_runs.append(
            {
                "run_index": run_index,
                "train_seconds": train,
                "inference_seconds": inference,
                "total_seconds": total,
                "peak_gpu_allocated_mib": allocated,
                "peak_gpu_reserved_mib": reserved,
            }
        )
    metrics = {
        "final_auc_mean": final_match(FINAL_AUC_RE, log),
        "final_auc_std": final_match(FINAL_AUC_STD_RE, log),
        "final_auprc_mean": final_match(FINAL_AUPRC_RE, log),
        "final_auprc_std": final_match(FINAL_AUPRC_STD_RE, log),
    }
    for field in (
        "train_seconds", "inference_seconds", "total_seconds",
        "peak_gpu_allocated_mib", "peak_gpu_reserved_mib",
    ):
        metrics.update(summarize([float(row[field]) for row in efficiency_runs], field))
    return metrics, efficiency_runs


def completed_matches_config(prior: dict, spec, config: FinalConfiguration, physical_gpu: str) -> bool:
    return prior.get("status") == "completed" and all(
        prior.get(field) == value
        for field, value in {
            "runs": str(RUNS),
            "hidden_dim": str(config.hidden_dim),
            "lr": str(config.lr),
            "epoch": str(config.epoch),
            "alpha": decimal(config.alpha),
            "beta": decimal(config.beta),
            "gamma": decimal(config.gamma),
            "gamma_centering": str(config.gamma_centering),
            "physical_gpu": physical_gpu,
            "gamma_normalize_bands": str(spec.gamma_normalize_bands),
            "gamma_input_tanh": str(spec.gamma_input_tanh),
            "gamma_coefficient_tanh": str(spec.gamma_coefficient_tanh),
        }.items()
    )


def run_dataset(
    dataset: str,
    *,
    data_dir: Path,
    results_root: Path,
    device: str,
    physical_gpu: str,
    gpu_description: str,
    resume: bool,
) -> dict:
    spec = resolve_dataset(dataset)
    config = FINAL_CONFIGURATIONS[spec.key]
    dataset_dir = results_root / safe_name(spec.cli_name)
    artifact_path = dataset_dir / "result.json"
    if resume and artifact_path.exists():
        try:
            prior = json.loads(artifact_path.read_text(encoding="utf-8"))
            if completed_matches_config(prior, spec, config, physical_gpu):
                print(f"[resume] {spec.cli_name}", flush=True)
                return prior
        except (OSError, json.JSONDecodeError):
            pass

    dataset_dir.mkdir(parents=True, exist_ok=True)
    log_path = dataset_dir / "run.log"
    command = [
        sys.executable,
        str(ROOT / "run.py"),
        "--dataset", spec.cli_name,
        "--data_dir", str(data_dir),
        "--hidden_dim", str(config.hidden_dim),
        "--lr", str(config.lr),
        "--epoch", str(config.epoch),
        "--alpha", decimal(config.alpha),
        "--beta", decimal(config.beta),
        "--gamma", decimal(config.gamma),
        "--gamma_centering", str(config.gamma_centering),
        "--runs", str(RUNS),
        "--tests", "1",
        "--device", device,
        *config.extra,
    ]
    command_text = subprocess.list2cmdline(command)
    started_at = now()
    started_clock = datetime.now(timezone.utc)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return_code = -1
    print(f"[start] {spec.cli_name}: runs={RUNS}, physical_gpu={physical_gpu}", flush=True)
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as handle:
            handle.write(
                f"Started UTC: {started_at}\n"
                f"Physical GPU: {physical_gpu}\n"
                f"GPU description: {gpu_description}\n"
                f"CUDA_VISIBLE_DEVICES: {cuda_visible}\n"
                f"Device argument: {device}\n"
                "Training-time AUC monitoring: enabled.\n"
                f"Command:\n{command_text}\n\n"
            )
            process = subprocess.run(
                command,
                cwd=dataset_dir,
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
    metrics, efficiency_runs = parse_log(log)
    oom_detected = bool(OOM_RE.search(log))
    wall_seconds = (datetime.now(timezone.utc) - started_clock).total_seconds()
    completed = (
        return_code == 0
        and bool(metrics["final_auc_mean"])
        and bool(metrics["final_auprc_mean"])
        and len(efficiency_runs) == RUNS
        and len({row["run_index"] for row in efficiency_runs}) == RUNS
    )
    status = "completed" if completed else "oom" if oom_detected else "failed"
    result = {
        "dataset": spec.cli_name,
        "runs": str(RUNS),
        "hidden_dim": str(config.hidden_dim),
        "lr": str(config.lr),
        "epoch": str(config.epoch),
        "alpha": decimal(config.alpha),
        "beta": decimal(config.beta),
        "gamma": decimal(config.gamma),
        "gamma_centering": str(config.gamma_centering),
        "physical_gpu": physical_gpu,
        "gpu_description": gpu_description,
        "cuda_visible_devices": cuda_visible,
        "gamma_normalize_bands": str(spec.gamma_normalize_bands),
        "gamma_input_tanh": str(spec.gamma_input_tanh),
        "gamma_coefficient_tanh": str(spec.gamma_coefficient_tanh),
        "status": status,
        "oom_detected": str(oom_detected),
        "return_code": str(return_code),
        "started_at": started_at,
        "finished_at": now(),
        "wall_seconds": f"{wall_seconds:.6f}",
        "log_path": str(log_path.relative_to(results_root)),
        "working_dir": str(dataset_dir.relative_to(results_root)),
        "command": command_text,
        "efficiency_runs": efficiency_runs,
        **metrics,
    }
    atomic_json(artifact_path, result)
    print(
        f"[done] {spec.cli_name}: status={status} "
        f"peak_gpu_allocated_mib={result['peak_gpu_allocated_mib_max'] or 'NA'}",
        flush=True,
    )
    return result


def load_result(dataset_dir: Path) -> dict | None:
    path = dataset_dir / "result.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def aggregate(results_root: Path, datasets: Iterable[str]) -> None:
    results = []
    efficiency_rows = []
    for dataset in datasets:
        spec = resolve_dataset(dataset)
        result = load_result(results_root / safe_name(spec.cli_name))
        if result is None:
            continue
        results.append(result)
        for run in result.get("efficiency_runs", []):
            efficiency_rows.append(
                {
                    "dataset": result.get("dataset", spec.cli_name),
                    "physical_gpu": result.get("physical_gpu", ""),
                    "gpu_description": result.get("gpu_description", ""),
                    **run,
                }
            )
    write_csv(results_root / "summary.csv", results, RESULT_FIELDS)
    write_csv(results_root / "efficiency_runs.csv", efficiency_rows, RUN_EFFICIENCY_FIELDS)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical_gpu", default="")
    parser.add_argument("--gpu_description", default="")
    parser.add_argument("--tsocial_beta", type=float)
    parser.add_argument("--tsocial_gamma", type=float)
    parser.add_argument(
        "--gamma_centering_override",
        type=int,
        choices=(0, 1),
        help="Force the fixed Gamma-centering switch for every requested dataset.",
    )
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--skip_aggregate", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run final configurations and write detection/efficiency summaries."""

    options = parse_args(argv)
    if options.tsocial_beta is not None or options.tsocial_gamma is not None:
        original = FINAL_CONFIGURATIONS["tsocial"]
        FINAL_CONFIGURATIONS["tsocial"] = replace(
            original,
            beta=(original.beta if options.tsocial_beta is None else options.tsocial_beta),
            gamma=(original.gamma if options.tsocial_gamma is None else options.tsocial_gamma),
        )
    if options.gamma_centering_override is not None:
        for key, configuration in tuple(FINAL_CONFIGURATIONS.items()):
            FINAL_CONFIGURATIONS[key] = replace(
                configuration,
                gamma_centering=options.gamma_centering_override,
            )
    datasets = tuple(resolve_dataset(dataset).cli_name for dataset in options.datasets)
    keys = [resolve_dataset(dataset).key for dataset in datasets]
    if len(keys) != len(set(keys)):
        raise ValueError("--datasets must not contain duplicates.")
    results_root = root_path(options.results_root)
    if options.dry_run:
        print(f"runs={RUNS}")
        for dataset in datasets:
            config = FINAL_CONFIGURATIONS[resolve_dataset(dataset).key]
            print(
                f"  {dataset}: hidden={config.hidden_dim}, lr={config.lr}, epoch={config.epoch}, "
                f"alpha={config.alpha}, beta={config.beta}, gamma={config.gamma}, extra={config.extra}",
            )
        print(f"total model runs={len(datasets) * RUNS}")
        return
    if options.aggregate_only:
        aggregate(results_root, datasets)
        return
    failures = []
    for dataset in datasets:
        result = run_dataset(
            dataset,
            data_dir=root_path(options.data_dir),
            results_root=results_root,
            device=options.device,
            physical_gpu=options.physical_gpu,
            gpu_description=options.gpu_description,
            resume=not options.no_resume,
        )
        if result["status"] == "failed":
            failures.append(dataset)
    if not options.skip_aggregate:
        aggregate(results_root, datasets)
    if failures:
        raise RuntimeError("Non-OOM failures: " + ", ".join(failures))


if __name__ == "__main__":
    main()
