"""Resumable launcher for the final-configuration TA-DiffRef A0--A6 study."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ta_diffref_ablation_config import (
    EXPERIMENT_VERSION,
    operator_protocol,
    parse_seeds,
    select_datasets,
    select_variants,
)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def _run_directory(root: Path, dataset_key: str, variant_code: str, seed: int) -> Path:
    return root / "runs" / dataset_key / variant_code / f"seed_{seed:02d}"


def _command(args, source_root: Path, spec, variant, seed: int, result_path: Path):
    command = [
        str(args.python_exe),
        str(source_root / "run.py"),
        "--dataset", spec.cli_name,
        "--data_dir", str(Path(args.data_dir).resolve()),
        "--device", args.device,
        "--runs", "1",
        "--tests", "1",
        "--seed_offset", str(seed),
        "--hidden_dim", str(spec.hidden_dim),
        "--lr", str(spec.learning_rate),
        "--epoch", str(spec.epochs),
        "--alpha", "1.0",
        "--beta", str(spec.global_deviation_weight),
        "--gamma", str(spec.waveshift_weight),
        "--alpha_mode", variant.alpha_mode,
        "--gamma_mode", "full",
        "--gamma_centering", str(spec.gamma_centering),
        "--result_json", str(result_path),
    ]
    if spec.key == "tsocial":
        command.extend((
            "--batch_size", "51200",
            "--eval_batch_size", "51200",
            "--batch_fanout", "8",
            "--num_workers", "0",
            "--dgl_graph_on_gpu", "1",
        ))
    return command


def _valid_completed(run_dir: Path, spec, variant, seed: int) -> bool:
    completed_path = run_dir / "completed.json"
    result_path = run_dir / "result.json"
    if not completed_path.is_file() or not result_path.is_file():
        return False
    try:
        completed = _read_json(completed_path)
        result = _read_json(result_path)
    except (OSError, json.JSONDecodeError):
        return False
    if completed.get("experiment_version") != EXPERIMENT_VERSION:
        return False
    if (
        result.get("status") != "completed"
        or result.get("dataset_key") != spec.key
        or result.get("variant") != variant.code
        or result.get("seed") != seed
    ):
        return False
    metrics = result.get("metrics", {})
    try:
        return math.isfinite(float(metrics["auc"])) and math.isfinite(float(metrics["auprc"]))
    except (KeyError, TypeError, ValueError):
        return False


def _next_attempt(run_dir: Path) -> int:
    attempt = 1
    while (run_dir / f"core_result_attempt_{attempt:02d}.json").exists() or (
        run_dir / f"failure_attempt_{attempt:02d}.json"
    ).exists():
        attempt += 1
    return attempt


def _run_one(args, source_root: Path, root: Path, spec, variant, seed: int) -> str:
    run_dir = _run_directory(root, spec.key, variant.code, seed)
    if _valid_completed(run_dir, spec, variant, seed):
        if args.resume:
            print(f"SKIP {spec.cli_name} {variant.code} seed={seed}", flush=True)
            return "skipped"
        print(f"REFUSE completed run without --resume: {run_dir}", flush=True)
        return "blocked"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        print(f"REFUSE existing incomplete run without --resume: {run_dir}", flush=True)
        return "blocked"

    run_dir.mkdir(parents=True, exist_ok=True)
    attempt = _next_attempt(run_dir)
    core_result = run_dir / f"core_result_attempt_{attempt:02d}.json"
    log_path = run_dir / "run.log"
    command = _command(args, source_root, spec, variant, seed, core_result)
    started_at = _timestamp()
    protocol = operator_protocol(spec, variant)
    _write_json(run_dir / "protocol.json", {
        "experiment_version": EXPERIMENT_VERSION,
        "dataset": spec.cli_name,
        "dataset_key": spec.key,
        "variant": variant.code,
        "alpha_mode": variant.alpha_mode,
        "reference_operator": variant.reference_operator,
        "seed": seed,
        "operator_protocol": protocol,
        "command": command,
    })

    print(f"RUN {spec.cli_name} {variant.code} seed={seed}: {run_dir}", flush=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(f"\n===== attempt {attempt} {started_at} =====\n")
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        try:
            completed_process = subprocess.run(
                command,
                cwd=run_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                timeout=None if args.timeout_hours == 0 else args.timeout_hours * 3600,
                check=False,
            )
            returncode = completed_process.returncode
        except subprocess.TimeoutExpired:
            failure = {
                "status": "timeout",
                "timestamp": _timestamp(),
                "timeout_hours": args.timeout_hours,
                "command": command,
            }
            _write_json(run_dir / f"failure_attempt_{attempt:02d}.json", failure)
            print(f"TIMEOUT {spec.cli_name} {variant.code} seed={seed}", flush=True)
            return "failed"
        except OSError as error:
            failure = {
                "status": "launch_error",
                "timestamp": _timestamp(),
                "error": repr(error),
                "command": command,
            }
            _write_json(run_dir / f"failure_attempt_{attempt:02d}.json", failure)
            print(f"LAUNCH ERROR {spec.cli_name} {variant.code} seed={seed}", flush=True)
            return "failed"

    if returncode != 0 or not core_result.is_file():
        _write_json(run_dir / f"failure_attempt_{attempt:02d}.json", {
            "status": "failed",
            "timestamp": _timestamp(),
            "returncode": returncode,
            "core_result_present": core_result.is_file(),
            "command": command,
        })
        print(f"FAILED {spec.cli_name} {variant.code} seed={seed}; see {log_path}", flush=True)
        return "failed"

    try:
        core = _read_json(core_result)
        runs = core.get("runs", [])
        if len(runs) != 1:
            raise ValueError("Expected exactly one run result.")
        auc = float(runs[0]["auc"])
        auprc = float(runs[0]["auprc"])
        if not math.isfinite(auc) or not math.isfinite(auprc):
            raise ValueError("AUROC/AUPRC is non-finite.")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        _write_json(run_dir / f"failure_attempt_{attempt:02d}.json", {
            "status": "invalid_or_nonfinite_result",
            "timestamp": _timestamp(),
            "error": str(error),
            "core_result_file": core_result.name,
        })
        print(f"INVALID {spec.cli_name} {variant.code} seed={seed}; see {log_path}", flush=True)
        return "failed"

    result = {
        "schema_version": 1,
        "experiment_version": EXPERIMENT_VERSION,
        "status": "completed",
        "completed_at": _timestamp(),
        "dataset": spec.cli_name,
        "dataset_key": spec.key,
        "variant": variant.code,
        "alpha_mode": variant.alpha_mode,
        "reference_operator": variant.reference_operator,
        "seed": seed,
        "fixed_configuration": {
            "hidden_dim": spec.hidden_dim,
            "learning_rate": spec.learning_rate,
            "epochs": spec.epochs,
            "lambda_ta_diffref": 1.0,
            "lambda_global_deviation": spec.global_deviation_weight,
            "lambda_waveshift": spec.waveshift_weight,
            "gamma_centering": spec.gamma_centering,
        },
        "operator_protocol": protocol,
        "metrics": runs[0],
        "core_result_file": core_result.name,
        "log_file": log_path.name,
    }
    _write_json(run_dir / "result.json", result)
    _write_json(run_dir / "completed.json", {
        "schema_version": 1,
        "experiment_version": EXPERIMENT_VERSION,
        "status": "completed",
        "completed_at": result["completed_at"],
        "attempt": attempt,
        "result_file": "result.json",
    })
    return "completed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=("all",))
    parser.add_argument("--variants", nargs="*", default=("all",))
    parser.add_argument("--seeds", nargs="*", default=("0-4",))
    parser.add_argument("--output_dir", default="return/ta_diffref_mechanism_ablation")
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--python_exe", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout_hours", type=float, default=72.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout_hours < 0:
        raise ValueError("--timeout_hours must be non-negative; 0 disables the timeout.")
    datasets = select_datasets(args.datasets)
    variants = select_variants(args.variants)
    seeds = parse_seeds(args.seeds)
    root = Path(args.output_dir).resolve()
    source_root = Path(__file__).resolve().parent
    total = len(datasets) * len(variants) * len(seeds)
    print(
        f"TA-DiffRef A0-A6: datasets={len(datasets)} variants={len(variants)} "
        f"seeds={len(seeds)} total={total} device={args.device}",
        flush=True,
    )
    if args.dry_run:
        for spec in datasets:
            for variant in variants:
                print(
                    f"{spec.cli_name:>10} {variant.code} {variant.alpha_mode:<45} "
                    f"seeds={seeds[0]}..{seeds[-1]}",
                    flush=True,
                )
        return 0

    status_counts: dict[str, int] = {}
    for spec in datasets:
        for variant in variants:
            for seed in seeds:
                status = _run_one(args, source_root, root, spec, variant, seed)
                status_counts[status] = status_counts.get(status, 0) + 1
    print("STATUS " + json.dumps(status_counts, sort_keys=True), flush=True)
    return 1 if status_counts.get("failed", 0) or status_counts.get("blocked", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
