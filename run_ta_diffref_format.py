"""Run all six TA-DiffRef formats with ten seeds each."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from ta_diffref_format_config import (
    EXPERIMENT_VERSION, operator_protocol, parse_seeds, select_datasets,
    select_run_variants,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _completed(run_dir: Path, spec, variant, seed: int) -> bool:
    result_path = run_dir / "result.json"
    if not (run_dir / "completed.json").is_file() or not result_path.is_file():
        return False
    try:
        result = _read_json(result_path)
        metrics = result["metrics"]
        return (
            result["experiment_version"] == EXPERIMENT_VERSION
            and result["dataset_key"] == spec.key
            and result["variant"] == variant.code
            and result["seed"] == seed
            and math.isfinite(float(metrics["auc"]))
            and math.isfinite(float(metrics["auprc"]))
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _command(args, source_root: Path, spec, variant, seed: int, core_path: Path) -> list[str]:
    command = [
        str(args.python_exe), str(source_root / "run.py"),
        "--dataset", spec.cli_name,
        "--data_dir", str(Path(args.data_dir).resolve()),
        "--device", args.device,
        "--runs", "1", "--tests", "1", "--seed_offset", str(seed),
        "--hidden_dim", str(spec.hidden_dim), "--lr", str(spec.learning_rate),
        "--epoch", str(spec.epochs),
        "--alpha", "1.0", "--beta", str(spec.global_deviation_weight),
        "--gamma", str(spec.waveshift_weight),
        "--alpha_mode", variant.alpha_mode,
        "--gamma_mode", "full", "--gamma_centering", str(spec.gamma_centering),
        "--result_json", str(core_path),
    ]
    if spec.key == "tsocial":
        command.extend((
            "--batch_size", "51200", "--eval_batch_size", "51200",
            "--batch_fanout", "8", "--num_workers", "0",
            "--dgl_graph_on_gpu", "1",
        ))
    return command


def _run_one(args, root: Path, source_root: Path, spec, variant, seed: int) -> str:
    run_dir = root / "runs" / spec.key / variant.code / f"seed_{seed:02d}"
    if _completed(run_dir, spec, variant, seed):
        print(f"SKIP {spec.cli_name} {variant.code} seed={seed}", flush=True)
        return "skipped"
    run_dir.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (run_dir / f"core_result_attempt_{attempt:02d}.json").exists() or (
        run_dir / f"failure_attempt_{attempt:02d}.json"
    ).exists():
        attempt += 1
    core_path = run_dir / f"core_result_attempt_{attempt:02d}.json"
    command = _command(args, source_root, spec, variant, seed, core_path)
    _write_json(run_dir / "protocol.json", {
        "experiment_version": EXPERIMENT_VERSION,
        "dataset": spec.cli_name, "dataset_key": spec.key,
        "variant": variant.code, "alpha_mode": variant.alpha_mode,
        "reference_operator": variant.reference_operator,
        "calibration": variant.calibration,
        "target_return": variant.target_return,
        "integration": variant.integration,
        "seed": seed, "operator_protocol": operator_protocol(spec, variant),
        "command": command,
    })
    print(f"RUN {spec.cli_name} {variant.code} seed={seed}", flush=True)
    log_path = run_dir / "run.log"
    try:
        with log_path.open("a", encoding="utf-8", newline="\n") as log:
            log.write(f"\n===== attempt {attempt} {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
            log.flush()
            process = subprocess.run(
                command, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT,
                timeout=None if args.timeout_hours == 0 else args.timeout_hours * 3600,
                check=False,
            )
        if process.returncode != 0 or not core_path.is_file():
            raise RuntimeError(f"returncode={process.returncode}; result_present={core_path.is_file()}")
        core = _read_json(core_path)
        runs = core["runs"]
        if len(runs) != 1:
            raise ValueError("Expected one core run")
        metrics = runs[0]
        if not all(math.isfinite(float(metrics[key])) for key in ("auc", "auprc")):
            raise ValueError("Non-finite AUROC/AUPRC")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError,
            json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        _write_json(run_dir / f"failure_attempt_{attempt:02d}.json", {
            "status": "failed", "error": str(error), "command": command,
        })
        print(f"FAILED {spec.cli_name} {variant.code} seed={seed}: {log_path}", flush=True)
        return "failed"
    result = {
        "experiment_version": EXPERIMENT_VERSION, "status": "completed",
        "dataset": spec.cli_name, "dataset_key": spec.key,
        "variant": variant.code, "alpha_mode": variant.alpha_mode, "seed": seed,
        "fixed_configuration": {
            "hidden_dim": spec.hidden_dim, "lr": spec.learning_rate,
            "epoch": spec.epochs, "alpha": 1.0,
            "beta": spec.global_deviation_weight,
            "gamma": spec.waveshift_weight,
            "gamma_centering": spec.gamma_centering,
        },
        "metrics": metrics, "core_result_file": core_path.name,
        "checkpoint": metrics.get("checkpoint"),
    }
    _write_json(run_dir / "result.json", result)
    _write_json(run_dir / "completed.json", {
        "experiment_version": EXPERIMENT_VERSION, "status": "completed",
        "result_file": "result.json", "attempt": attempt,
    })
    return "completed"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=("all",))
    parser.add_argument("--variants", nargs="*", default=("all",))
    parser.add_argument("--seeds", nargs="*", default=("0-9",))
    parser.add_argument("--output_dir", default="return/ta_diffref_format_final_f0_f5_10seed")
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--python_exe", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout_hours", type=float, default=72.0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    datasets = select_datasets(args.datasets)
    variants = select_run_variants(args.variants)
    seeds = parse_seeds(args.seeds)
    if args.timeout_hours < 0:
        parser.error("--timeout_hours must be non-negative")
    print(
        f"TA-DiffRef F0-F5: datasets={len(datasets)} variants={len(variants)} "
        f"seeds={len(seeds)} training_runs={len(datasets) * len(variants) * len(seeds)}",
        flush=True,
    )
    if args.dry_run:
        for spec in datasets:
            for variant in variants:
                print(
                    f"{spec.cli_name:>10} {variant.code} {variant.alpha_mode:<50} "
                    f"gamma_centering={spec.gamma_centering} seeds={seeds[0]}..{seeds[-1]}",
                    flush=True,
                )
        return 0
    root = Path(args.output_dir).resolve()
    source_root = Path(__file__).resolve().parent
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    for spec in datasets:
        for variant in variants:
            for seed in seeds:
                status = _run_one(args, root, source_root, spec, variant, seed)
                counts[status] += 1
    print("STATUS " + json.dumps(counts, sort_keys=True), flush=True)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
