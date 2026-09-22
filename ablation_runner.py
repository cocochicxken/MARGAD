"""Resumable per-seed runner for the 19 MARGAD branch ablations.

The module purposely does not import torch, DGL, or the training entry point.
Consequently ``--dry-run`` verifies the full matrix and its output paths on a
bare Windows control machine before a GPU job is launched.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ablation_config import (
    AblationVariant,
    matrix_rows,
    parse_seeds,
    selected_specs,
    selected_variants,
)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_directory(root: Path, variant: AblationVariant, dataset_key: str, seed: int) -> Path:
    return root / "runs" / variant.study / variant.code / dataset_key / f"seed_{seed:02d}"


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def build_parser() -> argparse.ArgumentParser:
    """Build the ablation-matrix launcher argument parser."""

    parser = argparse.ArgumentParser(
        description="Run the 19-config three-loss ablation matrix with isolated seed directories."
    )
    parser.add_argument("--studies", nargs="+", default=("all",), help="loss alpha gamma, or all")
    parser.add_argument(
        "--variants", nargs="*",
        help="optional exact IDs (for example L6 or A0,A1); applied after --studies",
    )
    parser.add_argument("--datasets", nargs="*", help="dataset names, comma-separated names, or all")
    parser.add_argument("--seeds", nargs="*", default=("0-4",), help="e.g. 0-4 or 0,1,4")
    parser.add_argument("--resume", action="store_true", help="skip atomically completed seeds; explicitly retry incomplete seeds")
    parser.add_argument("--dry-run", action="store_true", help="print matrix and paths without importing/running torch")
    parser.add_argument("--output_dir", default="three_loss_ablation_results")
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--python_exe", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epoch_override", type=int, help="optional smoke-test epoch budget")
    parser.add_argument("--beta_override", type=float, help="replace the active Beta weight")
    parser.add_argument("--gamma_override", type=float, help="replace the active Gamma weight")
    parser.add_argument(
        "--timeout_hours",
        type=float,
        default=72.0,
        help="hard per-seed wall-clock limit; 0 disables it (default: 72)",
    )
    parser.add_argument(
        "--keep_monitor_auc",
        action="store_true",
        help="leave historical label-only monitor logging on; never affects checkpoint selection",
    )
    parser.add_argument("--no_aggregate", action="store_true", help="do not refresh CSV summaries at the end")
    return parser


def _weights(args, variant: AblationVariant, spec) -> dict[str, float]:
    weights = variant.weight_overrides(spec)
    if "beta" in variant.active_losses and args.beta_override is not None:
        weights["beta"] = float(args.beta_override)
    if "gamma" in variant.active_losses and args.gamma_override is not None:
        weights["gamma"] = float(args.gamma_override)
    return weights


def _command(
    args,
    source_root: Path,
    variant: AblationVariant,
    spec,
    seed: int,
    core_result: Path,
    diagnostics: Path,
) -> list[str]:
    weights = _weights(args, variant, spec)
    gamma_centering = "0" if spec.key == "tsocial" else "1"
    command = [
        str(args.python_exe),
        str(source_root / "run.py"),
        "--dataset", spec.cli_name,
        "--data_dir", str(Path(args.data_dir).resolve()),
        "--device", args.device,
        "--runs", "1",
        "--seed_offset", str(seed),
        "--alpha", str(weights["alpha"]),
        "--beta", str(weights["beta"]),
        "--gamma", str(weights["gamma"]),
        "--alpha_mode", variant.alpha_mode,
        "--gamma_mode", variant.gamma_mode,
        "--gamma_centering", gamma_centering,
        "--result_json", str(core_result),
        "--diagnostics_json", str(diagnostics),
    ]
    if args.epoch_override is not None:
        command.extend(("--epoch", str(args.epoch_override)))
    if not args.keep_monitor_auc:
        command.append("--disable_monitor_auc")
    return command


def _run_one(args, source_root: Path, root: Path, variant: AblationVariant, spec, seed: int) -> str:
    run_dir = _run_directory(root, variant, spec.key, seed)
    completed = run_dir / "completed.json"
    if completed.is_file():
        if args.resume:
            print(f"SKIP completed {run_dir}", flush=True)
            return "skipped"
        print(f"REFUSE existing completed run without --resume: {run_dir}", flush=True)
        return "blocked"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        # A failed run is evidence.  Do not erase its log/checkpoint merely
        # because a non-resume invocation was issued later.
        print(f"REFUSE incomplete existing run without --resume: {run_dir}", flush=True)
        return "blocked"

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    attempt = 1
    while any((run_dir / f"{stem}_attempt_{attempt:02d}.json").exists() for stem in (
        "core_result", "diagnostics", "failure", "heartbeat",
    )):
        attempt += 1
    core_result = run_dir / f"core_result_attempt_{attempt:02d}.json"
    diagnostics = run_dir / f"diagnostics_attempt_{attempt:02d}.json"
    heartbeat = run_dir / f"heartbeat_attempt_{attempt:02d}.json"
    command = _command(args, source_root, variant, spec, seed, core_result, diagnostics)
    print(f"RUN {variant.code} {spec.cli_name} seed={seed}: {run_dir}", flush=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(f"\n===== attempt {attempt} {_timestamp()} =====\n")
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        try:
            process = subprocess.Popen(
                command,
                cwd=run_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
        except OSError as error:
            _write_json_atomic(run_dir / f"failure_attempt_{attempt:02d}.json", {
                "status": "launch_error",
                "timestamp": _timestamp(),
                "error": repr(error),
                "command": command,
            })
            return "failed"
        started = time.monotonic()
        timeout_seconds = float(args.timeout_hours) * 3600.0
        timed_out = False
        next_heartbeat = 0.0
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed >= next_heartbeat:
                _write_json_atomic(heartbeat, {
                    "status": "running",
                    "timestamp": _timestamp(),
                    "attempt": attempt,
                    "pid": process.pid,
                    "elapsed_seconds": round(elapsed, 3),
                    "timeout_hours": args.timeout_hours,
                })
                next_heartbeat = elapsed + 30.0
            if timeout_seconds > 0.0 and elapsed >= timeout_seconds:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=60)
                break
            # Heartbeat cadence is intentionally independent of loss values;
            # a run is never stopped for a plateau or slow improvement.
            time.sleep(5.0)
        returncode = process.returncode
        _write_json_atomic(heartbeat, {
            "status": "timed_out" if timed_out else "finished",
            "timestamp": _timestamp(),
            "attempt": attempt,
            "pid": process.pid,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "returncode": returncode,
            "timeout_hours": args.timeout_hours,
        })
    if timed_out:
        _write_json_atomic(run_dir / f"failure_attempt_{attempt:02d}.json", {
            "status": "timeout",
            "timestamp": _timestamp(),
            "returncode": returncode,
            "timeout_hours": args.timeout_hours,
            "command": command,
        })
        print(f"TIMEOUT {variant.code} {spec.cli_name} seed={seed}; see {log_path}", flush=True)
        return "failed"
    if returncode != 0 or not core_result.is_file() or not diagnostics.is_file():
        _write_json_atomic(run_dir / f"failure_attempt_{attempt:02d}.json", {
            "status": "failed",
            "timestamp": _timestamp(),
            "returncode": returncode,
            "core_result_present": core_result.is_file(),
            "diagnostics_present": diagnostics.is_file(),
            "command": command,
        })
        print(f"FAILED {variant.code} {spec.cli_name} seed={seed}; see {log_path}", flush=True)
        return "failed"

    core = _read_json(core_result)
    diagnostic = _read_json(diagnostics)
    runs = core.get("runs", [])
    if len(runs) != 1:
        _write_json_atomic(run_dir / f"failure_attempt_{attempt:02d}.json", {
            "status": "invalid_core_result",
            "timestamp": _timestamp(),
            "message": "Expected exactly one --runs=1 result.",
        })
        return "failed"
    weights = _weights(args, variant, spec)
    result = {
        "schema_version": 1,
        "status": "completed",
        "completed_at": _timestamp(),
        "study": variant.study,
        "variant": variant.code,
        "variant_description": variant.description,
        "dataset": spec.cli_name,
        "dataset_key": spec.key,
        "seed": seed,
        "active_losses": list(variant.active_losses),
        "weights": weights,
        "alpha_mode": variant.alpha_mode,
        "gamma_mode": variant.gamma_mode,
        "gamma_centering": 0 if spec.key == "tsocial" else 1,
        "epoch_override": args.epoch_override,
        "metrics": runs[0],
        "core_result_file": core_result.name,
        "diagnostics_file": diagnostics.name,
        "score_only_combinations": diagnostic.get("score_only_combinations", {}),
    }
    _write_json_atomic(run_dir / "result.json", result)
    _write_json_atomic(completed, {
        "schema_version": 1,
        "status": "completed",
        "completed_at": result["completed_at"],
        "result_file": "result.json",
        "attempt": attempt,
    })
    return "completed"


def _print_dry_run(args, root: Path, variants, specs, seeds) -> None:
    print("Three-loss ablation dry run (no torch/DGL import)")
    print(f"output_root: {root}")
    print(f"variants={len(variants)}, datasets={len(specs)}, seeds={len(seeds)}, total_runs={len(variants) * len(specs) * len(seeds)}")
    print("matrix:")
    for row in matrix_rows():
        if any(variant.code == row["code"] for variant in variants):
            print(
                f"  {row['code']:>2} [{row['study']}] losses={row['losses']} "
                f"alpha={row['alpha_mode']} gamma={row['gamma_mode']}"
            )
    first = _run_directory(root, variants[0], specs[0].key, seeds[0])
    last = _run_directory(root, variants[-1], specs[-1].key, seeds[-1])
    print(f"first_seed_directory: {first}")
    print(f"last_seed_directory:  {last}")
    monitor = "enabled" if args.keep_monitor_auc else "disabled"
    print(f"checkpoint policy: unsupervised minimum total training loss after the midpoint; monitor AUC {monitor}.")


def _filter_variants(variants: tuple[AblationVariant, ...], requested) -> tuple[AblationVariant, ...]:
    if not requested:
        return variants
    tokens: list[str] = []
    for item in requested:
        tokens.extend(part.strip() for part in item.split(",") if part.strip())
    if not tokens or any(token.lower() == "all" for token in tokens):
        return variants
    lookup = {variant.code.lower(): variant for variant in variants}
    unknown = [token for token in tokens if token.lower() not in lookup]
    if unknown:
        raise ValueError(
            f"Unknown/filtered-out variant(s): {', '.join(unknown)}. "
            f"Available after --studies: {', '.join(variant.code for variant in variants)}"
        )
    selected = {token.lower() for token in tokens}
    return tuple(variant for variant in variants if variant.code.lower() in selected)


def main(argv: list[str] | None = None) -> int:
    """Validate, optionally preview, and execute the requested run matrix."""

    args = build_parser().parse_args(argv)
    if args.epoch_override is not None and args.epoch_override <= 0:
        raise ValueError("--epoch_override must be positive.")
    if args.beta_override is not None and args.beta_override < 0.0:
        raise ValueError("--beta_override must be non-negative.")
    if args.gamma_override is not None and args.gamma_override < 0.0:
        raise ValueError("--gamma_override must be non-negative.")
    if args.timeout_hours < 0:
        raise ValueError("--timeout_hours must be non-negative (0 disables it).")
    variants = _filter_variants(selected_variants(args.studies), args.variants)
    specs = selected_specs(args.datasets)
    seeds = parse_seeds(args.seeds)
    if not variants or not specs or not seeds:
        raise ValueError("The selected matrix is empty.")
    source_root = Path(__file__).resolve().parent
    root = Path(args.output_dir).resolve()
    if args.dry_run:
        _print_dry_run(args, root, variants, specs, seeds)
        return 0

    statuses: dict[str, int] = {}
    for variant in variants:
        for spec in specs:
            for seed in seeds:
                status = _run_one(args, source_root, root, variant, spec, seed)
                statuses[status] = statuses.get(status, 0) + 1
    print("Run status counts: " + ", ".join(f"{name}={count}" for name, count in sorted(statuses.items())), flush=True)
    if not args.no_aggregate:
        from ablation_aggregate import aggregate

        aggregate(root)
    return 1 if statuses.get("failed", 0) or statuses.get("blocked", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
