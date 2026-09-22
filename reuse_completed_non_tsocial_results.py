"""Safely reuse completed non-T-Social MARGAD search artifacts.

The helper validates that a prior search used the same non-T-Social schedule
before copying only resumable artifacts (logs, JSON, and CSV), never model
checkpoints.  T-Social is deliberately excluded so a new T-Social policy can
run from scratch in the destination root.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from dataset_config import SUPPORTED_DATASETS, resolve_dataset


ROOT = Path(__file__).resolve().parent
TSOCIAL_KEY = "tsocial"
NON_TSOCIAL_DATASETS = tuple(
    dataset for dataset in SUPPORTED_DATASETS
    if resolve_dataset(dataset).key != TSOCIAL_KEY
)
EXPECTED_BETA = tuple(
    Decimal("0.05") + Decimal("0.05") * index for index in range(20)
)
EXPECTED_GAMMA = tuple(
    Decimal("0.05") + Decimal("0.05") * index for index in range(30)
)
EXPECTED_BETA_TEXT = {f"{value:.2f}" for value in EXPECTED_BETA}
EXPECTED_GAMMA_TEXT = {f"{value:.2f}" for value in EXPECTED_GAMMA}


def resolve_root(text: str) -> Path:
    path = Path(text)
    return (path if path.is_absolute() else ROOT / path).resolve()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unreadable JSON: {path}: {error}") from error


def as_text(value) -> str:
    return str(value).strip()


def validate_result(result: dict, path: Path, dataset: str) -> list[str]:
    """Return incompatibilities for one result artifact."""
    spec = resolve_dataset(dataset)
    expected_lr = "0.001" if spec.key == "elliptic" else "0.003"
    expected = {
        "dataset": spec.cli_name,
        "status": "completed",
        "hidden_dim": "64",
        "epoch": "100",
        "alpha": "1.00",
    }
    issues = []
    for field, value in expected.items():
        if as_text(result.get(field, "")) != value:
            issues.append(f"{path}: {field}={result.get(field)!r}, expected {value!r}")
    try:
        if abs(float(result.get("lr")) - float(expected_lr)) > 1e-12:
            issues.append(f"{path}: lr={result.get('lr')!r}, expected {expected_lr!r}")
    except (TypeError, ValueError):
        issues.append(f"{path}: invalid lr={result.get('lr')!r}")

    stage = as_text(result.get("stage", ""))
    beta = as_text(result.get("beta", ""))
    gamma = as_text(result.get("gamma", ""))
    if stage == "beta":
        if beta not in EXPECTED_BETA_TEXT or gamma != "0.00":
            issues.append(f"{path}: invalid beta-stage weights beta={beta}, gamma={gamma}")
    elif stage == "gamma":
        if beta not in EXPECTED_BETA_TEXT or gamma not in EXPECTED_GAMMA_TEXT:
            issues.append(f"{path}: invalid gamma-stage weights beta={beta}, gamma={gamma}")
    else:
        issues.append(f"{path}: unsupported stage {stage!r}")
    return issues


def validate_dataset(root: Path, dataset: str) -> tuple[Path, list[Path]]:
    """Validate one complete 110-trial non-T-Social result tree."""
    spec = resolve_dataset(dataset)
    dataset_dir = root / spec.cli_name
    trial_root = dataset_dir / "trials"
    paths = sorted(trial_root.glob("*/*/result.json"))
    issues = []
    if len(paths) != 110:
        issues.append(f"{dataset_dir}: found {len(paths)} result.json files, expected 110")

    beta_count = gamma_count = 0
    for path in paths:
        result = read_json(path)
        issues.extend(validate_result(result, path, dataset))
        stage = as_text(result.get("stage", ""))
        beta_count += stage == "beta"
        gamma_count += stage == "gamma"
    if beta_count != 20:
        issues.append(f"{dataset_dir}: found {beta_count} beta trials, expected 20")
    if gamma_count != 90:
        issues.append(f"{dataset_dir}: found {gamma_count} gamma trials, expected 90")
    if issues:
        raise RuntimeError("\n".join(issues))
    return dataset_dir, paths


def copy_without_checkpoints(source: Path, destination: Path) -> int:
    copied = 0
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() == ".pth":
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def destination_is_complete(root: Path, dataset: str) -> bool:
    directory = root / resolve_dataset(dataset).cli_name
    if not directory.exists():
        return False
    validate_dataset(root, dataset)
    return True


def main(argv: list[str] | None = None) -> None:
    """Validate source trials and copy only resumable non-checkpoint files."""

    parser = argparse.ArgumentParser(
        description="Reuse completed non-T-Social staged-search results safely."
    )
    parser.add_argument(
        "--source_root",
        default="beta_gamma_staged_top3_h64_e100_tsocial10_results",
    )
    parser.add_argument(
        "--destination_root",
        default="beta_gamma_staged_top3_h64_e100_tsocial10_results",
    )
    parser.add_argument("--dry_run", action="store_true")
    options = parser.parse_args(argv)
    source_root = resolve_root(options.source_root)
    destination_root = resolve_root(options.destination_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source result directory does not exist: {source_root}")
    if source_root == destination_root:
        raise ValueError("Source and destination result directories must differ.")

    validated = {}
    for dataset in NON_TSOCIAL_DATASETS:
        validated[dataset] = validate_dataset(source_root, dataset)
    print(
        "Validated 660 completed non-T-Social trials from "
        f"{source_root}",
        flush=True,
    )
    if options.dry_run:
        return

    destination_root.mkdir(parents=True, exist_ok=True)
    destination_state = {}
    for dataset in NON_TSOCIAL_DATASETS:
        destination_dir = destination_root / resolve_dataset(dataset).cli_name
        if destination_dir.exists() and any(destination_dir.iterdir()):
            if destination_is_complete(destination_root, dataset):
                destination_state[dataset] = "already_complete"
            else:
                raise RuntimeError(
                    f"Destination contains incomplete/incompatible artifacts: {destination_dir}. "
                    "Move it aside before reuse; it was not overwritten."
                )
        else:
            destination_state[dataset] = "copy"

    copied_summary = {}
    for dataset, (source_dir, _) in validated.items():
        if destination_state[dataset] == "already_complete":
            copied_summary[dataset] = "already complete; left unchanged"
            continue
        destination_dir = destination_root / resolve_dataset(dataset).cli_name
        copied_summary[dataset] = f"copied {copy_without_checkpoints(source_dir, destination_dir)} files"

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "datasets": list(NON_TSOCIAL_DATASETS),
        "policy": {
            "hidden_dim": 64,
            "epoch": 100,
            "lr": "0.003 except elliptic=0.001",
            "beta_trials": 20,
            "top_beta_count": 3,
            "gamma_trials": 30,
        },
        "copied": copied_summary,
        "checkpoints_copied": False,
    }
    (destination_root / "reused_non_tsocial_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for dataset, summary in copied_summary.items():
        print(f"{dataset}: {summary}", flush=True)


if __name__ == "__main__":
    main()
