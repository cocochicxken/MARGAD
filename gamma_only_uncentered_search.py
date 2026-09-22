"""Reuse top Global Deviation weights and search uncentered WaveShift weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from beta_gamma_weight_search import (
    BEST_FIELDS,
    GAMMA_VALUES,
    RESULT_FIELDS,
    atomic_json,
    choose,
    load_results,
    now,
    refresh_reports,
    resolve_dataset,
    root_path,
    run_trial,
    safe_name,
    write_csv,
)


DEFAULT_DATASETS = ("Facebook", "Reddit", "Amazon", "YelpChi", "elliptic", "tfinance")
DEFAULT_BETA_SOURCE_ROOT = "beta_gamma_staged_top3_h64_e100_tsocial10_results"
DEFAULT_SEARCH_ROOT = "non_tsocial_uncentered_gamma_search"


def load_reused_betas(source_root: Path, dataset: str) -> list[dict]:
    spec = resolve_dataset(dataset)
    path = source_root / safe_name(spec.cli_name) / "beta_selection.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Beta Top-3 file is missing for {spec.cli_name}: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Unreadable Beta selection: {path}") from error
    selected = payload.get("selected_betas", [])
    if len(selected) < 3:
        raise ValueError(f"Expected three completed Beta values in {path}.")
    selected = selected[:3]
    if any(row.get("status") != "completed" for row in selected):
        raise ValueError(f"Incomplete Beta selection in {path}.")
    return selected


def run_dataset(
    dataset: str,
    *,
    data_dir: Path,
    beta_source_root: Path,
    search_root: Path,
    device: str,
    resume: bool,
) -> None:
    spec = resolve_dataset(dataset)
    if spec.key == "tsocial":
        raise ValueError("This gamma-only control deliberately excludes T-Social.")
    selected_betas = load_reused_betas(beta_source_root, spec.cli_name)
    dataset_dir = search_root / safe_name(spec.cli_name)
    beta_values = [float(row["beta"]) for row in selected_betas]
    print(
        f"\n=== {spec.cli_name}: reuse Beta Top-3={beta_values}; "
        f"uncentered Gamma trials={len(beta_values) * len(GAMMA_VALUES)} ===",
        flush=True,
    )
    rows = [
        run_trial(
            spec.cli_name,
            "gamma",
            beta,
            gamma,
            data_dir=data_dir,
            search_root=search_root,
            device=device,
            resume=resume,
            gamma_centering=0,
        )
        for beta in beta_values
        for gamma in GAMMA_VALUES
    ]
    completed = [row for row in rows if row.get("status") == "completed"]
    expected = len(beta_values) * len(GAMMA_VALUES)
    if len(completed) != expected:
        raise RuntimeError(
            f"{spec.cli_name} Gamma search incomplete ({len(completed)}/{expected}). "
            "Inspect the failed logs and rerun the same command to resume."
        )
    best_gamma = choose(rows, "final_auc")
    if best_gamma is None:
        raise RuntimeError(f"No completed Gamma trial for {spec.cli_name}.")
    atomic_json(
        dataset_dir / "reused_beta_selection.json",
        {
            "source_root": str(beta_source_root),
            "source_file": str(beta_source_root / safe_name(spec.cli_name) / "beta_selection.json"),
            "reused_at": now(),
            "selected_betas": selected_betas,
            "reason": "Gamma=0 during Beta search, so the Gamma-centering switch has no effect.",
        },
    )
    atomic_json(
        dataset_dir / "staged_selection.json",
        {
            "selected_by": "final_auc",
            "gamma_centering": 0,
            "selected_betas": selected_betas,
            "best_gamma": best_gamma,
        },
    )
    refresh_reports(dataset_dir)
    selections = [
        {"dataset": spec.cli_name, "selection": f"reused_beta_rank_{rank}", **row}
        for rank, row in enumerate(selected_betas, start=1)
    ]
    selections.append({"dataset": spec.cli_name, "selection": "selected_gamma_by_final_auc", **best_gamma})
    write_csv(
        dataset_dir / f"best_configurations_{safe_name(spec.cli_name)}.csv",
        selections,
        BEST_FIELDS,
    )
    print(
        f"[select] {spec.cli_name}: beta={best_gamma['beta']} "
        f"gamma={best_gamma['gamma']} final_auc={best_gamma['final_auc']}",
        flush=True,
    )


def aggregate(search_root: Path, datasets: tuple[str, ...]) -> None:
    all_rows, selected_rows = [], []
    for dataset in datasets:
        directory = search_root / safe_name(resolve_dataset(dataset).cli_name)
        rows = load_results(directory)
        all_rows.extend(rows)
        selection_path = directory / f"best_configurations_{safe_name(resolve_dataset(dataset).cli_name)}.csv"
        if selection_path.is_file():
            # Per-dataset CSV remains the authoritative selected-configuration file.
            selected_rows.append({"dataset": resolve_dataset(dataset).cli_name, "file": str(selection_path)})
    write_csv(search_root / "all_gamma_trials.csv", all_rows, RESULT_FIELDS)
    atomic_json(search_root / "search_manifest.json", {
        "datasets": list(datasets),
        "gamma_centering": 0,
        "gamma_values": GAMMA_VALUES,
        "top_betas_per_dataset": 3,
        "selected_configuration_files": selected_rows,
    })


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--beta_source_root", default=DEFAULT_BETA_SOURCE_ROOT)
    parser.add_argument("--search_root", default=DEFAULT_SEARCH_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Launch the uncentered WaveShift search from validated prior results."""

    options = parse_args(argv)
    datasets = tuple(resolve_dataset(item).cli_name for item in options.datasets)
    keys = [resolve_dataset(item).key for item in datasets]
    if "tsocial" in keys or len(keys) != len(set(keys)):
        raise ValueError("Use each non-T-Social dataset at most once.")
    beta_source_root = root_path(options.beta_source_root).resolve()
    search_root = root_path(options.search_root).resolve()
    if options.dry_run:
        print(f"Gamma values={len(GAMMA_VALUES)}; planned trials={len(datasets) * 3 * len(GAMMA_VALUES)}")
        for dataset in datasets:
            betas = [row["beta"] for row in load_reused_betas(beta_source_root, dataset)]
            print(f"  {dataset}: reused_beta_top3={betas}; gamma_trials={3 * len(GAMMA_VALUES)}")
        return
    data_dir = root_path(options.data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    for dataset in datasets:
        run_dataset(
            dataset,
            data_dir=data_dir,
            beta_source_root=beta_source_root,
            search_root=search_root,
            device=options.device,
            resume=not options.no_resume,
        )
    aggregate(search_root, datasets)


if __name__ == "__main__":
    main()
