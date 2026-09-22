"""Declarative experiment matrix for the three-branch MARGAD ablations.

This module deliberately has no torch/DGL imports so that ``--dry-run`` and
the matrix tests work on a machine that only has a Python interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from dataset_config import DATASET_SPECS, DatasetSpec, resolve_dataset


LOSS_NAMES = ("alpha", "beta", "gamma")
ALPHA_MODES = (
    "full",
    "one_hop",
    "anonymous_two_hop",
    "fixed_equal_multiscale",
    "learned_no_coefficient_normalization",
    "learned_no_target_anonymization",
    "learned_no_degree_or_path_volume_normalization",
    "paper_matched_volume",
)
GAMMA_MODES = (
    "full",
    "fixed_shared_wavelet",
    "learned_global_shared_only",
    "learned_channel_specific_only",
    "dual_independent",
    "uncentered_cross_response",
)


@dataclass(frozen=True)
class AblationVariant:
    """One independently trained configuration in the publication matrix."""

    code: str
    study: str
    description: str
    active_losses: tuple[str, ...]
    alpha_mode: str = "full"
    gamma_mode: str = "full"

    @property
    def is_full_loss(self) -> bool:
        return self.active_losses == LOSS_NAMES

    def weight_overrides(self, spec: DatasetSpec) -> dict[str, float]:
        defaults = spec.defaults
        return {
            name: float(getattr(defaults, name)) if name in self.active_losses else 0.0
            for name in LOSS_NAMES
        }


# L6 is intentionally present only once.  Alpha and Gamma mechanism studies
# reference it as their common full-method baseline instead of retraining it.
LOSS_VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant("L0", "loss", "alpha only", ("alpha",)),
    AblationVariant("L1", "loss", "beta only", ("beta",)),
    AblationVariant("L2", "loss", "gamma only", ("gamma",)),
    AblationVariant("L3", "loss", "alpha + beta (no gamma)", ("alpha", "beta")),
    AblationVariant("L4", "loss", "alpha + gamma (no beta)", ("alpha", "gamma")),
    AblationVariant("L5", "loss", "beta + gamma (no alpha)", ("beta", "gamma")),
    AblationVariant("L6", "loss", "full alpha + beta + gamma", LOSS_NAMES),
)

ALPHA_VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant("A0", "alpha", "fixed one-hop random-walk affinity", LOSS_NAMES, "one_hop"),
    AblationVariant("A1", "alpha", "fixed anonymous two-hop random-walk affinity", LOSS_NAMES, "anonymous_two_hop"),
    AblationVariant("A2", "alpha", "fixed equal one/two-hop anonymous affinity", LOSS_NAMES, "fixed_equal_multiscale"),
    AblationVariant("A3", "alpha", "learned affinity without coefficient-sum normalization", LOSS_NAMES, "learned_no_coefficient_normalization"),
    AblationVariant("A4", "alpha", "learned affinity without target anonymization", LOSS_NAMES, "learned_no_target_anonymization"),
    AblationVariant("A5", "alpha", "learned raw sampled/path-volume affinity without normalization", LOSS_NAMES, "learned_no_degree_or_path_volume_normalization"),
    AblationVariant("A6", "alpha", "coefficient-matched raw A/A^2 volume normalization", LOSS_NAMES, "paper_matched_volume"),
)

GAMMA_VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant("G1", "gamma", "fixed shared wavelet centered deviation", LOSS_NAMES, gamma_mode="fixed_shared_wavelet"),
    AblationVariant("G2", "gamma", "learned global shared wavelet centered deviation", LOSS_NAMES, gamma_mode="learned_global_shared_only"),
    AblationVariant("G3", "gamma", "learned channel-specific wavelet centered deviation", LOSS_NAMES, gamma_mode="learned_channel_specific_only"),
    AblationVariant("G4", "gamma", "dual independent centered-response deviations", LOSS_NAMES, gamma_mode="dual_independent"),
    AblationVariant("G5", "gamma", "uncentered shared-vs-channel-specific response", LOSS_NAMES, gamma_mode="uncentered_cross_response"),
)


def parse_csv_or_space(values: Iterable[str] | None, available: Iterable[str]) -> tuple[str, ...]:
    """Resolve comma/space separated names case-insensitively in stable order."""
    available_tuple = tuple(available)
    lookup = {name.lower(): name for name in available_tuple}
    if not values:
        return available_tuple
    requested: list[str] = []
    for value in values:
        requested.extend(part.strip() for part in value.split(",") if part.strip())
    if not requested or any(item.lower() == "all" for item in requested):
        return available_tuple
    unknown = [item for item in requested if item.lower() not in lookup]
    if unknown:
        raise ValueError(f"Unknown value(s): {', '.join(unknown)}. Available: {', '.join(available_tuple)}")
    selected = {lookup[item.lower()] for item in requested}
    return tuple(item for item in available_tuple if item in selected)


def parse_seeds(values: Iterable[str] | None) -> tuple[int, ...]:
    """Parse ``0,1,2`` and inclusive ranges such as ``0-4``."""
    if not values:
        return (0, 1, 2, 3, 4)
    tokens: list[str] = []
    for value in values:
        tokens.extend(part.strip() for part in value.split(",") if part.strip())
    seeds: set[int] = set()
    for token in tokens:
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"Invalid seed range: {token}")
            seeds.update(range(start, end + 1))
        else:
            value = int(token)
            if value < 0:
                raise ValueError("Seeds must be non-negative.")
            seeds.add(value)
    if not seeds:
        raise ValueError("At least one seed is required.")
    return tuple(sorted(seeds))


def selected_variants(studies: Iterable[str] | None) -> tuple[AblationVariant, ...]:
    """Resolve requested study names to their ordered ablation variants."""

    selected = parse_csv_or_space(studies, ("loss", "alpha", "gamma"))
    variants: list[AblationVariant] = []
    if "loss" in selected:
        variants.extend(LOSS_VARIANTS)
    if "alpha" in selected:
        variants.extend(ALPHA_VARIANTS)
    if "gamma" in selected:
        variants.extend(GAMMA_VARIANTS)
    return tuple(variants)


def selected_specs(datasets: Iterable[str] | None) -> tuple[DatasetSpec, ...]:
    """Resolve requested dataset names to validated dataset specifications."""

    available = tuple(spec.cli_name for spec in DATASET_SPECS)
    names = parse_csv_or_space(datasets, available)
    return tuple(resolve_dataset(name) for name in names)


def planned_run_count(
    studies: Iterable[str] | None = None,
    datasets: Iterable[str] | None = None,
    seeds: Iterable[str] | None = None,
) -> int:
    """Return the Cartesian product size for studies, datasets, and seeds."""

    return len(selected_variants(studies)) * len(selected_specs(datasets)) * len(parse_seeds(seeds))


def matrix_rows() -> tuple[dict[str, str], ...]:
    """A compact table used by dry runs and the Chinese server README."""
    rows = []
    for variant in (*LOSS_VARIANTS, *ALPHA_VARIANTS, *GAMMA_VARIANTS):
        rows.append({
            "code": variant.code,
            "study": variant.study,
            "losses": "+".join(variant.active_losses),
            "alpha_mode": variant.alpha_mode,
            "gamma_mode": variant.gamma_mode,
            "description": variant.description,
        })
    return tuple(rows)
