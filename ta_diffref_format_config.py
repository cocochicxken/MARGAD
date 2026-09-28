"""F0--F5 TA-DiffRef formats with retained target return in final F5."""

from __future__ import annotations

from dataclasses import dataclass


EXPERIMENT_VERSION = "ta-diffref-f0-f5-retained-return-10seed-v2"
DEFAULT_SEEDS = tuple(range(10))


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    cli_name: str
    hidden_dim: int
    learning_rate: float
    epochs: int
    global_deviation_weight: float
    waveshift_weight: float
    gamma_centering: int
    trainer: str


@dataclass(frozen=True)
class VariantConfig:
    code: str
    alpha_mode: str
    reference_operator: str
    calibration: str
    target_return: str
    integration: str


# Dataset hyperparameters are fixed to the earlier ten-run study; the final
# operator is now F5, so its new results must not be labeled as Table 2 values.
DATASETS = (
    DatasetConfig("facebook", "Facebook", 64, 0.003, 70, 0.15, 0.50, 0, "full_graph"),
    DatasetConfig("reddit", "Reddit", 128, 0.003, 110, 0.35, 1.45, 1, "full_graph"),
    DatasetConfig("yelpchi", "YelpChi", 64, 0.003, 65, 0.15, 1.05, 1, "full_graph"),
    DatasetConfig("tfinance", "tfinance", 64, 0.003, 85, 1.00, 0.05, 1, "full_graph"),
    DatasetConfig("elliptic", "elliptic", 64, 0.001, 70, 0.30, 1.00, 0, "full_graph"),
    DatasetConfig("tsocial", "tsocial", 64, 0.003, 10, 0.85, 0.75, 0, "large_graph"),
)


# B = A - diag(A), D = diag(B1), P = D^-1 B, S = D^-1/2 B D^-1/2.
# F2--F5 retain the full model's learned three-head scale coefficients.
RUN_VARIANTS = (
    VariantConfig("F0", "one_hop", "P", "row normalization", "no self edge", "one hop"),
    VariantConfig("F1", "symmetric_one_hop", "S", "symmetric degree form", "no self edge", "one hop"),
    VariantConfig("F2", "learned_no_degree_or_path_volume_normalization", "B, B^2", "off", "retained", "adaptive three-head"),
    VariantConfig("F3", "full", "P, (P^2 - diag(P^2)) / (1 - diag(P^2))", "on", "removed", "adaptive three-head"),
    VariantConfig("F4", "learned_raw_target_anonymization", "B, B^2 - diag(B^2)", "off", "removed", "adaptive three-head"),
    VariantConfig("F5", "learned_no_target_anonymization", "P, P^2", "on", "retained", "adaptive three-head"),
)
ALL_VARIANTS = RUN_VARIANTS


def _tokens(values) -> list[str]:
    return [part.strip() for value in values or () for part in str(value).split(",") if part.strip()]


def select_datasets(values=None) -> tuple[DatasetConfig, ...]:
    tokens = _tokens(values)
    if not tokens or any(token.lower() == "all" for token in tokens):
        return DATASETS
    lookup = {name.lower(): spec for spec in DATASETS for name in (spec.key, spec.cli_name)}
    unknown = [token for token in tokens if token.lower() not in lookup]
    if unknown:
        raise ValueError("Unknown dataset(s): " + ", ".join(unknown))
    chosen = {lookup[token.lower()].key for token in tokens}
    return tuple(spec for spec in DATASETS if spec.key in chosen)


def select_run_variants(values=None) -> tuple[VariantConfig, ...]:
    tokens = _tokens(values)
    if not tokens or any(token.lower() == "all" for token in tokens):
        return RUN_VARIANTS
    lookup = {variant.code.lower(): variant for variant in RUN_VARIANTS}
    unknown = [token for token in tokens if token.lower() not in lookup]
    if unknown:
        raise ValueError("Unknown format variant(s): " + ", ".join(unknown))
    chosen = {token.lower() for token in tokens}
    return tuple(variant for variant in RUN_VARIANTS if variant.code.lower() in chosen)


def parse_seeds(values=None) -> tuple[int, ...]:
    tokens = _tokens(values)
    if not tokens:
        return DEFAULT_SEEDS
    seeds = []
    for token in tokens:
        if "-" in token:
            first, last = token.split("-", 1)
            start, stop = int(first), int(last)
            if stop < start:
                raise ValueError(f"Invalid seed range: {token}")
            seeds.extend(range(start, stop + 1))
        else:
            seeds.append(int(token))
    if any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be non-negative")
    return tuple(dict.fromkeys(seeds))


def operator_protocol(dataset: DatasetConfig, variant: VariantConfig) -> dict[str, str]:
    if dataset.trainer == "full_graph":
        return {
            "execution": "sparse full-graph propagation; no dense N-by-N two-hop matrix",
            "one_hop_operator": variant.reference_operator,
            "target_return": variant.target_return,
            "negative_reference": "same operator applied to sampled non-edge graph",
        }
    return {
        "execution": "two-layer DGL neighbor sampling; fanout 8",
        "one_hop_operator": variant.reference_operator,
        "raw_F2_F4": "sampled message/path sums, not exact full-graph B^2",
        "calibrated_F0_F3_F5": "sampled row means, not exact full-graph P^2",
        "raw_F4_return": "subtract sampled reciprocal two-hop return count times target embedding",
        "target_return": variant.target_return,
        "negative_reference": "random non-target sampled contexts with the same variant arithmetic",
    }
