"""Leakage-controlled loading and similarity construction for MKGMDA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import association_matrix


@dataclass(frozen=True)
class BenchmarkData:
    name: str
    mirna_names: np.ndarray
    disease_names: np.ndarray
    positive_pairs: np.ndarray
    disease_semantic_similarity: np.ndarray

    @property
    def n_mirna(self) -> int:
        return len(self.mirna_names)

    @property
    def n_disease(self) -> int:
        return len(self.disease_names)

    @property
    def full_association(self) -> np.ndarray:
        return association_matrix(
            self.positive_pairs, self.n_mirna, self.n_disease
        )


def _read_names(path: Path, indexed: bool) -> np.ndarray:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if indexed:
            fields = line.split("\t", maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"expected tab-separated index and name in {path}")
            names.append(fields[1].strip())
        else:
            names.append(line.strip())
    return np.asarray(names, dtype=str)


def _existing(path: Path) -> Path:
    """The file itself or its gzip-compressed copy (numpy reads both)."""
    return path if path.exists() else path.with_name(path.name + ".gz")


def load_benchmark(data_root: str | Path, dataset: str) -> BenchmarkData:
    """Load the unmodified MINIMDA benchmark files for HMDD v2.0 or v3.2."""

    root = Path(data_root)
    if dataset == "hmdd_v20":
        path = root / "hmdd_v20"
        mirna_names = _read_names(path / "miRNA number.txt", indexed=True)
        disease_names = _read_names(path / "disease number.txt", indexed=True)
        positive_pairs = (
            np.loadtxt(
                path / "known disease-miRNA association number.txt", dtype=np.int64
            ).reshape(-1, 2)
            - 1
        )
        semantic_1 = np.loadtxt(
            path / "disease semantic similarity matrix 1.txt", dtype=np.float32
        )
        semantic_2 = np.loadtxt(
            path / "disease semantic similarity matrix 2.txt", dtype=np.float32
        )
        disease_semantic = (semantic_1 + semantic_2) / 2.0
    elif dataset in ("hmdd_v32", "hmdd_v40", "rnadisease_v40"):  # v40: build_hmdd_v40.py / build_rnadisease_v40.py, same layout
        path = root / dataset
        mirna_names = _read_names(path / "miRNA number.txt", indexed=False)
        disease_names = _read_names(path / "disease number.txt", indexed=False)
        positive_pairs = (
            np.loadtxt(
                path / "known miRNA-disease association.txt", dtype=np.int64
            ).reshape(-1, 2)
            - 1
        )
        disease_semantic = np.loadtxt(
            _existing(path / "disease semantic similarity matrix.txt"), dtype=np.float32
        )
    else:
        raise ValueError("dataset must be 'hmdd_v20', 'hmdd_v32', or 'hmdd_v40'")

    expected_shape = (len(mirna_names), len(disease_names))
    if positive_pairs.min() < 0:
        raise ValueError(f"negative association index in {path}")
    if positive_pairs[:, 0].max() >= expected_shape[0]:
        raise ValueError(f"miRNA association index exceeds {expected_shape[0]}")
    if positive_pairs[:, 1].max() >= expected_shape[1]:
        raise ValueError(f"disease association index exceeds {expected_shape[1]}")
    if len(np.unique(positive_pairs, axis=0)) != len(positive_pairs):
        raise ValueError(f"duplicate positive associations in {path}")
    if disease_semantic.shape != (expected_shape[1], expected_shape[1]):
        raise ValueError("disease semantic similarity has an unexpected shape")

    return BenchmarkData(
        name=dataset,
        mirna_names=mirna_names,
        disease_names=disease_names,
        positive_pairs=positive_pairs.astype(np.int64),
        disease_semantic_similarity=disease_semantic.astype(np.float32),
    )


def gaussian_interaction_profile(profiles: np.ndarray) -> np.ndarray:
    """Compute a Gaussian interaction-profile kernel with bandwidth base 1."""

    profiles = np.asarray(profiles, dtype=np.float32)
    squared_norm = np.einsum("ij,ij->i", profiles, profiles)
    mean_squared_norm = float(squared_norm.mean())
    if mean_squared_norm <= 0:
        return np.eye(len(profiles), dtype=np.float32)
    gamma = 1.0 / mean_squared_norm
    squared_distance = (
        squared_norm[:, None]
        + squared_norm[None, :]
        - 2.0 * (profiles @ profiles.T)
    )
    np.maximum(squared_distance, 0.0, out=squared_distance)
    return np.exp(-gamma * squared_distance).astype(np.float32)


def mirna_functional_similarity(
    train_association: np.ndarray,
    disease_semantic_similarity: np.ndarray,
) -> np.ndarray:
    """Recompute Wang et al. functional similarity from training edges only.

    For miRNAs i and j with disease sets D_i and D_j, the score is the mean of
    the best cross-set disease semantic similarities in both directions.
    """

    association = np.asarray(train_association, dtype=np.float32)
    disease_similarity = np.asarray(disease_semantic_similarity, dtype=np.float32)
    if disease_similarity.shape != (association.shape[1], association.shape[1]):
        raise ValueError("disease semantic similarity shape is incompatible")

    membership = association > 0
    degrees = membership.sum(axis=1).astype(np.float32)

    # best_to_set[d, j] is the maximum semantic similarity between disease d
    # and any training disease associated with miRNA j. Precomputing this
    # table replaces the quadratic Python loop with matrix operations.
    best_to_set = np.zeros(
        (association.shape[1], association.shape[0]), dtype=np.float32
    )
    nonempty = degrees > 0
    for disease_index in range(association.shape[1]):
        candidates = np.where(
            membership,
            disease_similarity[disease_index][None, :],
            -np.inf,
        )
        best_to_set[disease_index, nonempty] = candidates[nonempty].max(axis=1)

    directed = association @ best_to_set
    denominator = degrees[:, None] + degrees[None, :]
    result = np.zeros_like(directed, dtype=np.float32)
    np.divide(
        directed + directed.T,
        denominator,
        out=result,
        where=denominator > 0,
    )
    return result


def integrate_similarity(curated: np.ndarray, gip: np.ndarray) -> np.ndarray:
    """Use curated similarity at positive entries and GIP at zero entries."""

    curated = np.asarray(curated, dtype=np.float32)
    gip = np.asarray(gip, dtype=np.float32)
    if curated.shape != gip.shape:
        raise ValueError("curated and GIP matrices must have the same shape")
    available = curated > 0
    integrated = np.where(available, curated, gip).astype(np.float32)
    np.fill_diagonal(integrated, 1.0)
    return integrated


def build_fold_inputs(
    train_association: np.ndarray,
    disease_semantic_similarity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct all association-derived inputs from a training fold only."""

    views = build_fold_views(train_association, disease_semantic_similarity)
    mirna_integrated = integrate_similarity(
        views["mirna_functional"], views["mirna_gip"]
    )
    disease_integrated = integrate_similarity(
        views["disease_semantic"], views["disease_gip"]
    )
    return mirna_integrated, disease_integrated, views["mirna_functional"]


def build_fold_views(
    train_association: np.ndarray,
    disease_semantic_similarity: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return separate fold-safe graph views without hard integration.

    Keeping the curated/functional and GIP kernels separate lets a model
    learn ordered view compositions instead of committing to the fixed rule
    ``curated if nonzero else GIP`` before training.
    """

    mirna_gip = gaussian_interaction_profile(train_association)
    disease_gip = gaussian_interaction_profile(train_association.T)
    mirna_functional = mirna_functional_similarity(
        train_association, disease_semantic_similarity
    )
    disease_semantic = np.asarray(
        disease_semantic_similarity, dtype=np.float32
    ).copy()
    np.fill_diagonal(mirna_gip, 1.0)
    np.fill_diagonal(disease_gip, 1.0)
    np.fill_diagonal(disease_semantic, 1.0)
    return {
        "mirna_functional": mirna_functional.astype(np.float32),
        "mirna_gip": mirna_gip.astype(np.float32),
        "disease_semantic": disease_semantic,
        "disease_gip": disease_gip.astype(np.float32),
    }
