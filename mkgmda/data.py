"""Graph construction and pair-sampling utilities for MKGMDA."""

from __future__ import annotations

import random
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int, use_cuda: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch without probing CUDA in CPU runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def association_matrix(
    positive_pairs: np.ndarray,
    n_mirna: int,
    n_disease: int,
) -> np.ndarray:
    matrix = np.zeros((n_mirna, n_disease), dtype=np.float32)
    if len(positive_pairs):
        matrix[positive_pairs[:, 0], positive_pairs[:, 1]] = 1.0
    return matrix


def build_knn_adjacency(similarity: np.ndarray, k: int) -> np.ndarray:
    """Build a weighted, symmetrized k-nearest-neighbor adjacency."""

    n = similarity.shape[0]
    if similarity.shape != (n, n):
        raise ValueError("similarity must be square")
    k = min(k, max(1, n - 1))
    scores = np.asarray(similarity, dtype=np.float32).copy()
    np.fill_diagonal(scores, -np.inf)
    neighbors = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    adjacency = np.zeros_like(scores, dtype=np.float32)
    rows = np.arange(n)[:, None]
    adjacency[rows, neighbors] = similarity[rows, neighbors]
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(adjacency, dtype=np.float32)
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    mask = degree > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])
    return (adjacency * inv_sqrt[:, None]) * inv_sqrt[None, :]


def make_bipartite_adjacency(association: np.ndarray) -> np.ndarray:
    n_mirna, n_disease = association.shape
    adjacency = np.zeros(
        (n_mirna + n_disease, n_mirna + n_disease), dtype=np.float32
    )
    adjacency[:n_mirna, n_mirna:] = association
    adjacency[n_mirna:, :n_mirna] = association.T
    return normalize_adjacency(adjacency)


def row_stochastic(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise a binary incidence matrix, leaving empty rows at zero."""

    weights = np.asarray(matrix, dtype=np.float64)
    degree = weights.sum(axis=1, keepdims=True)
    safe = np.where(degree > 0, degree, 1.0)
    return np.asarray(weights / safe, dtype=np.float32)


def build_tensors(
    association: np.ndarray,
    mirna_similarity: np.ndarray,
    disease_similarity: np.ndarray,
    knn_k: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Convert fold-specific features and graphs into model tensors."""

    n_mirna, n_disease = association.shape
    if mirna_similarity.shape != (n_mirna, n_mirna):
        raise ValueError("miRNA similarity shape does not match association matrix")
    if disease_similarity.shape != (n_disease, n_disease):
        raise ValueError("disease similarity shape does not match association matrix")
    return {
        "x_mirna": torch.as_tensor(
            mirna_similarity, dtype=torch.float32, device=device
        ),
        "x_disease": torch.as_tensor(
            disease_similarity, dtype=torch.float32, device=device
        ),
        "adj_local": torch.as_tensor(
            make_bipartite_adjacency(association),
            dtype=torch.float32,
            device=device,
        ),
        "adj_mirna": torch.as_tensor(
            normalize_adjacency(build_knn_adjacency(mirna_similarity, knn_k)),
            dtype=torch.float32,
            device=device,
        ),
        "adj_disease": torch.as_tensor(
            normalize_adjacency(build_knn_adjacency(disease_similarity, knn_k)),
            dtype=torch.float32,
            device=device,
        ),
        # Row-stochastic support operators for task-conditioned (meta) fusion.
        # Built from the training association matrix only, so a node with no
        # training edge yields an all-zero row and is routed to the cold-start
        # prior instead of to a degenerate task embedding.
        "support_mirna": torch.as_tensor(
            row_stochastic(association), dtype=torch.float32, device=device
        ),
        "support_disease": torch.as_tensor(
            row_stochastic(association.T), dtype=torch.float32, device=device
        ),
    }


def unlabeled_pool(full_association: np.ndarray) -> np.ndarray:
    """Return pairs absent from the complete benchmark positive set."""

    return np.argwhere(full_association == 0).astype(np.int64)


def exclude_pairs(pool: np.ndarray, excluded: np.ndarray, n_disease: int) -> np.ndarray:
    """Remove exact pairs from a candidate pool without Python-level set loops."""

    if len(excluded) == 0:
        return pool.copy()
    pool_ids = pool[:, 0] * n_disease + pool[:, 1]
    excluded_ids = excluded[:, 0] * n_disease + excluded[:, 1]
    return pool[~np.isin(pool_ids, excluded_ids)]


def sample_pairs(
    pool: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return np.empty((0, 2), dtype=np.int64)
    if len(pool) < count:
        raise ValueError(f"cannot sample {count} distinct pairs from a pool of {len(pool)}")
    indices = rng.choice(len(pool), size=count, replace=False)
    return pool[indices]


def labeled_pairs(
    positive: np.ndarray, unlabeled: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.vstack([positive, unlabeled]).astype(np.int64)
    labels = np.concatenate(
        [
            np.ones(len(positive), dtype=np.float32),
            np.zeros(len(unlabeled), dtype=np.float32),
        ]
    )
    return pairs, labels
