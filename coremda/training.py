"""Training and scoring routines shared by cross-validation and hold-out runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .data import build_knn_adjacency, build_tensors, labeled_pairs, normalize_adjacency, sample_pairs, set_seed
from .model import CoReMDA, contrastive_loss, proximity_loss


@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = 400
    batch_size: int = 4096
    hidden: int = 256
    layers: int = 1
    heads: int = 4
    dropout: float = 0.3
    knn_k: int = 20
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    negative_ratio: float = 1.0
    temperature: float = 0.2
    gamma_contrastive: float = 0.2
    mu_proximity: float = 0.5
    proximity_negatives: int = 5
    resample_unlabeled_each_epoch: bool = True
    use_global_view: bool = True
    use_global_prior: bool = False
    impute_cold: bool = False
    view_recovery: bool = False
    recovery_weight: float = 0.1
    global_mode: str = "attention"
    anova_decoder: bool = False
    view_dropout: float = 0.0
    recover_homogeneous: bool = False
    degree_recovery: bool = False
    degree_mode: str = "regress"
    degree_dropout: float = 0.0
    cold_branch: bool = False
    branch_neighbour_only: bool = False
    branch_bilinear: bool = False
    degree_offset: bool = False
    use_homogeneous_gate: bool = True
    fusion_mode: str = "attention"
    use_contrastive: bool = True
    use_proximity: bool = True
    score_prior: str = "none"
    edge_masking: bool = False
    loss_mode: str = "bce"
    pu_prior: float = 0.05
    mirna_external: str = ""
    disease_external: str = ""
    device: str = "cpu"

    def to_dict(self) -> dict:
        return asdict(self)


def nnpu_risk(logits: torch.Tensor, labels: torch.Tensor, prior: float) -> torch.Tensor:
    """Non-negative PU risk (Kiryo et al., 2017): unlabeled pairs are not negatives."""

    positive = labels > 0.5
    zero = logits.new_tensor(0.0)
    risk_p_pos = F.softplus(-logits[positive]).mean() if positive.any() else zero
    risk_p_neg = F.softplus(logits[positive]).mean() if positive.any() else zero
    risk_u_neg = F.softplus(logits[~positive]).mean() if (~positive).any() else zero
    negative_risk = risk_u_neg - prior * risk_p_neg
    if float(negative_risk) < 0.0:
        return -negative_risk
    return prior * risk_p_pos + negative_risk


def load_external(path: str, n_nodes: int, device: torch.device) -> torch.Tensor | None:
    """Row-standardised association-free node features plus a coverage flag."""

    if not path:
        return None
    data = np.load(path)
    features = np.asarray(data["features"], dtype=np.float32)
    if features.shape[0] != n_nodes:
        raise ValueError(f"{path}: expected {n_nodes} rows, got {features.shape[0]}")
    found = np.asarray(data["found"], dtype=bool) if "found" in data.files else np.ones(n_nodes, bool)
    mean = features[found].mean(axis=0)
    std = features[found].std(axis=0) + 1e-6
    features = np.where(found[:, None], (features - mean) / std, 0.0).astype(np.float32)
    features = np.concatenate([features, found[:, None].astype(np.float32)], axis=1)
    return torch.as_tensor(features, device=device)


def structural_prior(
    association: np.ndarray,
    mirna_similarity: np.ndarray,
    disease_similarity: np.ndarray,
    knn_k: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Training-structure prior for the structure-guided global view.

    P = K_m A K_d^T, where K is the row-normalised (KNN similarity adjacency + I):
    P[m, d] is the similarity-weighted count of training associations between the
    neighbourhoods of m and d. Each direction is scaled by its row maximum.
    """

    def row_normalised_knn(similarity: np.ndarray) -> np.ndarray:
        kernel = build_knn_adjacency(similarity, knn_k) + np.eye(len(similarity), dtype=np.float32)
        return kernel / kernel.sum(axis=1, keepdims=True)

    def row_max_scaled(matrix: np.ndarray) -> np.ndarray:
        peak = matrix.max(axis=1, keepdims=True)
        return np.where(peak > 0, matrix / np.maximum(peak, 1e-12), 0.0).astype(np.float32)

    prior = (
        row_normalised_knn(mirna_similarity)
        @ np.asarray(association, dtype=np.float32)
        @ row_normalised_knn(disease_similarity).T
    )
    return {
        "global_prior_md": torch.as_tensor(row_max_scaled(prior), device=device),
        "global_prior_dm": torch.as_tensor(row_max_scaled(prior.T), device=device),
    }


def cold_imputation(
    association: np.ndarray,
    mirna_similarity: np.ndarray,
    disease_similarity: np.ndarray,
    knn_k: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Row operators that replace a cold node's local view by its warm neighbours' mean.

    Warm rows are the identity. A cold row (no training edge) holds the node's k most similar
    warm nodes, weighted by similarity and normalised to sum to one; with no similar warm node
    it stays the identity.
    """

    def operator(similarity: np.ndarray, warm: np.ndarray) -> np.ndarray:
        n = len(similarity)
        out = np.eye(n, dtype=np.float32)
        candidates = np.where(warm[None, :], np.asarray(similarity, dtype=np.float32), -np.inf)
        np.fill_diagonal(candidates, -np.inf)
        k = min(knn_k, int(warm.sum()))
        for node in np.flatnonzero(~warm):
            neighbours = np.argpartition(-candidates[node], k - 1)[:k]
            weights = np.maximum(similarity[node, neighbours], 0.0)
            if weights.sum() > 0:
                out[node] = 0.0
                out[node, neighbours] = weights / weights.sum()
        return out

    association = np.asarray(association)
    return {
        "impute_mirna": torch.as_tensor(
            operator(mirna_similarity, association.sum(axis=1) > 0), device=device
        ),
        "impute_disease": torch.as_tensor(
            operator(disease_similarity, association.sum(axis=0) > 0), device=device
        ),
    }


def heterogeneous_adjacency(
    association: np.ndarray,
    mirna_similarity: np.ndarray,
    disease_similarity: np.ndarray,
    knn_k: int,
    device: torch.device,
) -> torch.Tensor:
    """Symmetric-normalised [[K_m, A], [A^T, K_d]] + I: training associations plus similarity KNN edges."""

    association = np.asarray(association, dtype=np.float32)
    n_mirna, n_disease = association.shape
    graph = np.zeros((n_mirna + n_disease, n_mirna + n_disease), dtype=np.float32)
    graph[:n_mirna, :n_mirna] = build_knn_adjacency(mirna_similarity, knn_k)
    graph[n_mirna:, n_mirna:] = build_knn_adjacency(disease_similarity, knn_k)
    graph[:n_mirna, n_mirna:] = association
    graph[n_mirna:, :n_mirna] = association.T
    graph += np.eye(len(graph), dtype=np.float32)
    scale = 1.0 / np.sqrt(graph.sum(axis=1))
    return torch.as_tensor(graph * scale[:, None] * scale[None, :], device=device)


def masked_bipartite_adjacency(
    association: torch.Tensor, masked_pairs: torch.Tensor
) -> torch.Tensor:
    """Symmetric-normalised bipartite adjacency with the given edges removed."""

    graph = association.clone()
    graph[masked_pairs[:, 0], masked_pairs[:, 1]] = 0.0
    n_mirna, n_disease = graph.shape
    mirna_degree = graph.sum(dim=1)
    disease_degree = graph.sum(dim=0)
    mirna_scale = torch.where(
        mirna_degree > 0, mirna_degree.clamp_min(1e-12).rsqrt(), torch.zeros_like(mirna_degree)
    )
    disease_scale = torch.where(
        disease_degree > 0, disease_degree.clamp_min(1e-12).rsqrt(), torch.zeros_like(disease_degree)
    )
    block = mirna_scale[:, None] * graph * disease_scale[None, :]
    adjacency = graph.new_zeros((n_mirna + n_disease, n_mirna + n_disease))
    adjacency[:n_mirna, n_mirna:] = block
    adjacency[n_mirna:, :n_mirna] = block.T
    return adjacency


def train_model(
    train_association: np.ndarray,
    full_association: np.ndarray,
    mirna_similarity: np.ndarray,
    disease_similarity: np.ndarray,
    train_positive: np.ndarray,
    train_unlabeled_pool: np.ndarray,
    config: TrainingConfig,
    progress: Callable[[dict], None] | None = None,
    additional_proximity_exclusions: np.ndarray | None = None,
    intrinsic_similarities: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[CoReMDA, dict[str, torch.Tensor], list[dict[str, float]]]:
    """Train one fold with full-graph encoding and balanced PU sampling."""

    device = torch.device(config.device)
    set_seed(config.seed, use_cuda=device.type == "cuda")
    tensors = build_tensors(
        train_association,
        mirna_similarity,
        disease_similarity,
        config.knn_k,
        device,
    )
    for key, path in (("x_mirna", config.mirna_external), ("x_disease", config.disease_external)):
        external = load_external(path, tensors[key].shape[0], device)
        if external is not None:
            tensors[key] = torch.cat([tensors[key], external], dim=1)
    if config.anova_decoder or config.cold_branch:
        def standardised_log_degree(degree: np.ndarray) -> torch.Tensor:
            values = np.log1p(degree.astype(np.float32))
            warm = degree > 0
            values = (values - values[warm].mean()) / (values[warm].std() + 1e-6)
            return torch.as_tensor(np.where(warm, values, 0.0).astype(np.float32), device=device)

        tensors["degree"] = {
            "mirna": standardised_log_degree(np.asarray(train_association).sum(axis=1)),
            "disease": standardised_log_degree(np.asarray(train_association).sum(axis=0)),
        }
        # Standard deviation of the warm log-degrees: turns the standardised degree back into log units.
        for kind, degree in (("mirna", np.asarray(train_association).sum(axis=1)),
                             ("disease", np.asarray(train_association).sum(axis=0))):
            tensors["degree"][f"{kind}_scale"] = float(np.log1p(degree[degree > 0].astype(np.float32)).std() + 1e-6)
    if config.global_mode == "intrinsic":
        if intrinsic_similarities is None:
            raise ValueError("the intrinsic view needs association-free similarities")
        tensors["intrinsic"] = {
            f"x_{kind}": torch.as_tensor(np.asarray(sim, dtype=np.float32), device=device)
            for kind, sim in zip(("mirna", "disease"), intrinsic_similarities)
        }
        tensors["intrinsic"].update({
            f"adj_{kind}": torch.as_tensor(
                normalize_adjacency(build_knn_adjacency(np.asarray(sim, dtype=np.float32), config.knn_k)),
                device=device,
            )
            for kind, sim in zip(("mirna", "disease"), intrinsic_similarities)
        })
    if config.global_mode == "diffusion":
        tensors["adj_union"] = heterogeneous_adjacency(
            train_association, mirna_similarity, disease_similarity, config.knn_k, device
        )
    if config.view_recovery or config.degree_recovery or config.cold_branch:
        tensors["cold_mirna"] = torch.as_tensor(train_association.sum(axis=1) == 0, device=device)
        tensors["cold_disease"] = torch.as_tensor(train_association.sum(axis=0) == 0, device=device)
    if config.impute_cold:
        tensors.update(
            cold_imputation(
                train_association, mirna_similarity, disease_similarity, config.knn_k, device
            )
        )
    if config.use_global_prior:
        tensors.update(
            structural_prior(
                train_association, mirna_similarity, disease_similarity, config.knn_k, device
            )
        )
    model = CoReMDA(
        mirna_input_dim=tensors["x_mirna"].shape[1],
        disease_input_dim=tensors["x_disease"].shape[1],
        n_mirna=train_association.shape[0],
        n_disease=train_association.shape[1],
        hidden=config.hidden,
        layers=config.layers,
        heads=config.heads,
        dropout=config.dropout,
        use_global_view=config.use_global_view,
        use_homogeneous_gate=config.use_homogeneous_gate,
        fusion_mode=config.fusion_mode,
        score_prior=config.score_prior,
        global_prior=config.use_global_prior,
        view_recovery=config.view_recovery,
        global_mode=config.global_mode,
        mirna_intrinsic_dim=None if intrinsic_similarities is None else len(intrinsic_similarities[0]),
        disease_intrinsic_dim=None if intrinsic_similarities is None else len(intrinsic_similarities[1]),
        anova_decoder=config.anova_decoder,
        view_dropout=config.view_dropout,
        recover_homogeneous=config.recover_homogeneous,
        degree_recovery=config.degree_recovery,
        degree_mode=config.degree_mode,
        degree_dropout=config.degree_dropout,
        cold_branch=config.cold_branch,
        branch_neighbour_only=config.branch_neighbour_only,
        branch_bilinear=config.branch_bilinear,
        degree_offset=config.degree_offset,
    ).to(device)
    model.set_training_degrees(
        torch.as_tensor(train_association, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_positive_tensor = torch.as_tensor(
        train_positive, dtype=torch.long, device=device
    )
    proximity_exclusions = np.asarray(full_association > 0, dtype=bool)
    if additional_proximity_exclusions is not None:
        additional = np.asarray(additional_proximity_exclusions, dtype=bool)
        if additional.shape != proximity_exclusions.shape:
            raise ValueError(
                "additional proximity exclusions must match the association matrix"
            )
        proximity_exclusions = np.logical_or(proximity_exclusions, additional)
    excluded_positive_mask = torch.as_tensor(
        proximity_exclusions, dtype=torch.bool, device=device
    )
    association_tensor = torch.as_tensor(
        train_association, dtype=torch.float32, device=device
    )
    numpy_rng = np.random.default_rng(config.seed)
    torch_rng = torch.Generator(device="cpu")
    torch_rng.manual_seed(config.seed)
    n_unlabeled = int(round(len(train_positive) * config.negative_ratio))
    fixed_unlabeled = sample_pairs(train_unlabeled_pool, n_unlabeled, numpy_rng)
    history: list[dict[str, float]] = []
    if config.batch_size <= 0:
        raise ValueError("batch size must be positive")

    for epoch in range(1, config.epochs + 1):
        if config.resample_unlabeled_each_epoch:
            epoch_unlabeled = sample_pairs(
                train_unlabeled_pool, n_unlabeled, numpy_rng
            )
        else:
            epoch_unlabeled = fixed_unlabeled
        sampled_pairs, sampled_labels = labeled_pairs(
            train_positive, epoch_unlabeled
        )
        pair_tensor = torch.as_tensor(sampled_pairs, dtype=torch.long, device=device)
        label_tensor = torch.as_tensor(
            sampled_labels, dtype=torch.float32, device=device
        )
        permutation = torch.randperm(
            len(sampled_pairs), generator=torch_rng
        ).to(device)
        totals = {
            "loss": 0.0,
            "bce": 0.0,
            "contrastive": 0.0,
            "proximity": 0.0,
        }
        steps = 0
        model.train()
        for start in range(0, len(sampled_pairs), config.batch_size):
            batch_indices = permutation[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            batch_pairs = pair_tensor[batch_indices]
            batch_positive = batch_pairs[label_tensor[batch_indices] > 0.5]
            if config.edge_masking:
                # Supervised positives are hidden from message passing so that
                # training matches inference, where test pairs are never edges.
                adjacency_local = masked_bipartite_adjacency(
                    association_tensor, batch_positive
                )
            else:
                adjacency_local = tensors["adj_local"]
            encoded = model.encode(
                tensors["x_mirna"],
                tensors["x_disease"],
                adjacency_local,
                tensors["adj_mirna"],
                tensors["adj_disease"],
                tensors.get("support_mirna"),
                tensors.get("support_disease"),
                tensors.get("global_prior_md"),
                tensors.get("global_prior_dm"),
                tensors.get("impute_mirna"),
                tensors.get("impute_disease"),
                tensors.get("cold_mirna"),
                tensors.get("cold_disease"),
                tensors.get("adj_union"),
                tensors.get("intrinsic"),
                tensors.get("degree"),
            )
            logits = model.decode(
                encoded["mirna_fused"],
                encoded["disease_fused"],
                pair_tensor[batch_indices],
            )
            if config.loss_mode == "nnpu":
                binary_cross_entropy = nnpu_risk(
                    logits, label_tensor[batch_indices], config.pu_prior
                )
            else:
                binary_cross_entropy = F.binary_cross_entropy_with_logits(
                    logits, label_tensor[batch_indices]
                )
            if config.use_contrastive:
                contrastive = contrastive_loss(
                    encoded["mirna_views"], config.temperature
                ) + contrastive_loss(
                    encoded["disease_views"], config.temperature
                )
            else:
                contrastive = binary_cross_entropy.new_tensor(0.0)
            if config.use_proximity:
                proximity = proximity_loss(
                    encoded["mirna_fused"],
                    encoded["disease_fused"],
                    batch_positive if config.edge_masking else train_positive_tensor,
                    excluded_positive_mask,
                    config.proximity_negatives,
                )
            else:
                proximity = binary_cross_entropy.new_tensor(0.0)
            loss = binary_cross_entropy
            loss = loss + config.gamma_contrastive * contrastive
            loss = loss + config.mu_proximity * proximity
            if config.view_recovery:
                loss = loss + config.recovery_weight * encoded["recovery_loss"]
            if config.degree_recovery:
                loss = loss + encoded["degree_loss"]
            if config.cold_branch:
                branch_logits = model.decode_branch(pair_tensor[batch_indices])
                if config.loss_mode == "nnpu":
                    loss = loss + nnpu_risk(branch_logits, label_tensor[batch_indices], config.pu_prior)
                else:
                    loss = loss + F.binary_cross_entropy_with_logits(branch_logits, label_tensor[batch_indices])
            loss.backward()
            branch_ids = {id(parameter) for parameter in model.branch_parameters()}
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if id(parameter) not in branch_ids], 5.0
            )
            if branch_ids:
                torch.nn.utils.clip_grad_norm_(model.branch_parameters(), 5.0)
            optimizer.step()
            totals["loss"] += float(loss.item())
            totals["bce"] += float(binary_cross_entropy.item())
            totals["contrastive"] += float(contrastive.item())
            totals["proximity"] += float(proximity.item())
            steps += 1

        row = {"epoch": epoch}
        row.update({name: value / steps for name, value in totals.items()})
        history.append(row)
        if progress is not None:
            progress(row)
    return model, tensors, history


def score_pairs(
    model: CoReMDA,
    tensors: dict[str, torch.Tensor],
    pairs: np.ndarray,
    batch_size: int = 65536,
) -> np.ndarray:
    """Score pairs after encoding the graph exactly once."""

    model.eval()
    device = tensors["x_mirna"].device
    chunks = []
    with torch.no_grad():
        encoded = model.encode(
            tensors["x_mirna"],
            tensors["x_disease"],
            tensors["adj_local"],
            tensors["adj_mirna"],
            tensors["adj_disease"],
            tensors.get("support_mirna"),
            tensors.get("support_disease"),
            tensors.get("global_prior_md"),
            tensors.get("global_prior_dm"),
            tensors.get("impute_mirna"),
            tensors.get("impute_disease"),
            tensors.get("cold_mirna"),
            tensors.get("cold_disease"),
            tensors.get("adj_union"),
            tensors.get("intrinsic"),
            tensors.get("degree"),
        )
        for start in range(0, len(pairs), batch_size):
            pair_tensor = torch.as_tensor(
                pairs[start : start + batch_size], dtype=torch.long, device=device
            )
            logits = model.decode(
                encoded["mirna_fused"],
                encoded["disease_fused"],
                pair_tensor,
            )
            if model.cold_branch:
                both_cold = tensors["cold_mirna"][pair_tensor[:, 0]] & tensors["cold_disease"][pair_tensor[:, 1]]
                logits = torch.where(both_cold, model.decode_branch(pair_tensor), logits)
            chunks.append(logits.detach().cpu().numpy())
    if not chunks:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)
