"""MKGMDA model: multi-view encoders, dual prediction, degree propagation, and the degree-corrected decoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(self, layers: int):
        super().__init__()
        self.layers = layers

    def forward(self, features: torch.Tensor, normalized_adjacency: torch.Tensor) -> torch.Tensor:
        layer_outputs = [features]
        hidden = features
        for _ in range(self.layers):
            hidden = normalized_adjacency @ hidden
            layer_outputs.append(hidden)
        return torch.stack(layer_outputs, dim=0).mean(dim=0)


class CrossTypeAttention(nn.Module):
    """Dense bidirectional attention between miRNA and disease nodes."""

    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden dimension must be divisible by attention heads")
        self.heads = heads
        self.head_dim = hidden // heads
        self.q_m = nn.Linear(hidden, hidden)
        self.k_m = nn.Linear(hidden, hidden)
        self.v_m = nn.Linear(hidden, hidden)
        self.q_d = nn.Linear(hidden, hidden)
        self.k_d = nn.Linear(hidden, hidden)
        self.v_d = nn.Linear(hidden, hidden)
        self.o_m = nn.Linear(hidden, hidden)
        self.o_d = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.norm_m = nn.LayerNorm(hidden)
        self.norm_d = nn.LayerNorm(hidden)

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        return values.view(values.size(0), self.heads, self.head_dim).transpose(0, 1)

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if bias is not None:
            logits = logits + bias
        weights = torch.softmax(logits, dim=-1)
        output = torch.matmul(self.dropout(weights), value)
        return output.transpose(0, 1).contiguous().view(query.size(1), -1)

    def forward(
        self,
        mirna_features: torch.Tensor,
        disease_features: torch.Tensor,
        prior_md: torch.Tensor | None = None,
        prior_dm: torch.Tensor | None = None,
        prior_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optional structural priors add ``prior_scale[h] * prior`` to head h's logits."""

        bias_m = bias_d = None
        if prior_scale is not None:
            bias_m = prior_scale.view(-1, 1, 1) * prior_md.unsqueeze(0)
            bias_d = prior_scale.view(-1, 1, 1) * prior_dm.unsqueeze(0)
        query_m = self._split_heads(self.q_m(mirna_features))
        key_m = self._split_heads(self.k_m(mirna_features))
        value_m = self._split_heads(self.v_m(mirna_features))
        query_d = self._split_heads(self.q_d(disease_features))
        key_d = self._split_heads(self.k_d(disease_features))
        value_d = self._split_heads(self.v_d(disease_features))

        output_m = self.o_m(self._attend(query_m, key_d, value_d, bias_m))
        output_d = self.o_d(self._attend(query_d, key_m, value_m, bias_d))
        output_m = self.norm_m(mirna_features + self.dropout(output_m))
        output_d = self.norm_d(disease_features + self.dropout(output_d))
        return output_m, output_d


class ProjectionHead(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ViewFusion(nn.Module):
    """Node-specific view interaction followed by type-specific view weights."""

    def __init__(self, hidden: int, dropout: float, n_views: int, mode: str = "attention"):
        super().__init__()
        if mode not in {"attention", "mean", "concat"}:
            raise ValueError("fusion mode must be attention, mean, or concat")
        self.mode = mode
        self.n_views = n_views
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.theta = nn.Parameter(torch.zeros(n_views))
        self.dropout = nn.Dropout(dropout)
        self.concat_projection = nn.Linear(hidden * n_views, hidden)

    def forward(
        self, views: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stacked = torch.stack(views, dim=1)
        if self.mode == "attention":
            query = self.query(stacked)
            key = self.key(stacked)
            value = self.value(stacked)
            attention = torch.softmax(
                torch.matmul(query, key.transpose(1, 2))
                / math.sqrt(stacked.size(-1)),
                dim=-1,
            )
            refined = stacked + self.dropout(torch.matmul(attention, value))
            view_weights = torch.softmax(self.theta, dim=0)
            fused = (refined * view_weights.view(1, -1, 1)).sum(dim=1)
        elif self.mode == "mean":
            refined = stacked
            attention = torch.full(
                (stacked.size(0), self.n_views, self.n_views),
                1.0 / self.n_views,
                dtype=stacked.dtype,
                device=stacked.device,
            )
            fused = stacked.mean(dim=1)
        else:
            refined = stacked
            attention = torch.full(
                (stacked.size(0), self.n_views, self.n_views),
                1.0 / self.n_views,
                dtype=stacked.dtype,
                device=stacked.device,
            )
            fused = self.concat_projection(stacked.flatten(start_dim=1))
        return fused, refined, attention


class MetaFusion(nn.Module):
    """Task-conditioned view weighting.

    Each node is treated as a few-shot task. The support set of a node is its
    observed counterpart set in the *training* bipartite graph: for a miRNA the
    diseases it is known to be associated with, for a disease the miRNAs. A
    permutation-invariant mean pooling over the support gives a task embedding,
    and a small meta-learner maps that embedding to node-specific view weights.
    Nodes with an empty support set fall back to a learned cold-start prior,
    which is the inductive case the fixed per-type weights cannot express.
    """

    def __init__(self, hidden: int, n_views: int, bottleneck: int = 64):
        super().__init__()
        self.n_views = n_views
        self.meta_learner = nn.Sequential(
            nn.Linear(hidden, bottleneck),
            nn.ELU(),
            nn.Linear(bottleneck, n_views),
        )
        self.cold_start_prior = nn.Parameter(torch.zeros(n_views))

    def forward(
        self,
        views: list[torch.Tensor],
        support_operator: torch.Tensor,
        counterpart_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stacked = torch.stack(views, dim=1)
        task_embedding = support_operator @ counterpart_features
        support_size = support_operator.sum(dim=1, keepdim=True)
        logits = self.meta_learner(task_embedding)
        logits = torch.where(
            support_size > 0,
            logits,
            self.cold_start_prior.unsqueeze(0).expand_as(logits),
        )
        view_weights = torch.softmax(logits, dim=1)
        fused = (stacked * view_weights.unsqueeze(-1)).sum(dim=1)
        return fused, stacked, view_weights


class MKGMDA(nn.Module):
    def __init__(
        self,
        n_mirna: int,
        n_disease: int,
        hidden: int = 256,
        layers: int = 1,
        heads: int = 4,
        dropout: float = 0.3,
        use_global_view: bool = True,
        use_homogeneous_gate: bool = True,
        fusion_mode: str = "attention",
        score_prior: str = "none",
        mirna_input_dim: int | None = None,
        disease_input_dim: int | None = None,
        global_prior: bool = False,
        view_recovery: bool = False,
        global_mode: str = "attention",
        diffusion_steps: int = 10,
        diffusion_alpha: float = 0.1,
        mirna_intrinsic_dim: int | None = None,
        disease_intrinsic_dim: int | None = None,
        anova_decoder: bool = False,
        view_dropout: float = 0.0,
        recover_homogeneous: bool = False,
        degree_recovery: bool = False,
        degree_mode: str = "regress",
        degree_dropout: float = 0.0,
        cold_branch: bool = False,
        branch_neighbour_only: bool = False,
        branch_bilinear: bool = False,
        degree_offset: bool = False,
    ):
        super().__init__()
        if score_prior not in {"none", "degree", "nodebias"}:
            raise ValueError(f"unknown score prior: {score_prior}")
        self.score_prior = score_prior
        self.use_global_view = use_global_view
        self.use_homogeneous_gate = use_homogeneous_gate
        self.view_names = ["local", "homogeneous"]
        if use_global_view:
            self.view_names.append("global")

        self.mirna_projection = nn.Sequential(nn.Linear(mirna_input_dim or n_mirna, hidden), nn.ELU())
        self.disease_projection = nn.Sequential(nn.Linear(disease_input_dim or n_disease, hidden), nn.ELU())
        self.homogeneous_gate = nn.Linear(hidden, hidden)
        self.local_encoder = LightGCN(layers)
        self.homogeneous_encoder = LightGCN(layers)
        self.cross_type_attention = CrossTypeAttention(hidden, heads, dropout)
        self.mirna_view_heads = nn.ModuleList(
            [ProjectionHead(hidden, dropout) for _ in self.view_names]
        )
        self.disease_view_heads = nn.ModuleList(
            [ProjectionHead(hidden, dropout) for _ in self.view_names]
        )
        self.fusion_mode = fusion_mode
        base_fusion_mode = "mean" if fusion_mode == "meta" else fusion_mode
        self.mirna_fusion = ViewFusion(
            hidden, dropout, len(self.view_names), mode=base_fusion_mode
        )
        self.disease_fusion = ViewFusion(
            hidden, dropout, len(self.view_names), mode=base_fusion_mode
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # Added at the end of __init__ on purpose: creating it earlier would
        # shift the RNG stream of every module below it and silently change the
        # baseline initialisation.
        if fusion_mode == "meta":
            self.mirna_meta_fusion = MetaFusion(hidden, len(self.view_names))
            self.disease_meta_fusion = MetaFusion(hidden, len(self.view_names))
        # Score priors are zero-initialised parameters created last, so they do not
        # consume the RNG stream and "none" reproduces the baseline bit for bit.
        if score_prior == "degree":
            self.degree_weight = nn.Parameter(torch.zeros(2))
            self.register_buffer("mirna_degree", torch.zeros(n_mirna))
            self.register_buffer("disease_degree", torch.zeros(n_disease))
        elif score_prior == "nodebias":
            self.mirna_bias = nn.Parameter(torch.zeros(n_mirna))
            self.disease_bias = nn.Parameter(torch.zeros(n_disease))
        # Structure-guided global view: one learnable scale per attention head for
        # the training-structure prior. Deterministic init, created last (no RNG use).
        self.global_prior_scale = (
            nn.Parameter(torch.ones(heads)) if global_prior and use_global_view else None
        )
        # Missing-view recovery (dual contrastive prediction, Lin et al. CVPR'21 / TPAMI'22):
        # the local view needs training edges, the other views do not. Created last so the
        # RNG stream of every module above is unchanged.
        self.view_recovery = view_recovery
        # Missing-view augmentation: during training a random fraction of warm nodes also receive the
        # recovered local view, so the fusion and decoder learn to use recovered views (cf. ModDrop).
        self.view_dropout = view_dropout
        # Global view: dense cross-type attention (original) or parameter-free personalised-PageRank
        # diffusion (APPNP, Gasteiger et al. ICLR'19) over the heterogeneous graph of training
        # associations plus both similarity KNN graphs.
        if global_mode not in {"attention", "diffusion", "intrinsic"}:
            raise ValueError(f"unknown global mode: {global_mode}")
        self.global_mode = global_mode
        self.diffusion_steps = diffusion_steps
        self.diffusion_alpha = diffusion_alpha
        # Intrinsic view: association-free similarities (miRNA sequence, disease MeSH semantics),
        # computed identically for warm and cold nodes; own projections, created last.
        if global_mode == "intrinsic" and use_global_view:
            if mirna_intrinsic_dim is None or disease_intrinsic_dim is None:
                raise ValueError("the intrinsic view needs the intrinsic input dimensions")
            self.mirna_intrinsic_projection = nn.Sequential(nn.Linear(mirna_intrinsic_dim, hidden), nn.ELU())
            self.disease_intrinsic_projection = nn.Sequential(nn.Linear(disease_intrinsic_dim, hidden), nn.ELU())
        # Functional-ANOVA decoder (Lengerich et al. AISTATS'20): logit = main effect of the miRNA +
        # main effect of the disease + interaction. Main effects are functions of node representations,
        # so cold nodes get them from their recovered views. The local view carries the node's
        # (standardised log) training degree, the association-dependent quantity behind the main effects.
        self.anova_decoder = anova_decoder
        if anova_decoder:
            self.degree_embedding = nn.ModuleDict({kind: nn.Linear(1, hidden) for kind in ("mirna", "disease")})
            self.main_effect = nn.ModuleDict({
                kind: nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1))
                for kind in ("mirna", "disease")
            })
        # Degree recovery: a node's training degree is an association-dependent attribute that is
        # missing for cold nodes. It is regressed from the association-free view on warm nodes; cold
        # nodes use the prediction, both in the local view and as an explicit main-effect term.
        self.degree_recovery = degree_recovery and anova_decoder
        if self.degree_recovery:
            self.degree_regressor = nn.ModuleDict({
                kind: nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1))
                for kind in ("mirna", "disease")
            })
            self.degree_main = nn.Parameter(torch.zeros(2))
        # With recover_homogeneous (intrinsic view only) both association-dependent views (local,
        # homogeneous) are missing for cold nodes and are recovered from the intrinsic view alone.
        self.n_missing_views = 2 if recover_homogeneous else 1
        # degree_mode "propagate": cold nodes take the mean (standardised log) degree of their warm
        # neighbours on the association-free KNN graph (feature propagation, Rossi et al. LoG'22).
        # degree_dropout: during training that fraction of warm nodes loses its association-dependent
        # inputs (observed degree and local view) and uses the recovered ones, as cold nodes do.
        if degree_mode not in {"regress", "propagate"}:
            raise ValueError(f"unknown degree mode: {degree_mode}")
        self.degree_mode = degree_mode
        self.degree_dropout = degree_dropout
        self._dropped = {}
        # Double-cold branch: a pair whose miRNA and disease both lack training edges has no
        # association-dependent view on either side, so neither DCP recovery nor the interaction
        # learned on warm pairs applies. It is scored by a separate branch that reads only the
        # association-free inputs: LightGCN on the intrinsic KNN graphs plus the warm-neighbour
        # propagated degree of both nodes (computed the same way, leave-self-out, for warm nodes in
        # training). Its parameters are drawn from a forked RNG and it is clipped separately, so the
        # main model trains bit for bit as without it.
        self.cold_branch = cold_branch
        if cold_branch:
            if mirna_intrinsic_dim is None or disease_intrinsic_dim is None:
                raise ValueError("the double-cold branch needs the intrinsic inputs and the training degrees")
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(20260924)
                self.branch_projection = nn.ModuleDict({
                    "mirna": nn.Sequential(nn.Linear(mirna_intrinsic_dim, hidden), nn.ELU()),
                    "disease": nn.Sequential(nn.Linear(disease_intrinsic_dim, hidden), nn.ELU()),
                })
                self.branch_interaction = nn.Sequential(
                    nn.Linear(hidden * 2, hidden), nn.ELU(), nn.Linear(hidden, 1)
                )
                self.branch_main = nn.Parameter(torch.zeros(2))
            self.branch_encoder = LightGCN(layers)
        # Neighbour-only branch encoding: a node is represented by its intrinsic neighbours alone
        # (self excluded), so warm training nodes are encoded as double-cold test nodes are and the
        # branch cannot memorise warm node identities.
        self.branch_neighbour_only = branch_neighbour_only
        # Purified branch interaction (functional ANOVA, Lengerich et al. AISTATS'20): a bilinear form of
        # embeddings centred over all nodes of each type has zero row and column means, so it cannot absorb
        # node propensity, which is carried by the propagated-degree main effects alone.
        # Degree-corrected offset (degree-corrected SBM, Karrer & Newman 2011: P(edge) ~ theta_i theta_j w):
        # the logit carries log(1 + degree) of both nodes with coefficient 1 -- observed for warm nodes,
        # propagated from warm neighbours for cold (and dropped) nodes. No parameters.
        self.degree_offset = degree_offset and degree_recovery and anova_decoder
        self.branch_bilinear = branch_bilinear and cold_branch
        if self.branch_bilinear:
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(20260925)
                self.branch_form = nn.Linear(hidden, hidden, bias=False)
                self.branch_bias = nn.Parameter(torch.zeros(1))
        if view_recovery:
            n_missing = self.n_missing_views
            n_available = len(self.view_names) - n_missing
            self.recover_local = nn.ModuleDict({
                kind: nn.Sequential(
                    nn.Linear(hidden * n_available, hidden), nn.ELU(), nn.Linear(hidden, hidden * n_missing)
                ) for kind in ("mirna", "disease")
            })
            self.recover_available = nn.ModuleDict({
                kind: nn.Sequential(
                    nn.Linear(hidden * n_missing, hidden), nn.ELU(), nn.Linear(hidden, hidden * n_available)
                ) for kind in ("mirna", "disease")
            })

    def set_training_degrees(self, train_association: torch.Tensor) -> None:
        """Store standardised log-degrees of the training association matrix."""

        if self.score_prior != "degree":
            return

        def standardise(values: torch.Tensor) -> torch.Tensor:
            values = torch.log1p(values.float())
            return (values - values.mean()) / values.std().clamp_min(1e-6)

        self.mirna_degree.copy_(standardise(train_association.sum(dim=1)))
        self.disease_degree.copy_(standardise(train_association.sum(dim=0)))

    def encode(
        self,
        x_mirna: torch.Tensor,
        x_disease: torch.Tensor,
        adjacency_local: torch.Tensor,
        adjacency_mirna: torch.Tensor,
        adjacency_disease: torch.Tensor,
        support_mirna: torch.Tensor | None = None,
        support_disease: torch.Tensor | None = None,
        global_prior_md: torch.Tensor | None = None,
        global_prior_dm: torch.Tensor | None = None,
        impute_mirna: torch.Tensor | None = None,
        impute_disease: torch.Tensor | None = None,
        cold_mirna: torch.Tensor | None = None,
        cold_disease: torch.Tensor | None = None,
        adjacency_union: torch.Tensor | None = None,
        intrinsic: dict[str, torch.Tensor] | None = None,
        degree: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden_m = self.mirna_projection(x_mirna)
        hidden_d = self.disease_projection(x_disease)

        local_all = self.local_encoder(
            torch.cat([hidden_m, hidden_d], dim=0), adjacency_local
        )
        local_m = local_all[: hidden_m.size(0)]
        local_d = local_all[hidden_m.size(0) :]
        if self.anova_decoder:
            if degree is None:
                raise ValueError("the ANOVA decoder needs the training degrees")
        if self.anova_decoder and not self.degree_recovery:
            local_m = local_m + self.degree_embedding["mirna"](degree["mirna"].unsqueeze(1))
            local_d = local_d + self.degree_embedding["disease"](degree["disease"].unsqueeze(1))
        # Nodes without training edges take the similarity-weighted mean of their warm
        # neighbours' local views (the operators are the identity on warm rows).
        if impute_mirna is not None:
            local_m = impute_mirna @ local_m
        if impute_disease is not None:
            local_d = impute_disease @ local_d

        if self.use_homogeneous_gate:
            homogeneous_input_m = hidden_m * torch.sigmoid(
                self.homogeneous_gate(hidden_m)
            )
            homogeneous_input_d = hidden_d * torch.sigmoid(
                self.homogeneous_gate(hidden_d)
            )
        else:
            homogeneous_input_m = hidden_m
            homogeneous_input_d = hidden_d
        homogeneous_m = self.homogeneous_encoder(
            homogeneous_input_m, adjacency_mirna
        )
        homogeneous_d = self.homogeneous_encoder(
            homogeneous_input_d, adjacency_disease
        )

        raw_views_m = [local_m, homogeneous_m]
        raw_views_d = [local_d, homogeneous_d]
        if self.use_global_view and self.global_mode == "intrinsic":
            if intrinsic is None:
                raise ValueError("the intrinsic view needs the intrinsic features and graphs")
            raw_views_m.append(self.homogeneous_encoder(
                self.mirna_intrinsic_projection(intrinsic["x_mirna"]), intrinsic["adj_mirna"]))
            raw_views_d.append(self.homogeneous_encoder(
                self.disease_intrinsic_projection(intrinsic["x_disease"]), intrinsic["adj_disease"]))
        elif self.use_global_view and self.global_mode == "diffusion":
            if adjacency_union is None:
                raise ValueError("diffusion global view requires the heterogeneous adjacency")
            start = torch.cat([hidden_m, hidden_d], dim=0)
            state = start
            for _ in range(self.diffusion_steps):
                state = (1.0 - self.diffusion_alpha) * (adjacency_union @ state) + self.diffusion_alpha * start
            raw_views_m.append(state[: hidden_m.size(0)])
            raw_views_d.append(state[hidden_m.size(0) :])
        elif self.use_global_view:
            if self.global_prior_scale is not None:
                if global_prior_md is None or global_prior_dm is None:
                    raise ValueError("structure-guided global view requires the prior matrices")
                global_m, global_d = self.cross_type_attention(
                    hidden_m, hidden_d, global_prior_md, global_prior_dm, self.global_prior_scale
                )
            else:
                global_m, global_d = self.cross_type_attention(hidden_m, hidden_d)
            raw_views_m.append(global_m)
            raw_views_d.append(global_d)

        degree_loss = hidden_m.new_tensor(0.0)
        if self.degree_recovery:
            if cold_mirna is None or cold_disease is None:
                raise ValueError("degree recovery requires the cold-node masks")
            used = {}
            for kind, views, cold in (("mirna", raw_views_m, cold_mirna), ("disease", raw_views_d, cold_disease)):
                if self.degree_mode == "propagate":
                    weights = intrinsic[f"adj_{kind}"] * (~cold).float().unsqueeze(0)
                    predicted = (weights @ degree[kind]) / weights.sum(dim=1).clamp_min(1e-12)
                else:
                    predicted = self.degree_regressor[kind](views[-1]).squeeze(1)
                    degree_loss = degree_loss + F.mse_loss(predicted[~cold], degree[kind][~cold])
                use_predicted = cold
                if self.training and self.degree_dropout > 0:
                    self._dropped[kind] = torch.rand(cold.shape, device=cold.device) < self.degree_dropout
                    use_predicted = cold | self._dropped[kind]
                used[kind] = torch.where(use_predicted, predicted, degree[kind])
                views[0] = views[0] + self.degree_embedding[kind](used[kind].unsqueeze(1))
            self._degree_used = used
            if self.degree_offset:
                self._offset = {kind: used[kind] * degree[f"{kind}_scale"] for kind in used}
        if self.cold_branch:
            branch = {}
            for kind in ("mirna", "disease"):
                adjacency = intrinsic[f"adj_{kind}"]
                warm = ~(cold_mirna if kind == "mirna" else cold_disease)
                weights = adjacency * warm.float().unsqueeze(0)
                propagated = (weights @ degree[kind]) / weights.sum(dim=1).clamp_min(1e-12)
                features = self.branch_projection[kind](intrinsic[f"x_{kind}"])
                if self.branch_neighbour_only:
                    embedding = (adjacency @ features) / adjacency.sum(dim=1, keepdim=True).clamp_min(1e-12)
                else:
                    embedding = self.branch_encoder(features, adjacency)
                if self.branch_bilinear:
                    embedding = embedding - embedding.mean(dim=0, keepdim=True)
                if self.degree_offset:
                    self._branch_offset = getattr(self, "_branch_offset", {})
                    self._branch_offset[kind] = propagated * degree[f"{kind}_scale"]
                branch[kind] = (embedding, propagated)
            self._branch = branch
        projected_m = [
            head(view) for head, view in zip(self.mirna_view_heads, raw_views_m)
        ]
        projected_d = [
            head(view) for head, view in zip(self.disease_view_heads, raw_views_d)
        ]
        recovery_loss = hidden_m.new_tensor(0.0)
        if self.view_recovery:
            if cold_mirna is None or cold_disease is None:
                raise ValueError("view recovery requires the cold-node masks")
            projected_m, loss_m = self._recover(projected_m, cold_mirna, "mirna")
            projected_d, loss_d = self._recover(projected_d, cold_disease, "disease")
            recovery_loss = loss_m + loss_d
        if self.fusion_mode == "meta":
            if support_mirna is None or support_disease is None:
                raise ValueError("meta fusion requires the support operators")
            fused_m, refined_m, attention_m = self.mirna_meta_fusion(
                projected_m, support_mirna, hidden_d
            )
            fused_d, refined_d, attention_d = self.disease_meta_fusion(
                projected_d, support_disease, hidden_m
            )
        else:
            fused_m, refined_m, attention_m = self.mirna_fusion(projected_m)
            fused_d, refined_d, attention_d = self.disease_fusion(projected_d)
        return {
            "mirna_fused": fused_m,
            "disease_fused": fused_d,
            "mirna_views": refined_m,
            "disease_views": refined_d,
            "mirna_view_attention": attention_m,
            "disease_view_attention": attention_d,
            "recovery_loss": recovery_loss,
            "degree_loss": degree_loss,
        }

    def _recover(
        self, views: list[torch.Tensor], cold: torch.Tensor, kind: str
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Dual prediction between the local view and the edge-free views.

        Warm nodes train both predictors (local <- available, available <- local); cold nodes,
        whose local view is empty, receive the predicted local view.
        """

        n_missing = self.n_missing_views
        missing, available = torch.cat(views[:n_missing], dim=1), torch.cat(views[n_missing:], dim=1)
        predicted = self.recover_local[kind](available)
        warm = ~cold
        loss = F.mse_loss(predicted[warm], missing[warm]) + F.mse_loss(
            self.recover_available[kind](missing[warm]), available[warm]
        )
        use_recovered = cold
        if self.training and self.view_dropout > 0:
            use_recovered = cold | (torch.rand(cold.shape, device=cold.device) < self.view_dropout)
        if self.training and self.degree_dropout > 0 and kind in self._dropped:
            use_recovered = use_recovered | self._dropped[kind]
        recovered = torch.where(use_recovered.unsqueeze(1), predicted, missing)
        return [*recovered.split(views[0].size(1), dim=1), *views[n_missing:]], loss

    def branch_parameters(self) -> list[nn.Parameter]:
        if not self.cold_branch:
            return []
        extra = [*self.branch_form.parameters(), self.branch_bias] if self.branch_bilinear else []
        return [*self.branch_projection.parameters(), *self.branch_interaction.parameters(), self.branch_main, *extra]

    def decode_branch(self, pairs: torch.Tensor) -> torch.Tensor:
        (z_m, deg_m), (z_d, deg_d) = self._branch["mirna"], self._branch["disease"]
        if self.branch_bilinear:
            logits = (self.branch_form(z_m[pairs[:, 0]]) * z_d[pairs[:, 1]]).sum(dim=1) + self.branch_bias
        else:
            logits = self.branch_interaction(torch.cat([z_m[pairs[:, 0]], z_d[pairs[:, 1]]], dim=1)).squeeze(1)
        logits = logits + self.branch_main[0] * deg_m[pairs[:, 0]] + self.branch_main[1] * deg_d[pairs[:, 1]]
        if self.degree_offset:
            logits = logits + self._branch_offset["mirna"][pairs[:, 0]] + self._branch_offset["disease"][pairs[:, 1]]
        return logits

    def decode(
        self,
        mirna_fused: torch.Tensor,
        disease_fused: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        pair_features = torch.cat(
            [mirna_fused[pairs[:, 0]], disease_fused[pairs[:, 1]]], dim=1
        )
        logits = self.predictor(pair_features).squeeze(1)
        if self.anova_decoder:
            logits = logits + self.main_effect["mirna"](mirna_fused[pairs[:, 0]]).squeeze(1)
            logits = logits + self.main_effect["disease"](disease_fused[pairs[:, 1]]).squeeze(1)
        if self.degree_recovery:
            logits = logits + self.degree_main[0] * self._degree_used["mirna"][pairs[:, 0]]
            logits = logits + self.degree_main[1] * self._degree_used["disease"][pairs[:, 1]]
        if self.degree_offset:
            logits = logits + self._offset["mirna"][pairs[:, 0]] + self._offset["disease"][pairs[:, 1]]
        if self.score_prior == "degree":
            logits = logits + self.degree_weight[0] * self.mirna_degree[pairs[:, 0]]
            logits = logits + self.degree_weight[1] * self.disease_degree[pairs[:, 1]]
        elif self.score_prior == "nodebias":
            logits = logits + self.mirna_bias[pairs[:, 0]] + self.disease_bias[pairs[:, 1]]
        return logits


def contrastive_loss(refined_views: torch.Tensor, temperature: float) -> torch.Tensor:
    if refined_views.size(1) < 2:
        return refined_views.sum() * 0.0
    loss = refined_views.new_tensor(0.0)
    count = 0
    for first in range(refined_views.size(1)):
        for second in range(first + 1, refined_views.size(1)):
            z_first = F.normalize(refined_views[:, first, :], dim=1)
            z_second = F.normalize(refined_views[:, second, :], dim=1)
            logits = z_first @ z_second.T / temperature
            labels = torch.arange(z_first.size(0), device=z_first.device)
            loss = loss + 0.5 * (
                F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.T, labels)
            )
            count += 1
    return loss / count


def proximity_loss(
    mirna_fused: torch.Tensor,
    disease_fused: torch.Tensor,
    positive_pairs: torch.Tensor,
    excluded_positive_mask: torch.Tensor,
    negative_per_positive: int,
) -> torch.Tensor:
    positive_mirna = positive_pairs[:, 0]
    positive_disease = positive_pairs[:, 1]
    positive_score = (
        mirna_fused[positive_mirna] * disease_fused[positive_disease]
    ).sum(dim=1)
    negative_probability = (~excluded_positive_mask[positive_mirna]).float()
    if (negative_probability.sum(dim=1) == 0).any():
        raise ValueError("at least one miRNA has no eligible proximity negative")
    negative_disease = torch.multinomial(
        negative_probability, negative_per_positive, replacement=True
    )
    negative_score = (
        mirna_fused[positive_mirna].unsqueeze(1)
        * disease_fused[negative_disease]
    ).sum(dim=2)
    return -F.logsigmoid(positive_score).mean() - F.logsigmoid(
        -negative_score
    ).mean()
