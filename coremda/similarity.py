"""Association-free (intrinsic) similarities and the profile-aware integration of association-derived ones."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np

from .preprocessing import BenchmarkData


def fuse_gene_knowledge(base: np.ndarray, gene: np.ndarray, mode: str = "max") -> np.ndarray:
    """Intrinsic similarity X: max(base, K) for pairs whose two entities both have gene annotation, else base."""
    annotated = ((gene > 0) & ~np.eye(len(gene), dtype=bool)).any(axis=1)
    both = annotated[:, None] & annotated[None, :]
    fused = np.maximum(base, gene) if mode == "max" else (base + gene) / 2
    out = np.where(both, fused, base).astype(np.float32)
    np.fill_diagonal(out, 1.0)
    return out


def profile_aware_integration(curated: np.ndarray, gip: np.ndarray, has_profile: np.ndarray) -> np.ndarray:
    """Integrated similarity S: curated similarity where positive, else GIP only between two warm nodes."""
    defined = has_profile[:, None] & has_profile[None, :]
    integrated = np.where(curated > 0, curated, np.where(defined, gip, 0.0)).astype(np.float32)
    np.fill_diagonal(integrated, 1.0)
    return integrated


def load_intrinsic(data_root: Path, bench: BenchmarkData, gene_knowledge: str = "max") -> tuple[np.ndarray, np.ndarray]:
    """Base intrinsic similarities (miRNA precursor 3-mer cosine, disease MeSH), optionally fused with gene knowledge."""
    folder = Path(data_root) / bench.name
    sequence = np.load(folder / "mirna_sequence_similarity.npy").astype(np.float32)
    semantic = np.asarray(bench.disease_semantic_similarity, dtype=np.float32).copy()
    np.fill_diagonal(semantic, 1.0)
    if gene_knowledge != "none":
        with np.load(folder / "gene_knowledge_similarity.npz", allow_pickle=False) as g:
            assert [str(x) for x in g["mirna_names"]] == [str(x) for x in bench.mirna_names]
            assert [str(x) for x in g["disease_names"]] == [str(x) for x in bench.disease_names]
            sequence = fuse_gene_knowledge(sequence, g["mirna"].astype(np.float32), gene_knowledge)
            semantic = fuse_gene_knowledge(semantic, g["disease"].astype(np.float32), gene_knowledge)
    return sequence, semantic


# ---- rebuilding the miRNA sequence similarity (data/<dataset>/mirna_sequence_similarity.npy) from miRBase ----
NUCS = "ACGU"


def load_hairpins(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    name = None
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                token = line[1:].split()[0]
                name = token.lower() if token.startswith("hsa-") else None
                if name:
                    seqs[name] = ""
            elif name:
                seqs[name] += line.strip().upper().replace("T", "U")
    return seqs


def resolve(name: str, seqs: dict[str, str]) -> str | None:
    key = name.lower()
    if key in seqs:
        return seqs[key]
    for suffix in ("-1", "-2", "-3", "a", "b", "a-1", "a-2", "b-1"):
        if key + suffix in seqs:
            return seqs[key + suffix]
    stripped = key.rsplit("-", 1)[0] if key.rsplit("-", 1)[-1].isdigit() else key
    if stripped in seqs:
        return seqs[stripped]
    hits = [value for alias, value in seqs.items() if alias.startswith(key)]
    return hits[0] if len(hits) == 1 else None


def kmer_vector(seq: str, k: int = 3) -> np.ndarray:
    kmers = ["".join(item) for item in product(NUCS, repeat=k)]
    index = {kmer: i for i, kmer in enumerate(kmers)}
    vec = np.zeros(len(kmers), dtype=np.float32)
    for i in range(len(seq) - k + 1):
        j = index.get(seq[i : i + k])
        if j is not None:
            vec[j] += 1.0
    total = float(vec.sum())
    return vec / total if total > 0 else vec


def sequence_similarity(names, seqs: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity of 3-mer frequency vectors of miRBase precursors; zero for miRNAs without a precursor."""
    vectors = np.zeros((len(names), 4 ** 3), dtype=np.float32)
    found = np.zeros(len(names), dtype=bool)
    for i, name in enumerate(names):
        seq = resolve(str(name), seqs)
        if seq is None or len(seq) < 10:
            continue
        found[i] = True
        vectors[i] = kmer_vector(seq)
    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    similarity = unit @ unit.T
    similarity[~found, :] = 0.0
    similarity[:, ~found] = 0.0
    np.fill_diagonal(similarity, 1.0)
    return similarity.astype(np.float32), found
