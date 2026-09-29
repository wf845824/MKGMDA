"""Evidence for a ranked case-study list from sources that do not enter the model: HMDD v4.0 records, dbDEMC 3.0
low-throughput validations, and TCGA tumour-versus-normal expression; plus, as interpretation only, validated miRTarBase
targets in the KEGG disease pathway with a hypergeometric enrichment test (miRTarBase also enters the model).

  python analysis/case_evidence.py --ranked results/case/colon_holdout.csv --disease colon --output results/case/colon_evidence.csv

The same rule is applied to the top-k and to the full ranked list: a miRNA is supported by HMDD if it is a held-out
HMDD v3.2 association (holdout runs; with --related, of any disease of the family) or has an HMDD v4.0 record for the
disease family below. Colon neoplasms are treated as the colorectal family, because the benchmark lists colorectal and
rectal neoplasms as separate diseases (run the case study with --related for both).

Inputs are read from external_data/case_evidence/ (see README): TCGA-<PROJECT>.mirna.tsv.gz (UCSC Xena GDC hub),
dbdemc_MiRExpAll.txt and dbdemc_validations.txt (dbDEMC 3.0), kegg_<pathway>.txt (KEGG gene lists),
hmdd_v4_alldata.txt (HMDD v4.0), and external_data/miRTarBase_10_hsa_MTI.csv.
"""
import argparse
import csv
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "external_data" / "case_evidence"
MTI = ROOT / "external_data" / "miRTarBase_10_hsa_MTI.csv"
STRONG = ("luciferase", "western blot", "qrt-pcr", "qpcr", "real-time pcr")
DISEASES = {
    "colon": dict(tcga=["COAD"], dbdemc={"colorectal cancer", "colon cancer"}, kegg="hsa05210",
                  hmdd={"Colonic Neoplasms", "Colorectal Neoplasms", "Rectal Neoplasms", "Colon Adenocarcinoma",
                        "colon carcinoma", "Colorectal Adenocarcinoma", "rectum adenocarcinoma",
                        "Colorectal Neoplasms, Hereditary Nonpolyposis"}),
    "breast": dict(tcga=["BRCA"], dbdemc={"breast cancer"}, kegg="hsa05224",
                   hmdd={"Breast Neoplasms", "Breast Neoplasms, Basal Cell", "Carcinoma, Ductal, Breast",
                         "Triple Negative Breast Neoplasms", "Inflammatory Breast Neoplasms",
                         "Her2-Positive Breast Cancer", "Quadruple Negative Breast Cancer",
                         "Bone-Associated Metastatic Breast Cancer"}),
}


def family(name):
    """Map precursor (hsa-mir-16-1) and mature (hsa-miR-16-5p) names to one key (mir-16)."""
    n = name.strip().lower().replace("hsa-", "").replace("*", "")
    n = re.sub(r"-(5p|3p)$", "", n)
    m = re.match(r"^((?:mir|let)-[0-9]+[a-z]*)-[0-9]+$", n)
    return m.group(1) if m else n


def bh(p):
    p = np.asarray(p, float); order = np.argsort(p); q = np.empty_like(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    return q


def tcga_de(projects):
    rpm, tumour_cols, normal_cols = defaultdict(lambda: 0.0), [], []
    rows = {}
    for proj in projects:
        with gzip.open(SRC / f"TCGA-{proj}.mirna.tsv.gz", "rt") as fh:
            header = fh.readline().rstrip("\n").split("\t")[1:]
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                rows.setdefault(family(parts[0]), []).append((proj, np.array(parts[1:], float)))
            tumour_cols.append((proj, [i for i, s in enumerate(header) if s[13:15] == "01"]))
            normal_cols.append((proj, [i for i, s in enumerate(header) if s[13:15] == "11"]))
    stats = {}
    for fam, entries in rows.items():
        tum, nor = [], []
        for proj, values in [(p, sum(2 ** v - 1 for q, v in entries if q == p)) for p in projects]:
            tum.append(values[dict(tumour_cols)[proj]]); nor.append(values[dict(normal_cols)[proj]])
        tum, nor = np.log2(np.concatenate(tum) + 1), np.log2(np.concatenate(nor) + 1)
        if max(2 ** tum.mean() - 1, 2 ** nor.mean() - 1) < 1.0:
            continue  # not expressed
        stats[fam] = dict(log2fc=float(tum.mean() - nor.mean()), p=float(mannwhitneyu(tum, nor).pvalue))
    q = bh([s["p"] for s in stats.values()])
    for s, qv in zip(stats.values(), q):
        s["fdr"] = float(qv)
    n_t = sum(len(c) for _, c in tumour_cols); n_n = sum(len(c) for _, c in normal_cols)
    return stats, n_t, n_n


def dbdemc(cancers):
    ht, lt = defaultdict(lambda: {"UP": set(), "DOWN": set()}), defaultdict(set)
    with open(SRC / "dbdemc_MiRExpAll.txt") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["CancerType"] in cancers and r["Species"] == "Homo sapiens":
                ht[family(r["miRNA_ID"])][r["Status"]].add(r["ExperimentID"])
    with open(SRC / "dbdemc_validations.txt") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["Cancer_Type"] in cancers:
                lt[family(r["miRNA_ID"])].add(r["PubMed_ID"])
    return ht, lt


def mirtarbase():
    targets, genes = defaultdict(dict), {}
    with open(MTI, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["Species (miRNA)"] != "hsa" or r["Support Type"] != "Functional MTI":
                continue
            if not any(k in r["Experiments"].lower() for k in STRONG) or not r["Target Gene (Entrez ID)"]:
                continue
            gid = int(float(r["Target Gene (Entrez ID)"]))
            genes[gid] = r["Target Gene"]
            targets[family(r["miRNA"])].setdefault(gid, set()).add(r["References (PMID)"].split(".")[0])
    return targets, genes


def kegg(pathway):
    with open(SRC / f"kegg_{pathway}.txt") as fh:
        return {int(line.split("\t")[1].strip().split(":")[1]) for line in fh if line.strip()}


def base_name(name: str) -> str:
    """Collapse genomic copy numbers, e.g. hsa-mir-16-1 -> hsa-mir-16."""
    return re.sub(r"^(hsa-(?:mir|let)-[0-9a-z]+?)-\d+$", r"\1", name.lower().strip())


def hmdd_v4(names):
    norm = lambda s: " ".join(s.casefold().split())
    wanted = {norm(n) for n in names}
    found = defaultdict(set)
    with open(SRC / "hmdd_v4_alldata.txt", encoding="utf-8", errors="ignore") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if norm(row["disease"]) in wanted:
                found[base_name(row["miRNA"])].add(row["PMID"])
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ranked", required=True, help="output of run_case_study.py")
    p.add_argument("--disease", choices=sorted(DISEASES), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--top", type=int, default=30)
    a = p.parse_args()
    cfg = DISEASES[a.disease]
    ranked = list(csv.DictReader(open(a.ranked)))
    targets, gene_names = mirtarbase(); universe = set(gene_names)
    de, n_t, n_n = tcga_de(cfg["tcga"]); ht, lt = dbdemc(cfg["dbdemc"]); path = kegg(cfg["kegg"]) & universe
    v4 = hmdd_v4(cfg["hmdd"])
    rows = []
    for r in ranked:
        fam = family(r["mirna"]); tg = targets.get(fam, {})
        hits = sorted(set(tg) & path, key=lambda g: -len(tg[g])); d = de.get(fam)
        held = r.get("held_out_family_hmdd_v32", r.get("held_out_hmdd_v32", ""))
        rows.append(dict(rank=int(r["rank"]), mirna=r["mirna"], held_out_hmdd_v32=held,
                         held_out_target=r.get("held_out_hmdd_v32", ""),
                         hmdd_v4=";".join(sorted(v4.get(base_name(r["mirna"]), set()))),
                         dbdemc_lowthroughput=";".join(sorted(lt.get(fam, set()))),
                         tcga_log2fc=d["log2fc"] if d else "", tcga_fdr=d["fdr"] if d else "",
                         pathway_targets=";".join(gene_names[g] for g in hits),
                         pathway_p=float(hypergeom.sf(len(hits) - 1, len(universe), len(path), len(tg))) if hits else 1.0))
    for x, q in zip(rows, bh([x["pathway_p"] for x in rows])):
        x["pathway_fdr"] = float(q)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def rate(sel, f):
        sel = list(sel); return [sum(bool(f(x)) for x in sel), len(sel)]
    top = rows[: a.top]; expressed = [x for x in rows if x["tcga_fdr"] != ""]
    summary = {"tcga_samples": {"tumour": n_t, "normal": n_n}}
    for key, f in (("hmdd_v4", lambda x: x["hmdd_v4"]), ("dbdemc_lowthroughput", lambda x: x["dbdemc_lowthroughput"]),
                   ("pathway_target", lambda x: x["pathway_targets"]), ("pathway_enriched", lambda x: x["pathway_fdr"] < 0.05)):
        summary[key] = {"top": rate(top, f), "all": rate(rows, f)}
    hmdd = lambda x: x["held_out_hmdd_v32"] == "True" or bool(x["hmdd_v4"])
    summary["hmdd"] = {"top": rate(top, hmdd), "all": rate(rows, hmdd)}
    summary["confirmed"] = {"top": rate(top, lambda x: hmdd(x) or x["dbdemc_lowthroughput"]),
                            "all": rate(rows, lambda x: hmdd(x) or x["dbdemc_lowthroughput"])}
    if rows[0]["held_out_target"] != "":
        def rank_auc(flags):
            y = np.asarray(flags, bool); pos = np.flatnonzero(y) + 1; n_pos, n_neg = y.sum(), (~y).sum()
            return float(1 - (pos - 1 - np.arange(n_pos)).sum() / (n_pos * n_neg))
        summary["rank_auc_target"] = rank_auc([x["held_out_target"] == "True" for x in rows])
        summary["rank_auc_family"] = rank_auc([x["held_out_hmdd_v32"] == "True" for x in rows])
    summary["tcga_de"] = {"top": rate([x for x in top if x["tcga_fdr"] != ""], lambda x: x["tcga_fdr"] < 0.05),
                          "all": rate(expressed, lambda x: x["tcga_fdr"] < 0.05)}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
