# MKGMDA

Code and data for **"MKGMDA: multi-level knowledge-guided graph learning for miRNA–disease association prediction in new diseases and miRNAs"**.

MKGMDA stands for **Multi-level Knowledge-Guided graph learning for miRNA–Disease Association prediction**.
Project repository: <https://github.com/wf845824/MKGMDA>.

A new miRNA or disease has no known association, so its association-derived inputs (association neighbours,
functional similarity, GIP kernel, degree) are missing or uninformative. MKGMDA estimates the missing local representation and degree of
such a *cold* entity from an intrinsic graph built from precursor sequence or MeSH similarity and gene-level knowledge,
uses the recovered degree explicitly in a degree-corrected decoder, and scores pairs of two cold entities with an
association-free branch.

## Installation

```bash
conda create -n mkgmda python=3.10
conda activate mkgmda
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.1.2 (CUDA 11.8), NumPy 1.26.4, and SciPy 1.11.4.
Commands use `cuda:0` by default; pass `--device cpu` to run on CPU.

## Data (`data/`)

| Folder | Source | Contents |
|---|---|---|
| `hmdd_v20`, `hmdd_v32` | processed benchmarks distributed with MINIMDA (<https://github.com/chengxu123/MINIMDA>) | miRNA and disease lists, known associations, disease MeSH similarity |
| `hmdd_v40` | HMDD v4.0 as processed by RGCNMDA (<https://doi.org/10.5281/zenodo.21358475>); MeSH similarity rebuilt with `scripts/build_hmdd_v40.py` | same layout |

Each folder also contains the association-free inputs used by the model:

* `mirna_sequence_similarity.npy`: cosine similarity of 3-mer frequency vectors of miRBase precursors
  (`mkgmda.similarity.sequence_similarity` applied to `data/mirbase_hairpin.fa`, the hairpin file downloaded from
  <https://www.mirbase.org/> in September 2026);
* `gene_knowledge_similarity.npz`: gene-knowledge similarities K^m (miRTarBase functional targets) and K^d
  (DisGeNET v7 disease genes), both computed over the HumanNet-XN network by `scripts/build_gene_knowledge.py`.
  No HMDD association enters these matrices.

MKGMDA computes every association-derived input from the training associations of each fold.

## Reproducing the results

Every model variant of the paper is a preset in `configs/presets.json`; `mkgmda` is the reported model.

```bash
# MKGMDA in the main comparison (Table 3): 3 datasets x 4 scenarios, five test folds each
for ds in hmdd_v20 hmdd_v32 hmdd_v40; do
  for sp in random colddisease coldmirna coldboth; do
    python run_mkgmda.py --preset mkgmda --dataset $ds --split $sp --eval test --save-scores \
        --output results/main/${ds}__${sp}.json
  done
done

# ablation (Table 4 and Supplementary Table S3), e.g. without dual prediction
python run_mkgmda.py --preset no_dual_prediction --dataset hmdd_v32 --split colddisease --output results/ablation/no_dcp.json
# the model without gene knowledge (row "w/o gene knowledge")
python run_mkgmda.py --preset no_gene_knowledge --dataset hmdd_v32 --split coldboth --output results/ablation/no_gene.json

# parameter sensitivity (Supplementary Fig. S1): the preset plus one override, grid in configs/sensitivity.json
python run_mkgmda.py --preset mkgmda --knn-k 10 --dataset hmdd_v32 --split coldmirna --output results/sens/k10.json

# Development splits carved from the HMDD v3.2 training folds
python run_mkgmda.py --preset mkgmda --dataset hmdd_v32 --split coldboth --eval dev --output results/dev/cb.json
```

Scenarios: `random` (five-fold split of the known associations), `colddisease` / `coldmirna` (all pairs of a held-out
fifth of the diseases or miRNAs are removed from training), and `coldboth` (miRNAs and diseases are held out
independently; test pairs have both endpoints held out). Folds use seed 42.
Unlabeled test pairs are drawn at 1:1 from the scenario-specific candidate set. The five training seeds are 42, 142, 242, 342, and 442.

## Case studies

```bash
# new disease: all HMDD v3.2 associations of the disease are removed and every input is rebuilt.
# The benchmark lists colorectal and rectal neoplasms separately, so the colon case removes the whole family.
python run_case_study.py --mode holdout --disease "Colonic Neoplasms" \
    --related "Colorectal Neoplasms" --related "Rectal Neoplasms" --output results/case/colon_holdout.csv
python run_case_study.py --mode holdout --disease "Breast Neoplasms" --output results/case/breast_holdout.csv
# novel associations: trained on all HMDD v3.2 associations, unrecorded miRNAs are ranked
python run_case_study.py --mode novel --disease "Colonic Neoplasms" \
    --related "Colorectal Neoplasms" --related "Rectal Neoplasms" --output results/case/colon_novel.csv
python run_case_study.py --mode novel --disease "Breast Neoplasms" --output results/case/breast_novel.csv
# evidence from HMDD v4.0, dbDEMC 3.0, TCGA, and (interpretation only) KEGG pathway targets,
# with the same rule for the top-30 and for all candidates
python analysis/case_evidence.py --ranked results/case/colon_holdout.csv --disease colon --output results/case/colon_evidence.csv
```

The complete rankings of both case studies, with the evidence for every candidate, are provided as CSV files in
[`results/case_studies/`](results/case_studies/) (column descriptions in its README).

`analysis/case_evidence.py` reads third-party files that we do not redistribute; place them in
`external_data/case_evidence/`:

| File | Source |
|---|---|
| `hmdd_v4_alldata.txt` | HMDD v4.0, <http://www.cuilab.cn/hmdd> |
| `dbdemc_MiRExpAll.txt`, `dbdemc_validations.txt` | dbDEMC 3.0, <https://www.biosino.org/dbDEMC/> |
| `TCGA-COAD.mirna.tsv.gz`, `TCGA-BRCA.mirna.tsv.gz` | GDC miRNA expression via the UCSC Xena GDC hub, <https://gdc.xenahubs.net/> |
| `kegg_hsa05210.txt`, `kegg_hsa05224.txt` | KEGG pathway gene lists, <https://www.kegg.jp/> |

and `external_data/miRTarBase_10_hsa_MTI.csv` (miRTarBase 10, <https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_2025/>).

## Rebuilding the association-free inputs (optional)

```bash
pip install numba==0.60.0 pandas==2.3.2
python scripts/build_gene_knowledge.py   # external_data/: HumanNet-XN.tsv, all_gene_disease_associations.tsv.gz, miRTarBase_10_hsa_MTI.csv
python scripts/build_hmdd_v40.py         # external_data/: MeSH 2026 XML (mesh/), RGCNMDA release (rgcnmda/)
```

HumanNet v2: <https://www.inetbio.org/humannetv2/>; DisGeNET v7: <https://www.disgenet.com/>; MeSH: <https://www.nlm.nih.gov/mesh/>.
The sequence similarity is rebuilt with `mkgmda.similarity.load_hairpins` and `sequence_similarity` from `data/mirbase_hairpin.fa`.

## Repository layout

```
mkgmda/          model (model.py), training loop (training.py), graphs and sampling (data.py),
                  fold-wise inputs (preprocessing.py), splits (splits.py), intrinsic similarities (similarity.py)
run_mkgmda.py    training and evaluation under the strict protocol
run_case_study.py case studies
analysis/         case-study evidence
configs/          presets of all model variants; sensitivity grid
scripts/          rebuilding of the gene-knowledge similarities and of the HMDD v4.0 benchmark
data/             benchmarks, association-free similarities, and the miRBase hairpin file
```

## Citation

Citation details will be added upon publication.
