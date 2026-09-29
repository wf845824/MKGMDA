"""CoReMDA: contrastive multi-view learning with missing-information recovery for cold-start miRNA-disease association prediction."""

from .model import CoReMDA
from .preprocessing import BenchmarkData, build_fold_inputs, load_benchmark
