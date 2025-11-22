"""Federated Chemometrics package public API."""

from .simulator.core import SpectralSimulator
from .preprocess.pipeline import PreprocessPipeline
from .federated.orchestrator import FederatedOrchestrator
from .models.linear import LinearModel
from .conformal.predictor import ConformalPredictor
from .conformal.cqr import CQRConformal
from .data.registry import list_datasets, load_dataset, register_dataset, DatasetInfo
from .data.datasets import load_local_csv
from .data.profile import compute_profile
from .data.pca import compute_pca_profile
from .data import builtin as _data_builtin  # auto-register built-ins

__all__ = [
    "SpectralSimulator",
    "PreprocessPipeline",
    "FederatedOrchestrator",
    "LinearModel",
    "ConformalPredictor",
    "CQRConformal",
    "list_datasets",
    "load_dataset",
    "register_dataset",
    "DatasetInfo",
    "load_local_csv",
    "compute_profile",
    "compute_pca_profile",
    "_data_builtin",
]
