from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class ClientConfig(BaseModel):
    n_samples: int = Field(ge=1)
    modality: str = "NIR"


class SimulatorConfig(BaseModel):
    n_sites: int = Field(ge=1)
    n_samples_per_site: int = Field(ge=1)
    wavelengths: List[float]
    modalities: List[str]
    seed: Optional[int] = None


class PreprocessConfig(BaseModel):
    snv: bool = False
    msc: bool = False
    detrend: bool = False
    savgol: Optional[dict] = None
    scale: Optional[str] = None


class FederatedConfig(BaseModel):
    rounds: int = Field(ge=1)
    algo: str = "fedavg"
    prox_mu: float = 0.0
    clip_norm: Optional[float] = None
    dp_noise_std: Optional[float] = None


class ExperimentConfig(BaseModel):
    simulator: SimulatorConfig
    preprocess: PreprocessConfig
    federated: FederatedConfig

