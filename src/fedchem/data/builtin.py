from __future__ import annotations

from .registry import DatasetInfo, register_dataset
from .parsers import parse_tecator_mat


# Tecator NIR dataset (meat samples; commonly distributed as MAT)
# Source reference: Models Life Science (KU) Food Data repository
# URL is provided for convenience; downloads require explicit consent via FEDCHEM_ALLOW_DOWNLOAD=1
register_dataset(
    DatasetInfo(
        name="tecator",
        license="Research/educational use (verify source terms)",
        # Try a known ASCII mirror first (lib.stat), then the historical KU .mat URL.
        url="http://www.models.life.ku.dk/fooddata/tecator.mat",
        urls=[
            "http://lib.stat.cmu.edu/datasets/tecator",
            "http://www.models.life.ku.dk/fooddata/tecator.mat",
        ],
        filename="tecator.mat",
        sha256=None,  # Consider setting if you mirror a stable copy
        parser=parse_tecator_mat,
    )
)

