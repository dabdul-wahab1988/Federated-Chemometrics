from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol
from ..types import FloatArray


@dataclass
class ClientBatch:
    X: FloatArray
    y: FloatArray


class ParametricModel(Protocol):
    def fit(self, X: FloatArray, y: FloatArray) -> "ParametricModel":
        ...

    def predict(self, X: FloatArray) -> FloatArray:
        ...

    def get_params(self) -> Dict[str, FloatArray]:
        ...

    def set_params(self, params: Dict[str, FloatArray]) -> None:
        ...
