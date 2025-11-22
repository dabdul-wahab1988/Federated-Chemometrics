from __future__ import annotations

from typing import Protocol
from ..types import FloatArray


class BaseModel(Protocol):
    def fit(self, X: FloatArray, y: FloatArray) -> "BaseModel":
        ...

    def predict(self, X: FloatArray) -> FloatArray:
        ...
