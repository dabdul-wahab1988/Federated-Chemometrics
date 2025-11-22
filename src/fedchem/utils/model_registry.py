"""Small, explicit registry for model label -> class mapping.

Keep the mapping explicit and minimal to avoid fragile dynamic imports.
Add entries here for any model class you want to expose via `PIPELINE.model`.
"""
from typing import Any, Callable, Dict

try:
    from fedchem.models.linear import LinearModel
except Exception:  # pragma: no cover - defensive import
    LinearModel = None  # type: ignore

try:
    from fedchem.models.pls import PLSModel
except Exception:  # pragma: no cover
    PLSModel = None  # type: ignore

try:
    from fedchem.models.parametric_pls import ParametricPLSModel
except Exception:  # pragma: no cover
    ParametricPLSModel = None  # type: ignore

# Add supported labels here. Values should be the class object (callable) or None
MODEL_REGISTRY: Dict[str, Callable[..., Any] | None] = {
    "LinearModel": LinearModel,
    "PLSModel": PLSModel,
    "ParametricPLSModel": ParametricPLSModel,
}

def instantiate_model(label: str | None, *args, **kwargs):
    """Instantiate a model by label.

    - label: string label from `PIPELINE.model` (case-sensitive). If None,
      will default to `LinearModel` when available.
    - args/kwargs passed to the model constructor.

    Raises ValueError if the label is unknown or class is unavailable.
    """
    lab = label or "LinearModel"
    cls = MODEL_REGISTRY.get(lab)
    if cls is None:
        raise ValueError(f"Unknown or unavailable model label: {lab}")
    return cls(*args, **kwargs)
