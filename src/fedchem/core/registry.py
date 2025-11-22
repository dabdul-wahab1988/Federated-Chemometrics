from __future__ import annotations

from typing import Any, Callable, Dict


_MODELS: Dict[str, Any] = {}
_ALGOS: Dict[str, Any] = {}


def register_model(name: str) -> Callable[[Any], Any]:
    def deco(cls: Any) -> Any:
        _MODELS[name.lower()] = cls
        return cls
    return deco


def register_algo(name: str) -> Callable[[Any], Any]:
    def deco(cls: Any) -> Any:
        _ALGOS[name.lower()] = cls
        return cls
    return deco


def get_model(name: str) -> Any:
    return _MODELS[name.lower()]


def get_algo(name: str) -> Any:
    return _ALGOS[name.lower()]


def list_models() -> Dict[str, Any]:
    return dict(_MODELS)


def list_algos() -> Dict[str, Any]:
    return dict(_ALGOS)
