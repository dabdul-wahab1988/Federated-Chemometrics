from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class DPConfig:
    epsilon: Optional[float]
    delta: float
    clip_feature: Optional[float]
    clip_target: Optional[float]
    subsample_q: Optional[float]


def build_privacy_entry(
    site_id: str,
    *,
    reconstruction_error: float,
    membership_auc: float | None,
    dp_config: DPConfig,
    dp_metadata: Dict[str, Any] | None,
    communication_bytes: int,
    paired_samples: int | None,
) -> Dict[str, Any]:
    """Assemble a JSON-serialisable privacy metadata entry."""
    entry: Dict[str, Any] = {
        "site_id": site_id,
        "reconstruction_error": float(reconstruction_error),
        "membership_auc": None if membership_auc is None else float(membership_auc),
        "communication_bytes": int(communication_bytes),
        "paired_samples": None if paired_samples is None else int(paired_samples),
        "dp_config": asdict(dp_config),
    }
    if dp_metadata is not None:
        entry["dp_metadata"] = dp_metadata
    return entry

