from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
from pathlib import Path


def hash_config(cfg: Dict[str, Any]) -> str:
    data = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass
class RunManifest:
    config_hash: str
    seeds: Dict[str, int]
    env: Dict[str, str]

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))


def capture_env(keys: Optional[list[str]] = None) -> Dict[str, str]:
    keys = keys or ["OS", "COMPUTERNAME", "USERNAME", "PROCESSOR_IDENTIFIER", "PYTHONPATH"]
    out: Dict[str, str] = {}
    for k in keys:
        v = os.environ.get(k)
        if v:
            out[k] = v
    return out

