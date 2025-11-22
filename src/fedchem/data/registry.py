from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Any, List
import hashlib
import os
import urllib.request

from ..types import FloatArray
from . import alt_sources
import numpy as np


DatasetParser = Callable[[Path], Tuple[FloatArray, Optional[FloatArray], List[float], Dict[str, Any]]]


@dataclass
class DatasetInfo:
    name: str
    license: str
    filename: Optional[str]
    sha256: Optional[str]
    parser: DatasetParser
    url: Optional[str] = None
    urls: Optional[List[str]] = None


_REGISTRY: Dict[str, DatasetInfo] = {}


def register_dataset(info: DatasetInfo) -> None:
    _REGISTRY[info.name.lower()] = info


def list_datasets() -> Dict[str, DatasetInfo]:
    return dict(_REGISTRY)


def get_dataset(name: str) -> DatasetInfo:
    return _REGISTRY[name.lower()]


def default_cache_dir() -> Path:
    # Cross-platform cache directory
    # Allow overriding cache dir via environment for project-local caches
    env = os.environ.get("FEDCHEM_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    # For testing, allow forcing the perceived OS name via an env var so tests
    # can exercise non-nt branches without mutating the global os.name.
    forced_name = os.environ.get("FEDCHEM_FORCE_OS_NAME")
    name_to_check = forced_name if forced_name is not None else os.name
    if name_to_check == "nt":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        return Path(base) / "fedchem" / "datasets"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "fedchem" / "datasets"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _maybe_download(info: DatasetInfo, cache_dir: Path, allow_download: Optional[bool]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not info.filename and not info.url:
        # Custom parser expected to resolve its own location (e.g., closure over a local path)
        return cache_dir
    if not info.filename:
        raise FileNotFoundError("Dataset has no filename; supply a local file or custom parser.")
    dest = cache_dir / info.name / info.filename
    if dest.exists():
        # verify checksum if provided
        if info.sha256 and _sha256_file(dest) != info.sha256:
            dest.unlink(missing_ok=True)
        else:
            return dest
    if allow_download is None:
        allow_download = os.environ.get("FEDCHEM_ALLOW_DOWNLOAD", "0") == "1"
    # If downloads are disabled and there are no candidate URLs, surface a
    # FileNotFoundError (no way to obtain the dataset). If downloads are
    # disabled but URLs exist, raise a PermissionError to indicate the user
    # must explicitly enable downloads.
    if not allow_download:
        if not info.url and not info.urls:
            raise FileNotFoundError(f"Dataset {info.name} not cached and no URL provided. Place file at {dest}")
        raise PermissionError(
            f"Refusing to download dataset {info.name}. Set FEDCHEM_ALLOW_DOWNLOAD=1 or pass allow_download=True."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    # candidate URLs: prefer explicit list, fall back to single url
    candidates: List[str] = []
    if info.urls:
        candidates.extend(info.urls)
    if info.url:
        candidates.append(info.url)
    if not candidates:
        raise FileNotFoundError(f"Dataset {info.name} not cached and no URL provided. Place file at {dest}")

    last_err: Optional[Exception] = None
    for candidate in candidates:
        try:
            # Prefer urlretrieve (easier to monkeypatch in tests); fall back to urlopen
            try:
                # urlretrieve writes directly to the destination path
                urllib.request.urlretrieve(candidate, str(tmp))  # type: ignore[attr-defined]
            except Exception:
                # Use urlopen so we can inspect headers and avoid saving HTML/error pages as dataset files
                with urllib.request.urlopen(candidate) as resp:  # nosec - controlled by registry
                    content_type = resp.getheader("Content-Type", "") or ""
                    prefix = resp.read(512)
                    try:
                        prefix_text = prefix.decode("utf-8", errors="ignore").lstrip()
                    except Exception:
                        prefix_text = ""
                    if content_type.startswith("text/") or prefix_text.startswith("<"):
                        # If the candidate returned HTML, for known datasets try dataset-specific fallbacks
                        if info.name.lower() == "tecator":
                            alt_sources.fetch_tecator_from_libstat(dest)
                            return dest
                        # otherwise try next candidate
                        last_err = ValueError(f"Dataset URL {candidate} returned non-binary content (Content-Type={content_type})")
                        continue
                    # write the prefix and the rest to the tmp file
                    with tmp.open("wb") as f:
                        f.write(prefix)
                        chunk = resp.read(8192)
                        while chunk:
                            f.write(chunk)
                            chunk = resp.read(8192)
            # verify checksum if provided
            if info.sha256:
                got = _sha256_file(tmp)
                if got != info.sha256:
                    tmp.unlink(missing_ok=True)
                    raise ValueError(f"Checksum mismatch for {info.name}: expected {info.sha256}, got {got}")
            tmp.replace(dest)
            return dest
        except Exception as e:
            last_err = e
            # try next candidate
            continue

    # If we exhausted candidates, raise last error or generic
    if last_err:
        raise last_err
    raise FileNotFoundError(f"Dataset {info.name} not cached and no URL provided. Place file at {dest}")


def load_dataset(name: str, *, cache_dir: Optional[Path] = None, allow_download: Optional[bool] = None) -> Tuple[FloatArray, Optional[FloatArray], List[float], Dict[str, Any]]:
    info = get_dataset(name)
    cache_dir = cache_dir or default_cache_dir()
    path = _maybe_download(info, cache_dir, allow_download)
    X, y, wavelengths, meta = info.parser(path)
    meta = dict(meta)
    meta.update({"dataset": info.name, "path": str(path), "license": info.license})
    return X, y, wavelengths, meta
