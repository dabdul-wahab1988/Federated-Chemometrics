from pathlib import Path
import json
import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_git_short_hash() -> Optional[str]:
    try:
        out = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.STDOUT)
        return out.decode('utf-8').strip()
    except Exception:
        return None


def human_readable_bytes(nbytes: Optional[int]) -> str:
    if nbytes is None:
        return '0 B'
    try:
        val = float(nbytes)
    except Exception:
        return '0 B'
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if val < 1024.0:
            return f"{val:3.1f} {unit}"
        val /= 1024.0
    return f"{val:.1f} PB"


def save_figure_and_metadata(fig, out_file: Path, metadata: dict, fmt: str = 'png', dpi: int = 300):
    """Save a matplotlib figure and a JSON metadata sidecar.
    """
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_file), format=fmt, dpi=dpi, bbox_inches='tight')
    meta_file = out_file.with_suffix(out_file.suffix + '.metadata.json')
    try:
        with open(meta_file, 'w', encoding='utf-8') as fh:
            json.dump(metadata, fh, indent=2)
    except Exception as e:
        logger.exception('Failed to write metadata file: %s', e)
