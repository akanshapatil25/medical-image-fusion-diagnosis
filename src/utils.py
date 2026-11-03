from __future__ import annotations

import json
from pathlib import Path


def ensure_parent(path_like):
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data, path_like):
    path = ensure_parent(path_like)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
    return path
