"""
session.py — where a run lives on disk.

Everything a run needs (receptor, ligand, poses, complex) goes in one
directory, and the choices made along the way go in a manifest beside them.

That matters for more than tidiness: clicking a residue in the sequence grid
navigates the page, which ends the Streamlit session. Because the manifest is
on disk and the run id is in the URL, the app picks up exactly where it was.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_ROOT = os.environ.get("COVDOCK_RUNS", os.path.abspath("runs"))
MANIFEST = "manifest.json"

# A run id reaches us from the ?s= URL parameter, and it is joined onto the
# runs root to make a directory that delete() hands to shutil.rmtree. Anything
# not matching the id create() generates is refused, so "?s=../../.." cannot
# point that outside the root.
RUN_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")


def valid_run_id(run_id: Optional[str]) -> bool:
    return bool(run_id and RUN_ID_RE.match(run_id))


@dataclass
class WorkSession:
    run_id: str
    root: str = DEFAULT_ROOT
    data: Dict[str, Any] = field(default_factory=dict)

    # ── paths ────────────────────────────────────────────────────────────────
    @property
    def dir(self) -> str:
        return os.path.join(self.root, self.run_id)

    def path(self, *parts: str) -> str:
        """
        A path inside this run's directory.

        Refuses to escape it: file names reach here from an upload's extension
        among other places, and a name containing ".." would otherwise resolve
        outside the run.
        """
        target = os.path.abspath(os.path.join(self.dir, *parts))
        root = os.path.abspath(self.dir)
        if target != root and not target.startswith(root + os.sep):
            raise ValueError(f"path escapes the run directory: {os.path.join(*parts)}")
        return target

    @property
    def manifest_path(self) -> str:
        return self.path(MANIFEST)

    # ── lifecycle ────────────────────────────────────────────────────────────
    @classmethod
    def create(cls, root: str = DEFAULT_ROOT) -> "WorkSession":
        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        session = cls(run_id=run_id, root=root)
        os.makedirs(session.dir, exist_ok=True)
        session.data = {"created": time.time(), "run_id": run_id}
        session.save()
        return session

    @classmethod
    def load(cls, run_id: str, root: str = DEFAULT_ROOT) -> Optional["WorkSession"]:
        if not valid_run_id(run_id):
            return None          # includes "../.." and anything else invented
        session = cls(run_id=run_id, root=root)
        if not os.path.isfile(session.manifest_path):
            return None
        try:
            with open(session.manifest_path) as fh:
                session.data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None
        return session

    @classmethod
    def load_or_create(cls, run_id: Optional[str],
                       root: str = DEFAULT_ROOT) -> "WorkSession":
        if run_id:
            existing = cls.load(run_id, root=root)
            if existing is not None:
                return existing
        return cls.create(root=root)

    def save(self) -> None:
        os.makedirs(self.dir, exist_ok=True)
        self.data["updated"] = time.time()
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.data, fh, indent=2, default=str)
        os.replace(tmp, self.manifest_path)

    def delete(self) -> None:
        # rmtree on a caller-supplied path: refuse anything that is not a run
        # directory sitting directly under the root.
        target = os.path.abspath(self.dir)
        root = os.path.abspath(self.root)
        if not valid_run_id(self.run_id) or os.path.dirname(target) != root:
            raise ValueError(f"refusing to delete {target}: not a run directory")
        shutil.rmtree(target, ignore_errors=True)

    # ── manifest access ──────────────────────────────────────────────────────
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any, save: bool = True) -> None:
        self.data[key] = value
        if save:
            self.save()

    def update(self, **values: Any) -> None:
        self.data.update(values)
        self.save()

    def clear(self, *keys: str) -> None:
        for key in keys:
            self.data.pop(key, None)
        self.save()

    # ── files ────────────────────────────────────────────────────────────────
    def write_bytes(self, name: str, payload: bytes) -> str:
        os.makedirs(self.dir, exist_ok=True)
        target = self.path(name)
        with open(target, "wb") as fh:
            fh.write(payload)
        return target

    def write_text(self, name: str, text: str) -> str:
        os.makedirs(self.dir, exist_ok=True)
        target = self.path(name)
        with open(target, "w") as fh:
            fh.write(text)
        return target

    def read_text(self, name: str) -> Optional[str]:
        target = self.path(name)
        if not os.path.isfile(target):
            return None
        with open(target, errors="replace") as fh:
            return fh.read()

    def has(self, name: str) -> bool:
        target = self.path(name)
        return os.path.isfile(target) and os.path.getsize(target) > 0

    def files(self) -> List[str]:
        if not os.path.isdir(self.dir):
            return []
        return sorted(f for f in os.listdir(self.dir) if f != MANIFEST)

    def size_mb(self) -> float:
        total = 0
        for dirpath, _dirs, names in os.walk(self.dir):
            for name in names:
                total += os.path.getsize(os.path.join(dirpath, name))
        return round(total / 1_048_576, 2)


def prune(root: str = DEFAULT_ROOT, max_age_hours: float = 48.0,
          keep: Optional[str] = None) -> int:
    """Remove old run directories. Returns how many were deleted."""
    if not os.path.isdir(root):
        return 0
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for name in os.listdir(root):
        if name == keep:
            continue
        target = os.path.join(root, name)
        if not os.path.isdir(target):
            continue
        try:
            mtime = os.path.getmtime(target)
        except OSError:
            continue
        if mtime < cutoff:
            shutil.rmtree(target, ignore_errors=True)
            removed += 1
    return removed


def list_runs(root: str = DEFAULT_ROOT, limit: int = 20) -> List[Dict[str, Any]]:
    if not os.path.isdir(root):
        return []
    rows = []
    for name in sorted(os.listdir(root), reverse=True)[:limit]:
        session = WorkSession.load(name, root=root)
        if session is None:
            continue
        rows.append({
            "run_id": name,
            "protein": session.get("protein_name", "—"),
            "ligand": session.get("ligand_name", "—"),
            "updated": session.get("updated"),
        })
    return rows
