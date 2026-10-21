"""
rmsd.py — heavy-atom RMSD between the poses in a gnina output SDF.

All poses share the receptor's coordinate frame, so the RMSD is computed *in
place*: no superposition, which is what "how far did the ligand move" means in
docking. Hydrogens are ignored.

Two methods, chosen automatically:

  symmetry-corrected   RDKit CalcRMS: the atom mapping that minimises the RMSD
                       is used, so a flipped piperidine or a rotated phenyl is
                       not counted as a large move. Needs identical topology.
  atom order           Plain index-by-index RMSD. Used only if RDKit cannot
                       match a pair; it overstates the RMSD for symmetric groups.

gnina writes every pose of one run with the same atom order, so atom order is
a valid fallback *within a file*.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

SYMMETRY = "symmetry-corrected, in place (no superposition)"
ATOM_ORDER = "atom order, in place (no superposition)"


@dataclass
class RmsdResult:
    n_poses: int
    matrix: Optional[List[List[float]]] = None      # n x n, Å, symmetric
    method: str = ""
    n_heavy: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.matrix is not None

    def to_reference(self, ref: int) -> List[float]:
        """RMSD of every pose to pose `ref` (0-based)."""
        return list(self.matrix[ref]) if self.matrix else []

    def rows(self) -> List[dict]:
        """Matrix as table rows: {"Pose": 1, "P1": 0.0, "P2": 0.83, ...}."""
        out = []
        for i, line in enumerate(self.matrix or []):
            row = {"Pose": i + 1}
            for j, v in enumerate(line):
                row[f"P{j + 1}"] = round(v, 2)
            out.append(row)
        return out


def _read_blocks(path: str) -> List[str]:
    blocks, current = [], []
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.strip() == "$$$$":
                blocks.append("".join(current))
                current = []
            else:
                current.append(line)
    if "".join(current).strip():
        blocks.append("".join(current))
    return blocks


def _heavy_coords(block: str) -> Optional[List[Tuple[float, float, float]]]:
    """Heavy-atom coordinates, SDF order, from a V2000 block (no RDKit)."""
    lines = block.split("\n")
    try:
        n = int(lines[3][0:3])
    except (IndexError, ValueError):
        return None
    out = []
    for i in range(n):
        try:
            row = lines[4 + i]
            if row[31:34].strip().upper() in ("H", "D"):
                continue
            out.append((float(row[0:10]), float(row[10:20]), float(row[20:30])))
        except (IndexError, ValueError):
            return None
    return out


def _index_rmsd(a: Sequence[Tuple[float, float, float]],
                b: Sequence[Tuple[float, float, float]]) -> float:
    return math.sqrt(sum((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2
                         for p, q in zip(a, b)) / len(a))


def _heavy_mol(block: str):
    """Heavy-atom RDKit molecule for one pose, or None."""
    from rdkit import Chem

    mol = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)
    if mol is None:
        mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
        if mol is None:
            return None
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
    try:
        return Chem.RemoveHs(mol, sanitize=False)
    except Exception:                                         # noqa: BLE001
        return None


def pose_rmsd_matrix(sdf_path: str, max_matches: int = 5000) -> RmsdResult:
    """Pairwise heavy-atom RMSD of every pose in `sdf_path`."""
    if not os.path.isfile(sdf_path):
        return RmsdResult(0, warnings=[f"{sdf_path} not found."])
    blocks = _read_blocks(sdf_path)
    n = len(blocks)
    if n == 0:
        return RmsdResult(0, warnings=["No poses in the file."])

    coords = [_heavy_coords(b) for b in blocks]
    if any(c is None or not c for c in coords):
        return RmsdResult(n, warnings=["A pose could not be read (V2000 SDF "
                                       "expected); RMSD not computed."])
    sizes = {len(c) for c in coords}
    if len(sizes) != 1:
        return RmsdResult(n, warnings=[
            f"Poses have different heavy-atom counts {sorted(sizes)}; "
            "RMSD between different molecules is meaningless."])
    n_heavy = sizes.pop()
    matrix = [[0.0] * n for _ in range(n)]
    warnings: List[str] = []

    mols = None
    try:
        from rdkit.Chem import rdMolAlign
        mols = [_heavy_mol(b) for b in blocks]
        if any(m is None for m in mols):
            mols = None
    except ImportError:
        warnings.append("RDKit is not installed; used atom-order RMSD.")

    method = SYMMETRY if mols else ATOM_ORDER
    for i in range(n):
        for j in range(i + 1, n):
            value = None
            if mols:
                try:
                    value = float(rdMolAlign.CalcRMS(mols[j], mols[i],
                                                     maxMatches=max_matches))
                except Exception:                             # noqa: BLE001
                    value = None
                    if method == SYMMETRY:
                        method = ATOM_ORDER
                        warnings.append(
                            f"RDKit could not match poses {i + 1} and {j + 1}; "
                            "fell back to atom-order RMSD, which overstates "
                            "the value for symmetric groups.")
            if value is None:
                value = _index_rmsd(coords[i], coords[j])
            matrix[i][j] = matrix[j][i] = value
    return RmsdResult(n, matrix=matrix, method=method, n_heavy=n_heavy,
                      warnings=warnings)


def write_matrix_csv(result: RmsdResult, path: str) -> str:
    rows = result.rows()
    if not rows:
        raise ValueError("No RMSD matrix to write.")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow([f"# heavy-atom RMSD (A), {result.method}"])
        writer.writerow(list(rows[0].keys()))
        for row in rows:
            writer.writerow(list(row.values()))
    return path
