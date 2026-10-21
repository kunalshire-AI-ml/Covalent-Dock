"""
complex_builder.py — receptor + gnina covalent pose(s) → one bonded complex PDB.

gnina writes the reacting residue (e.g. Ser195 N, CA, C, O, CB, OG) into every
covalent pose, on top of the receptor's own copy of it. Taken as ligand atoms,
those copies give a ligand that overlaps the protein, and "the ligand atom
nearest the receptor atom" is then the copy of the receptor atom itself, at
0.00 Å — a LINK record that bonds Ser195 OG to a duplicate of Ser195 OG.

So the bond is stated, not inferred:

  receptor atom   the covalent_rec_atom selector, e.g. H:195:OG
  ligand atom     the pose atom bonded to the pose's copy of that atom
  bond            receptor OG – ligand C8, written as LINK + CONECT

Copies of the reacting residue are recognised first by position (within 0.2 Å
of the receptor atom) and then along the pose's bonds by element and
connectivity, so a side chain gnina rotated is still recognised. The copies are
removed from the ligand and the residue stays whole in the protein. When gnina
moved the side chain and a single pose is written, the residue is written at
that pose's coordinates, so the bond length in the file is the docked one.

Poses that carry no residue atoms (ligand only) fall back to the warhead SMARTS
and geometry: among the atoms the pattern matches, the one nearest the receptor
atom; failing that, the nearest heavy atom.

Pure Python; RDKit is imported only to match the warhead SMARTS.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

XYZ = Tuple[float, float, float]

# Longest bond worth calling covalent (S–C is 1.81 Å, plus slack for a pose
# that was optimised loosely). Past this, a "bond" is an artefact.
MAX_COVALENT_BOND = 2.6
# Shorter than this is not a bond but two copies of one atom.
MIN_COVALENT_BOND = 0.5
# A pose atom closer than this to a receptor atom is a copy of it.
OVERLAP_TOL = 0.2
# Heavy-atom bond cut-off when reading the receptor residue's own bonds.
RESIDUE_BOND_MAX = 1.95
BACKBONE = {"N", "CA", "C", "O", "OXT"}
_HYDROGENS = ("H", "D")

# Same order as visualization/viewer.py POSE_PALETTE (green, cyan, magenta,
# orange, purple, blue), repeated for pose 7 onwards.
_PML_COLORS = ("green", "cyan", "magenta", "orange", "purple", "blue")


# ─────────────────────────────────────────────────────────────────────────────
#  SDF helpers  (pure Python — no cheminformatics library needed)
# ─────────────────────────────────────────────────────────────────────────────

def extract_sdf_block(sdf_path: str, index: int = 0) -> str:
    """Return one molecule block from an SDF (0-based pose index)."""
    blocks, current = [], []
    with open(sdf_path, errors="replace") as fh:
        for line in fh:
            if line.strip() == "$$$$":
                blocks.append("".join(current))
                current = []
                if len(blocks) > index:
                    break
            else:
                current.append(line)
    if not blocks and current:
        blocks.append("".join(current))
    if index >= len(blocks):
        raise ValueError(f"Pose {index + 1} not found in {sdf_path}")
    block = blocks[index]
    if not block.strip():
        raise ValueError(f"Pose {index + 1} in {sdf_path} is empty")
    return block


def parse_sdf_properties(sdf_path: str, index: int = 0) -> Dict[str, str]:
    """gnina writes scores as SDF tags; pull them off one pose."""
    props: Dict[str, str] = {}
    key = None
    pose = 0
    with open(sdf_path, errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if s == "$$$$":
                pose += 1
                if pose > index:
                    break
                props, key = {}, None
                continue
            if pose != index:
                continue
            if s.startswith(">") and "<" in s:
                key = s.split("<")[1].split(">")[0]
            elif key and s:
                props[key] = s
                key = None
    return props


def count_poses(sdf_path: str) -> int:
    if not os.path.isfile(sdf_path):
        return 0
    with open(sdf_path, errors="replace") as fh:
        return sum(1 for line in fh if line.strip() == "$$$$")


def _read_sdf_blocks(path: str) -> List[str]:
    blocks: List[str] = []
    current: List[str] = []
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


def _block_scores(block: str) -> Dict[str, str]:
    """Score tags of one pose block."""
    props: Dict[str, str] = {}
    key = None
    for line in block.splitlines():
        s = line.strip()
        if s.startswith(">") and "<" in s:
            key = s.split("<")[1].split(">")[0]
        elif key and s:
            props[key] = s
            key = None
    return props


def _parse_sdf_molecule(block: str):
    """([(x, y, z, element)], [(i, j, order)]) from a V2000 block, 0-based."""
    lines = block.split("\n")
    if len(lines) < 5:
        return [], []
    try:
        n_atoms, n_bonds = int(lines[3][0:3]), int(lines[3][3:6])
    except ValueError:
        return [], []
    atoms = []
    for i in range(n_atoms):
        row = lines[4 + i] if 4 + i < len(lines) else ""
        try:
            atoms.append((float(row[0:10]), float(row[10:20]), float(row[20:30]),
                          row[31:34].strip() or "C"))
        except ValueError:
            return [], []          # a partial read would renumber the atoms
    bonds = []
    for k in range(n_bonds):
        idx = 4 + n_atoms + k
        if idx >= len(lines):
            break
        try:
            a, b, t = (int(lines[idx][0:3]) - 1, int(lines[idx][3:6]) - 1,
                       int(lines[idx][6:9]))
        except ValueError:
            continue
        if 0 <= a < n_atoms and 0 <= b < n_atoms:
            bonds.append((a, b, t))
    return atoms, bonds


def _smarts_candidates(block: str, smarts: Optional[str]) -> List[int]:
    """Atoms of this pose matching the warhead SMARTS (first atom of each match)."""
    if not smarts:
        return []
    try:
        from rdkit import Chem
    except ImportError:
        return []
    mol = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)
    if mol is None:
        mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
        if mol is None:
            return []
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return []
    return sorted({m[0] for m in mol.GetSubstructMatches(patt, uniquify=True)})


# ─────────────────────────────────────────────────────────────────────────────
#  PDB record writing
# ─────────────────────────────────────────────────────────────────────────────

def link_record(rec_atom_name: str, rec_resname: str, rec_chain: str,
                rec_resseq: int, lig_atom_name: str, lig_resname: str,
                lig_chain: str, lig_resseq: int, distance: float) -> str:
    """A PDB LINK record for the covalent bond (fixed-column format)."""
    buf = [" "] * 80

    def put(col: int, text: str) -> None:          # col is 1-based
        buf[col - 1:col - 1 + len(text)] = list(text)

    def name_field(name: str) -> str:
        return f" {name:<3}" if len(name) < 4 else name[:4]

    put(1, "LINK")
    put(13, name_field(rec_atom_name)); put(18, f"{rec_resname:>3}")
    put(22, rec_chain); put(23, f"{rec_resseq:>4}")
    put(43, name_field(lig_atom_name)); put(48, f"{lig_resname:>3}")
    put(52, lig_chain); put(53, f"{lig_resseq:>4}")
    put(60, "  1555"); put(67, "  1555"); put(74, f"{distance:>5.2f}")
    return "".join(buf).rstrip()


def conect_records(serial_a: int, serial_b: int) -> List[str]:
    return [f"CONECT{serial_a:>5}{serial_b:>5}",
            f"CONECT{serial_b:>5}{serial_a:>5}"]


def _conect_lines(pairs: Iterable[Tuple[int, int, int]]) -> List[str]:
    """CONECT lines; a double/triple bond lists its partner 2/3 times."""
    nbrs: Dict[int, List[int]] = {}
    for a, b, order in pairs:
        reps = order if order in (2, 3) else 1
        nbrs.setdefault(a, []).extend([b] * reps)
        nbrs.setdefault(b, []).extend([a] * reps)
    out: List[str] = []
    for a in sorted(nbrs):
        targets = nbrs[a]
        for i in range(0, len(targets), 4):
            out.append(f"CONECT{a:>5}" + "".join(f"{t:>5}" for t in targets[i:i + 4]))
    return out


def _with_xyz(line: str, xyz: Sequence[float]) -> str:
    padded = line.ljust(54)
    return padded[:30] + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}" + line[54:]


# ─────────────────────────────────────────────────────────────────────────────
#  Receptor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _RecAtom:
    serial: int
    name: str
    raw_name: str          # columns 13-16, kept for the LINK record
    resname: str
    chain: str
    resseq: str            # residue number + insertion code, as text
    element: str
    xyz: XYZ
    line: int              # index into the receptor coordinate lines


def _read_receptor(path: str):
    """(coordinate lines, header lines, CONECT lines, parsed atoms)."""
    coord: List[str] = []
    header: List[str] = []
    conect: List[str] = []
    atoms: List[_RecAtom] = []
    with open(path, errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n").rstrip()
            if line.startswith(("ATOM  ", "HETATM")):
                p = line.ljust(80)
                try:
                    serial = int(p[6:11])
                    xyz = (float(p[30:38]), float(p[38:46]), float(p[46:54]))
                except ValueError:
                    coord.append(line)
                    continue
                name = p[12:16].strip()
                atoms.append(_RecAtom(
                    serial=serial, name=name, raw_name=p[12:16],
                    resname=p[17:20].strip(), chain=p[21].strip() or "A",
                    resseq=p[22:26].strip() + p[26].strip(),
                    element=(p[76:78].strip() or name[:1]).upper(),
                    xyz=xyz, line=len(coord)))
                coord.append(line)
            elif line.startswith("TER"):
                coord.append(line)
            elif line.startswith("CONECT"):
                conect.append(line)
            elif line.startswith(("REMARK", "HEADER", "TITLE", "HELIX", "SHEET",
                                  "SSBOND", "LINK", "CRYST")):
                header.append(line)
    return coord, header, conect, atoms


class _Grid:
    """Cell list over the receptor, for the 0.2 Å copy test."""

    def __init__(self, atoms: Sequence[_RecAtom], cell: float = 2.0):
        self.cell = cell
        self.bins: Dict[Tuple[int, int, int], List[_RecAtom]] = {}
        for a in atoms:
            self.bins.setdefault(self._key(a.xyz), []).append(a)

    def _key(self, p: Sequence[float]) -> Tuple[int, int, int]:
        c = self.cell
        return (math.floor(p[0] / c), math.floor(p[1] / c), math.floor(p[2] / c))

    def nearest_within(self, p: Sequence[float], tol: float) -> Optional[_RecAtom]:
        i0, j0, k0 = self._key(p)
        best, best_d = None, tol
        for i in (i0 - 1, i0, i0 + 1):
            for j in (j0 - 1, j0, j0 + 1):
                for k in (k0 - 1, k0, k0 + 1):
                    for a in self.bins.get((i, j, k), ()):
                        d = math.dist(a.xyz, p)
                        if d <= best_d:
                            best, best_d = a, d
        return best


def _anchor(atoms: Sequence[_RecAtom], selector: str
            ) -> Tuple[Optional[_RecAtom], List[_RecAtom]]:
    """The selector's atom and every atom of its residue."""
    parts = [s.strip() for s in selector.split(":")]
    if len(parts) != 3:
        return None, []
    chain, resnum, name = parts
    residue = [a for a in atoms if a.chain == (chain or "A") and a.resseq == resnum]
    return next((a for a in residue if a.name == name), None), residue


def _residue_bonds(residue: Sequence[_RecAtom]) -> Dict[int, List[_RecAtom]]:
    heavy = [a for a in residue if a.element not in _HYDROGENS]
    nb: Dict[int, List[_RecAtom]] = {a.serial: [] for a in heavy}
    for i, a in enumerate(heavy):
        for b in heavy[i + 1:]:
            if math.dist(a.xyz, b.xyz) < RESIDUE_BOND_MAX:
                nb[a.serial].append(b)
                nb[b.serial].append(a)
    return nb


def _parent(h: _RecAtom, residue: Sequence[_RecAtom]) -> Optional[_RecAtom]:
    heavy = [a for a in residue if a.element not in _HYDROGENS]
    return min(heavy, key=lambda a: math.dist(a.xyz, h.xyz)) if heavy else None


# ─────────────────────────────────────────────────────────────────────────────
#  The reacting residue inside a gnina pose
# ─────────────────────────────────────────────────────────────────────────────

def _is_h(element: str) -> bool:
    return element.upper() in _HYDROGENS


def _anchor_copies(atoms, nbrs, residue: Sequence[_RecAtom], grid: _Grid
                   ) -> Tuple[Dict[int, _RecAtom], int]:
    """
    Pose atom index → the residue atom it duplicates, plus how many pose atoms
    sit on receptor atoms outside the residue.

    Seeded by position, then grown along the pose's bonds: a neighbour of a
    mapped atom is the residue atom bonded to the same partner with the same
    element. That second step is topology, not position, so a side chain gnina
    rotated is still recognised.
    """
    in_residue = {a.serial for a in residue}
    mapped: Dict[int, _RecAtom] = {}
    elsewhere = 0
    for i, (x, y, z, el) in enumerate(atoms):
        if _is_h(el):
            continue
        hit = grid.nearest_within((x, y, z), OVERLAP_TOL)
        if hit is None:
            continue
        if hit.serial in in_residue:
            mapped[i] = hit
        else:
            elsewhere += 1

    crystal = _residue_bonds(residue)
    used = {a.serial for a in mapped.values()}
    queue = list(mapped)
    while queue:
        i = queue.pop()
        for j in nbrs[i]:
            if j in mapped or _is_h(atoms[j][3]):
                continue
            el = atoms[j][3].upper()
            cands = [a for a in crystal.get(mapped[i].serial, [])
                     if a.serial not in used and a.element == el]
            if len(cands) == 1:
                mapped[j] = cands[0]
                used.add(cands[0].serial)
                queue.append(j)
    return mapped, elsewhere


def _bonded_atom(atoms, nbrs, copies: Dict[int, _RecAtom], anchor: _RecAtom,
                 pool: Sequence[int], smarts_hits: Sequence[int],
                 rec_xyz: XYZ) -> Tuple[Optional[int], str]:
    """The ligand atom that carries the covalent bond, and how it was found."""
    anchor_copy = next((i for i, a in copies.items() if a.serial == anchor.serial), None)
    if anchor_copy is not None:
        partners = [j for j in nbrs[anchor_copy]
                    if j not in copies and not _is_h(atoms[j][3])]
        if len(partners) == 1:
            return partners[0], f"bonded to the pose's copy of {anchor.name}"
        if partners:
            preferred = [j for j in partners if j in smarts_hits] or partners
            ref = atoms[anchor_copy][:3]
            k = min(preferred, key=lambda j: math.dist(atoms[j][:3], ref))
            return k, (f"one of {len(partners)} atoms bonded to the pose's copy of "
                       f"{anchor.name}")
    if not pool:
        return None, ""
    hits = [i for i in smarts_hits if i in set(pool)]
    k = min(hits or pool, key=lambda i: math.dist(atoms[i][:3], rec_xyz))
    return k, ("warhead pattern, nearest match" if hits else "nearest heavy atom")


# ─────────────────────────────────────────────────────────────────────────────
#  Results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PoseBond:
    """The covalent bond of one pose; `pose` is also the ligand residue number."""
    pose: int
    rec_chain: str
    rec_resseq: int
    rec_atom: str
    lig_atom: str
    distance: float
    rec_xyz: Tuple[float, float, float]
    lig_xyz: Tuple[float, float, float]
    how: str = ""                       # how the ligand atom was identified
    sdf_index: Optional[int] = None     # 0-based atom index in the pose SDF

    def as_dict(self) -> Dict[str, object]:
        """JSON-safe, for the session manifest and the 3-D viewer."""
        return {"pose": self.pose, "rec_atom": self.rec_atom,
                "lig_atom": self.lig_atom, "distance": self.distance,
                "prot": list(self.rec_xyz), "lig": list(self.lig_xyz),
                "how": self.how, "sdf_index": self.sdf_index}


@dataclass
class MergeResult:
    path: str
    n_poses: int
    bonds: List[PoseBond] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    anchor_copies: List[str] = field(default_factory=list)   # e.g. ["N", "CA", ...]
    moved_poses: List[int] = field(default_factory=list)     # side chain moved by gnina
    n_receptor_atoms: int = 0
    n_ligand_atoms: int = 0                                  # first written pose

    @property
    def bond(self) -> Optional[PoseBond]:
        return self.bonds[0] if self.bonds else None


# ─────────────────────────────────────────────────────────────────────────────
#  Merge: receptor PDB + gnina poses SDF → one bonded complex PDB
# ─────────────────────────────────────────────────────────────────────────────

def merge_complex(
    receptor_pdb: str,
    poses_sdf: str,
    out_path: str,
    pose_index: Optional[int] = None,       # None = every pose
    resname: str = "LIG",
    chain: str = "Z",
    covalent_rec_atom: Optional[str] = None,   # "A:12:SG"
    lig_smarts: Optional[str] = None,          # warhead pattern given to gnina
    remarks: Sequence[str] = (),
) -> MergeResult:
    """
    Receptor + docked pose(s) in one PDB. Pose N is residue N of the ligand
    chain, so every pose is visible at once. With `covalent_rec_atom`, each
    pose gets a LINK record and a CONECT pair to the receptor atom, and gnina's
    copy of the reacting residue is removed from the ligand.

    `poses_sdf` must be gnina's -o file. The --out_flex file holds receptor
    residues, not the ligand, and must never be merged as the ligand.
    """
    coord, header, rec_conect, rec_atoms = _read_receptor(receptor_pdb)
    if not rec_atoms:
        raise ValueError(f"No receptor atoms read from {receptor_pdb}")
    max_serial = max(a.serial for a in rec_atoms)

    blocks = _read_sdf_blocks(poses_sdf)
    if not blocks:
        raise ValueError(f"No poses found in {poses_sdf}")
    if pose_index is None:
        indices = list(range(len(blocks)))
    elif 0 <= pose_index < len(blocks):
        indices = [pose_index]
    else:
        raise ValueError(f"Pose {pose_index + 1} not found "
                         f"(file has {len(blocks)} poses)")
    single = len(indices) == 1

    warnings: List[str] = []
    anchor: Optional[_RecAtom] = None
    residue: List[_RecAtom] = []
    if covalent_rec_atom:
        anchor, residue = _anchor(rec_atoms, covalent_rec_atom)
        if anchor is None:
            warnings.append(f"Receptor atom {covalent_rec_atom} was not found in "
                            f"{os.path.basename(receptor_pdb)}; no covalent bond "
                            "was written.")
    grid = _Grid(rec_atoms)

    serial = max_serial
    lig_lines: List[str] = []
    link_lines: List[str] = []
    bond_pairs: List[Tuple[int, int, int]] = []
    bonds: List[PoseBond] = []
    copy_names: Set[str] = set()
    moved_poses: List[int] = []
    substitute: Dict[int, XYZ] = {}       # receptor serial → coordinates (single pose)
    n_written = 0
    n_lig_first = 0
    warned_elsewhere = False
    score_remarks: List[str] = []

    for i in indices:
        pose_no = i + 1
        atoms, sdf_bonds = _parse_sdf_molecule(blocks[i])
        if not atoms:
            warnings.append(f"Pose {pose_no} could not be read (V2000 SDF "
                            "expected); skipped.")
            continue
        nbrs: Dict[int, List[int]] = {k: [] for k in range(len(atoms))}
        for a, b, _t in sdf_bonds:
            nbrs[a].append(b)
            nbrs[b].append(a)

        # ── gnina's copy of the reacting residue ─────────────────────────────
        copies: Dict[int, _RecAtom] = {}
        if anchor is not None:
            copies, elsewhere = _anchor_copies(atoms, nbrs, residue, grid)
            if elsewhere and not warned_elsewhere:
                warned_elsewhere = True
                warnings.append(
                    f"{elsewhere} pose atom(s) sit on receptor atoms outside "
                    f"{anchor.resname}{anchor.resseq}. This SDF may hold receptor "
                    "atoms (gnina --out_flex) rather than ligand poses; merge the "
                    "-o output instead.")
        drop = set(copies)
        for k, (_x, _y, _z, el) in enumerate(atoms):
            if _is_h(el) and nbrs[k] and all(n in drop for n in nbrs[k]):
                drop.add(k)
        keep = [k for k in range(len(atoms)) if k not in drop]
        if not keep:
            warnings.append(f"Pose {pose_no}: no ligand atoms left after removing "
                            "the residue copy; skipped.")
            continue
        n_written += 1
        copy_names.update(a.name for a in copies.values())

        # ── side chain moved by gnina? ───────────────────────────────────────
        moved: Dict[int, XYZ] = {}
        for k, a in copies.items():
            shift = math.dist(atoms[k][:3], a.xyz)
            if shift <= OVERLAP_TOL:
                continue
            if a.name in BACKBONE:
                warnings.append(f"Pose {pose_no}: gnina moved backbone atom "
                                f"{a.name} by {shift:.2f} Å; the receptor backbone "
                                "is kept.")
            else:
                moved[a.serial] = tuple(atoms[k][:3])
        if moved:
            moved_poses.append(pose_no)
            if single:
                substitute.update(moved)

        # ── ligand atoms ─────────────────────────────────────────────────────
        serial_of: Dict[int, int] = {}
        names: Dict[int, str] = {}
        per_element: Dict[str, int] = {}
        for k in keep:
            x, y, z, element = atoms[k]
            serial += 1
            if serial > 99999:
                raise ValueError("More than 99,999 atoms; PDB serial numbers overflow.")
            el = element.upper()
            per_element[el] = per_element.get(el, 0) + 1
            name = f"{el}{per_element[el]}"[:4]
            serial_of[k] = serial
            names[k] = name
            name_field = f" {name:<3}" if len(name) < 4 else name
            lig_lines.append(
                "HETATM" + f"{serial:>5}" + " " + name_field + " "
                + f"{resname:<3}" + " " + chain + f"{pose_no:>4}" + "    "
                + f"{x:>8.3f}{y:>8.3f}{z:>8.3f}" + "  1.00  0.00"
                + " " * 10 + f"{el:>2}")
        if n_written == 1:
            n_lig_first = len(keep)
        for a, b, order in sdf_bonds:
            if a in serial_of and b in serial_of:
                bond_pairs.append((serial_of[a], serial_of[b], order))

        scores = _block_scores(blocks[i])
        bits = [f"{key} {scores[key]}" for key in
                ("minimizedAffinity", "CNNscore", "CNNaffinity") if key in scores]
        if bits:
            score_remarks.append(f"REMARK   Pose {pose_no}: " + "  ".join(bits))

        # ── the covalent bond ────────────────────────────────────────────────
        if anchor is None:
            continue
        rec_xyz = substitute.get(anchor.serial, anchor.xyz) if single else anchor.xyz
        pool = [k for k in keep if not _is_h(atoms[k][3])
                and grid.nearest_within(atoms[k][:3], OVERLAP_TOL) is None]
        k, how = _bonded_atom(atoms, nbrs, copies, anchor, pool,
                              _smarts_candidates(blocks[i], lig_smarts), rec_xyz)
        if k is None or k not in serial_of:
            warnings.append(f"Pose {pose_no}: no ligand atom could be identified "
                            "as the bonded one; no LINK written.")
            continue
        lig_xyz = tuple(atoms[k][:3])
        distance = math.dist(lig_xyz, rec_xyz)
        if distance < MIN_COVALENT_BOND:
            warnings.append(
                f"Pose {pose_no}: the chosen ligand atom is {distance:.2f} Å from "
                f"{covalent_rec_atom} — a duplicate of it, not a bonded atom. "
                "No LINK written for this pose.")
            continue
        try:
            rec_resseq = int(anchor.resseq)
        except ValueError:
            rec_resseq = int("".join(ch for ch in anchor.resseq if ch.isdigit()) or 0)
        link_lines.append(link_record(
            anchor.raw_name, anchor.resname, anchor.chain, rec_resseq,
            names[k], resname, chain, pose_no, distance))
        bond_pairs.append((anchor.serial, serial_of[k], 1))
        bonds.append(PoseBond(pose=pose_no, rec_chain=anchor.chain,
                              rec_resseq=rec_resseq, rec_atom=anchor.name,
                              lig_atom=names[k], distance=distance,
                              rec_xyz=tuple(rec_xyz), lig_xyz=lig_xyz,
                              how=how, sdf_index=k))
        if distance > MAX_COVALENT_BOND:
            warnings.append(
                f"Pose {pose_no}: {covalent_rec_atom} to {names[k]} is "
                f"{distance:.2f} Å, long for a covalent bond. The bond is "
                "still written; this pose may not be the covalent one.")

    if not lig_lines:
        raise ValueError("No ligand atoms could be read from the SDF.")

    if moved_poses and not single:
        warnings.append(
            f"gnina moved the {anchor.resname}{anchor.resseq} side chain in pose(s) "
            f"{', '.join(map(str, moved_poses))}. This all-poses file shows the "
            "receptor's rotamer, so those bonds are drawn long; write one pose "
            "(pose_index=…) to see the docked rotamer.")

    # ── receptor: the leaving proton and hydrogens left behind ───────────────
    drop_rec: Set[int] = set()
    if anchor is not None:
        for h in residue:
            if h.element not in _HYDROGENS:
                continue
            parent = _parent(h, residue)
            if parent is not None and (parent.serial == anchor.serial
                                       or parent.serial in substitute):
                drop_rec.add(h.serial)
    rec_lines = list(coord)
    by_serial = {a.serial: a for a in rec_atoms}
    for s, xyz in substitute.items():
        rec_lines[by_serial[s].line] = _with_xyz(rec_lines[by_serial[s].line], xyz)
    if drop_rec:
        rec_lines = [l for l in rec_lines
                     if not (l.startswith(("ATOM", "HETATM"))
                             and l[6:11].strip().isdigit()
                             and int(l[6:11]) in drop_rec)]
    kept_conect: List[str] = []
    for line in rec_conect:
        body = line.rstrip()
        try:
            nums = [int(body[c:c + 5]) for c in range(6, len(body), 5)]
        except ValueError:
            continue
        nums = [n for n in nums if n not in drop_rec]
        if len(nums) >= 2 and nums[0] not in drop_rec:
            kept_conect.append("CONECT" + "".join(f"{n:>5}" for n in nums))

    # ── write ────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(f"REMARK   Complex: {os.path.basename(receptor_pdb)}"
                 f" + {os.path.basename(poses_sdf)}\n")
        fh.write("REMARK   Pose " + (str(pose_index + 1) if pose_index is not None
                                     else f"1-{n_written} (one residue per pose)") + "\n")
        if anchor is not None and copy_names:
            fh.write(f"REMARK   {anchor.resname}{anchor.resseq} is complete in the "
                     "protein; gnina's copy of it was removed from the ligand\n")
        for b in bonds:
            fh.write(f"REMARK   Covalent bond, pose {b.pose}: "
                     f"{b.rec_chain}:{b.rec_resseq}:{b.rec_atom} - "
                     f"{resname} {b.lig_atom} ({b.distance:.2f} A)\n")
        if substitute:
            fh.write(f"REMARK   {anchor.resname}{anchor.resseq} side chain at the "
                     "docked rotamer (moved by gnina)\n")
        for line in score_remarks:
            fh.write(line + "\n")
        for line in remarks:
            fh.write(f"REMARK   {line}\n")
        for line in header:
            fh.write(line + "\n")
        for line in link_lines:
            fh.write(line + "\n")
        for line in rec_lines:
            fh.write(line + "\n")
        if not (rec_lines and rec_lines[-1].startswith("TER")):
            fh.write("TER\n")
        for line in lig_lines:
            fh.write(line + "\n")
        fh.write("TER\n")
        for line in kept_conect + _conect_lines(bond_pairs):
            fh.write(line + "\n")
        fh.write("END\n")

    return MergeResult(
        path=out_path, n_poses=n_written, bonds=bonds, warnings=warnings,
        anchor_copies=sorted(copy_names), moved_poses=moved_poses,
        n_receptor_atoms=sum(1 for l in rec_lines if l.startswith(("ATOM", "HETATM"))),
        n_ligand_atoms=n_lig_first,
    )


def merge_complex_models(
    receptor_pdb: str,
    poses_sdf: str,
    out_path: str,
    resname: str = "LIG",
    chain: str = "Z",
    covalent_rec_atom: Optional[str] = None,
    lig_smarts: Optional[str] = None,
) -> MergeResult:
    """
    Every pose in one file, as separate MODEL frames rather than one frame.

    merge_complex(pose_index=None) already writes all poses together, but it
    puts each one in its own *residue* of the ligand chain — so they all show
    at once, overlaid. That answers "how do the poses compare"; it does not
    give you a complex per pose to step through.

    Here each pose becomes a MODEL: receptor plus that one pose, with its own
    LINK and CONECT. Twenty poses give twenty frames, which PyMOL and Chimera
    read as states and 3Dmol as frames.

    The per-pose content comes from merge_complex itself, so the chemistry —
    how the bonded atom is identified, how gnina's copy of the anchor residue
    is handled — is identical to the single-pose file.
    """
    n_poses = count_poses(poses_sdf)
    if n_poses == 0:
        raise ValueError(f"no poses in {poses_sdf}")

    header: List[str] = []
    frames: List[List[str]] = []
    bonds: List[PoseBond] = []
    warnings: List[str] = []
    moved: List[int] = []
    anchor_copies: List[str] = []
    n_rec = n_lig = 0

    tmp_dir = tempfile.mkdtemp(prefix="covdock_models_")
    try:
        for index in range(n_poses):
            tmp = os.path.join(tmp_dir, f"pose{index + 1}.pdb")
            try:
                one = merge_complex(
                    receptor_pdb, poses_sdf, tmp, pose_index=index,
                    resname=resname, chain=chain,
                    covalent_rec_atom=covalent_rec_atom, lig_smarts=lig_smarts)
            except Exception as exc:                          # noqa: BLE001
                warnings.append(f"pose {index + 1} skipped: {exc}")
                continue

            bonds.extend(one.bonds)
            moved.extend(one.moved_poses)
            warnings.extend(f"pose {index + 1}: {w}" for w in one.warnings)
            if not anchor_copies:
                anchor_copies = list(one.anchor_copies)
            n_rec, n_lig = one.n_receptor_atoms, one.n_ligand_atoms

            with open(tmp, errors="replace") as fh:
                lines = [l.rstrip("\n") for l in fh]
            # Only coordinates and the per-pose connectivity belong inside a
            # MODEL. HELIX, SHEET, SSBOND and the rest are header records, and
            # a viewer that meets them inside a model block ignores them — the
            # cartoon then has no secondary structure and renders as a tube.
            # They describe the receptor, which is the same in every frame, so
            # they are written once above the first MODEL.
            body = [l for l in lines
                    if l.startswith(("ATOM", "HETATM", "TER", "LINK", "CONECT"))]
            if not header:
                header = [l for l in lines
                          if not l.startswith(("ATOM", "HETATM", "TER", "LINK",
                                               "CONECT", "END"))]
            frames.append(body)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not frames:
        raise ValueError("no pose could be merged into a complex")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(f"REMARK   {len(frames)} model(s), one per pose\n")
        for number, body in enumerate(frames, 1):
            fh.write(f"MODEL     {number:>4}\n")
            for line in body:
                fh.write(line + "\n")
            fh.write("ENDMDL\n")
        fh.write("END\n")

    return MergeResult(path=out_path, n_poses=len(frames), bonds=bonds,
                       warnings=warnings, anchor_copies=anchor_copies,
                       moved_poses=moved, n_receptor_atoms=n_rec,
                       n_ligand_atoms=n_lig)


def simple_merge_complex(
    receptor_pdb: str,
    poses_sdf: str,
    out_path: str,
    pose_index: Optional[int] = None,
    resname: str = "LIG",
    chain: str = "Z",
    covalent_rec_atom: Optional[str] = None,
    lig_smarts: Optional[str] = None,
) -> str:
    """Compatibility wrapper around merge_complex; returns the output path."""
    return merge_complex(receptor_pdb, poses_sdf, out_path, pose_index=pose_index,
                         resname=resname, chain=chain,
                         covalent_rec_atom=covalent_rec_atom,
                         lig_smarts=lig_smarts).path


# ─────────────────────────────────────────────────────────────────────────────
#  Older entry points, kept so existing callers keep working
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComplexResult:
    path: str
    pose_index: int
    scores: Dict[str, str]
    bond_distance: Optional[float]
    receptor_atom: Optional[str]
    ligand_atom: Optional[str]
    n_receptor_atoms: int
    n_ligand_atoms: int
    warnings: List[str] = field(default_factory=list)
    receptor_xyz: Optional[Tuple[float, float, float]] = None
    ligand_xyz: Optional[Tuple[float, float, float]] = None


def build_complex(
    receptor_pdb: str,
    poses_sdf: str,
    out_path: str,
    pose_index: int = 0,
    covalent_rec_atom: Optional[str] = None,
    rec_resname: str = "CYS",
    lig_resname: str = "LIG",
    lig_chain: str = "Z",
    lig_resseq: int = 1,
    ligand_name: str = "ligand",
    lig_atom_index: Optional[int] = None,
    remarks: Sequence[str] = (),
    lig_smarts: Optional[str] = None,
) -> ComplexResult:
    """
    Deprecated: one pose through merge_complex, in the old result shape.

    `rec_resname`, `lig_resseq`, `ligand_name` and `lig_atom_index` are
    accepted and ignored. The residue name is read from the receptor, the
    ligand residue number is the pose number, and the bonded atom is read
    from the pose: an index from the input ligand file does not survive
    gnina's reordering.
    """
    merged = merge_complex(receptor_pdb, poses_sdf, out_path, pose_index=pose_index,
                           resname=lig_resname, chain=lig_chain,
                           covalent_rec_atom=covalent_rec_atom,
                           lig_smarts=lig_smarts, remarks=remarks)
    b = merged.bond
    return ComplexResult(
        path=merged.path, pose_index=pose_index,
        scores=parse_sdf_properties(poses_sdf, pose_index),
        bond_distance=b.distance if b else None,
        receptor_atom=covalent_rec_atom,
        ligand_atom=b.lig_atom if b else None,
        n_receptor_atoms=merged.n_receptor_atoms,
        n_ligand_atoms=merged.n_ligand_atoms,
        warnings=merged.warnings,
        receptor_xyz=b.rec_xyz if b else None,
        ligand_xyz=b.lig_xyz if b else None,
    )


def build_all_poses(receptor_pdb: str, poses_sdf: str, out_dir: str,
                    prefix: str = "complex", **kwargs) -> List[ComplexResult]:
    """Deprecated: one complex PDB per pose, through merge_complex."""
    os.makedirs(out_dir, exist_ok=True)
    return [build_complex(receptor_pdb, poses_sdf,
                          os.path.join(out_dir, f"{prefix}_pose{i + 1}.pdb"),
                          pose_index=i, **kwargs)
            for i in range(count_poses(poses_sdf))]


# ─────────────────────────────────────────────────────────────────────────────
#  Viewing helpers
# ─────────────────────────────────────────────────────────────────────────────

def write_pymol_script(pdb_path: str, out_path: str, bonds: Sequence[PoseBond],
                       n_poses: Optional[int] = None, resname: str = "LIG",
                       chain: str = "Z",
                       pose_numbers: Optional[Sequence[int]] = None) -> str:
    """
    A .pml that shows every pose as sticks in its own colour and draws the
    covalent bond explicitly (`bond` + a solid orange distance), so it is
    visible even in a viewer that ignores LINK/CONECT. Keep it next to the PDB.
    """
    nums = list(pose_numbers) if pose_numbers else list(range(1, (n_poses or len(bonds) or 1) + 1))
    lig = f"cplx and chain {chain} and resn {resname}"
    cmds = [
        f"# keep this file next to {os.path.basename(pdb_path)}",
        f"load {os.path.basename(pdb_path)}, cplx",
        # A receptor re-saved from a viewer usually loses its HELIX and SHEET
        # records, and the cartoon then renders as a featureless tube. dss
        # assigns secondary structure from the geometry, and costs nothing
        # when the records were already there.
        "dss cplx",
        "hide everything",
        "show cartoon, cplx and polymer",
        "color grey80, cplx and polymer",
        f"select lig, {lig}",
        "show sticks, lig",
        "select site, byres (cplx and polymer within 5 of lig)",
        "show lines, site",
    ]
    for i in nums:
        colour = _PML_COLORS[(i - 1) % len(_PML_COLORS)]
        cmds.append(f"color {colour}, lig and resi {i} and elem C")
    cmds.append("util.cnc lig")
    for b in bonds:
        rec = (f"(cplx and chain {b.rec_chain} and resi {b.rec_resseq} "
               f"and name {b.rec_atom})")
        atom = f"({lig} and resi {b.pose} and name {b.lig_atom})"
        cmds += [f"bond {rec}, {atom}", f"distance cov_{b.pose}, {rec}, {atom}"]
    if bonds:
        b = bonds[0]
        cmds += [f"show sticks, cplx and chain {b.rec_chain} and resi {b.rec_resseq}",
                 "color orange, cov_*", "set dash_gap, 0", "set dash_width, 3"]
    cmds.append("zoom lig, 8")
    with open(out_path, "w") as fh:
        fh.write("\n".join(cmds) + "\n")
    return out_path


def find_covalent_bond(
    complex_pdb: str,
    selector: str,
    lig_resname: str = "LIG",
    lig_resseq: Optional[int] = None,
) -> Optional[Dict]:
    """
    Nearest-heavy-atom bond geometry from a merged complex PDB. Prefer the
    `MergeResult.bonds` from merge_complex, which knows the reacted atom; this
    is for complexes built elsewhere. `lig_resseq` limits it to one pose.
    """
    if not selector:
        return None
    parts = selector.split(":")
    if len(parts) != 3:
        return None
    sel_chain, sel_resnum, sel_atom = parts

    rec_xyz = None
    lig_heavy_atoms = []
    with open(complex_pdb, errors="replace") as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            padded = line.ljust(80)
            chain = padded[21].strip() or "A"
            resseq = padded[22:26].strip()
            atom_name = padded[12:16].strip()
            resname = padded[17:20].strip()
            element = padded[76:78].strip() or atom_name[:1]
            try:
                x, y, z = (float(padded[30:38]), float(padded[38:46]),
                           float(padded[46:54]))
            except ValueError:
                continue
            if chain == sel_chain and resseq == sel_resnum and atom_name == sel_atom \
                    and resname != lig_resname:
                rec_xyz = (x, y, z)
            if (resname == lig_resname and element.upper() not in ("H", "D")
                    and (lig_resseq is None or resseq == str(lig_resseq))):
                lig_heavy_atoms.append((x, y, z, atom_name))

    if rec_xyz is None or not lig_heavy_atoms:
        return None
    # Duplicates of the receptor atom are not bonds.
    lig_heavy_atoms = [a for a in lig_heavy_atoms
                       if math.dist(a[:3], rec_xyz) >= MIN_COVALENT_BOND] or lig_heavy_atoms
    best = min(lig_heavy_atoms, key=lambda a: math.dist(a[:3], rec_xyz))
    return {"rec_xyz": list(rec_xyz), "lig_xyz": list(best[:3]),
            "distance": math.dist(best[:3], rec_xyz),
            "rec_atom": sel_atom, "lig_atom": best[3]}
