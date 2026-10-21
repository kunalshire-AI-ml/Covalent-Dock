"""
prepare_ligand.py — 3-D ligand files for gnina.

The RDKit path (ETKDGv3 + MMFF94) and the OpenBabel fallback are lifted from
prep_ligands.py; what is new is that the molecule being written is normally the
*bound form* produced by chemistry/bound_form.py, and that the reactive atom
index is checked after writing, because the SMARTS has to match this file.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

XYZ = Tuple[float, float, float]


@dataclass
class Placement:
    """What the initial covalent placement did to the ligand's coordinates."""
    moved: bool
    start_distance: Optional[float] = None    # reactive atom → anchor, before
    final_distance: Optional[float] = None    # …and after
    translation: XYZ = (0.0, 0.0, 0.0)
    rotated: bool = False
    note: str = ""
    required_box: Optional[float] = None   # box edge that holds the placed ligand


@dataclass
class LigandPrep:
    path: str
    name: str
    method: str
    n_atoms: int
    n_conformers: int
    reactive_idx: Optional[int] = None
    warnings: List[str] = field(default_factory=list)
    placement: Optional[Placement] = None

    def report(self) -> str:
        bits = [f"{self.n_atoms} atoms", f"{self.n_conformers} conformer(s)",
                self.method]
        if self.placement and self.placement.moved:
            bits.append(f"placed {self.placement.final_distance:.2f} Å "
                        "from the receptor atom")
        return " · ".join(bits)


# ─────────────────────────────────────────────────────────────────────────────
#  Initial covalent placement
# ─────────────────────────────────────────────────────────────────────────────
#
# An unbound ligand is embedded wherever RDKit's conformer generator puts it,
# usually around the origin, while the protein sits at its crystallographic
# coordinates. Those two frames have nothing to do with each other, so the
# reactive atom can start tens of ångström outside a box centred on the
# receptor atom — gnina then reports "initial pose not within box" and searches
# from wherever it can reach.
#
# So before writing the SDF the ligand is moved as a rigid body: rotated about
# its reactive atom to point away from the protein, then translated so that
# atom sits one bond length off the receptor atom. Bond lengths, angles and
# torsions are all untouched. This is only a starting point — gnina still does
# the real search, and none of it asserts that this is the binding pose.

# Ideal heavy-atom bond length to the attacking nucleophile (Å).
COVALENT_BOND_LENGTH = {"S": 1.81, "SE": 1.95, "N": 1.47, "O": 1.43, "C": 1.54}
DEFAULT_BOND_LENGTH = 1.80


def _sub(a: Sequence[float], b: Sequence[float]) -> XYZ:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _length(v: Sequence[float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _unit(v: Sequence[float]) -> Optional[XYZ]:
    n = _length(v)
    return (v[0] / n, v[1] / n, v[2] / n) if n > 1e-9 else None


def _cross(a: Sequence[float], b: Sequence[float]) -> XYZ:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def escape_vector(anchor_xyz: Sequence[float],
                  protein_xyz: Sequence[Sequence[float]],
                  radius: float = 8.0) -> XYZ:
    """
    Unit vector pointing from the local protein bulk out towards solvent.

    The centroid of the atoms packed around the receptor atom sits inside the
    protein, so the direction from that centroid to the atom itself points out
    of the pocket — which is the side a ligand has to approach from.
    """
    near = [p for p in protein_xyz if _length(_sub(p, anchor_xyz)) <= radius]
    pool = near or list(protein_xyz)
    if not pool:
        return (1.0, 0.0, 0.0)
    centroid = (sum(p[0] for p in pool) / len(pool),
                sum(p[1] for p in pool) / len(pool),
                sum(p[2] for p in pool) / len(pool))
    return _unit(_sub(anchor_xyz, centroid)) or (1.0, 0.0, 0.0)


def _rotation_matrix(src: XYZ, dst: XYZ) -> List[List[float]]:
    """Rodrigues rotation taking unit vector `src` onto unit vector `dst`."""
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(src, dst))))
    if dot > 1.0 - 1e-9:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    if dot < -1.0 + 1e-9:
        # Antiparallel: spin 180° about any axis perpendicular to src.
        seed = (1.0, 0.0, 0.0) if abs(src[0]) < 0.9 else (0.0, 1.0, 0.0)
        axis = _unit(_cross(src, seed)) or (0.0, 0.0, 1.0)
        angle = math.pi
    else:
        axis = _unit(_cross(src, dst)) or (0.0, 0.0, 1.0)
        angle = math.acos(dot)

    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return [
        [t * x * x + c,     t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c,     t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]


def place_reactive_atom(
    mol: Chem.Mol,
    reactive_idx: int,
    anchor_xyz: Sequence[float],
    anchor_element: str = "S",
    protein_xyz: Sequence[Sequence[float]] = (),
    bond_length: Optional[float] = None,
    orient: bool = True,
) -> Placement:
    """
    Move the ligand so `reactive_idx` starts one bond length from `anchor_xyz`.

    Edits every conformer of `mol` in place and reports what it did. A rigid
    transform, so the conformer's internal geometry survives exactly.
    """
    if mol.GetNumConformers() == 0:
        return Placement(moved=False, note="Ligand has no 3-D conformer to place.")
    if not (0 <= reactive_idx < mol.GetNumAtoms()):
        return Placement(moved=False,
                         note=f"Reactive atom {reactive_idx} is outside the molecule.")

    if bond_length is None:
        bond_length = COVALENT_BOND_LENGTH.get(
            (anchor_element or "").strip().upper(), DEFAULT_BOND_LENGTH)

    anchor = (float(anchor_xyz[0]), float(anchor_xyz[1]), float(anchor_xyz[2]))
    out = escape_vector(anchor, protein_xyz)
    target = (anchor[0] + out[0] * bond_length,
              anchor[1] + out[1] * bond_length,
              anchor[2] + out[2] * bond_length)

    first = mol.GetConformer()
    start = first.GetAtomPosition(reactive_idx)
    start_distance = _length(_sub((start.x, start.y, start.z), anchor))

    rotated = False
    for conf in mol.GetConformers():
        positions = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
        coords = [(p.x, p.y, p.z) for p in positions]
        pivot = coords[reactive_idx]

        if orient and len(coords) > 1:
            # Turn the reactive atom's open face towards the nucleophile.
            #
            # The direction away from its own substituents is where an incoming
            # nucleophile has room — backside of a leaving group, or the empty
            # lobe above a carbonyl. Orienting on the whole-molecule centroid
            # instead says nothing about the atoms bonded to the reactive atom,
            # and those are exactly the ones that end up jammed into the
            # protein: on 1A46 it left a substituent 0.5 A from Ser195 OG.
            neighbours = [nb.GetIdx() for nb
                          in mol.GetAtomWithIdx(reactive_idx).GetNeighbors()
                          if nb.GetAtomicNum() > 1]
            if neighbours:
                base = (sum(coords[i][0] for i in neighbours) / len(neighbours),
                        sum(coords[i][1] for i in neighbours) / len(neighbours),
                        sum(coords[i][2] for i in neighbours) / len(neighbours))
                facing = _unit(_sub(pivot, base))
                # The nucleophile sits on the far side of the reactive atom
                # from `out`, so the open face has to point back along it.
                want: XYZ = (-out[0], -out[1], -out[2])
            else:
                centroid = (sum(c[0] for c in coords) / len(coords),
                            sum(c[1] for c in coords) / len(coords),
                            sum(c[2] for c in coords) / len(coords))
                facing = _unit(_sub(centroid, pivot))
                want = out
            if facing is not None:
                matrix = _rotation_matrix(facing, want)
                rotated = True
                coords = [
                    tuple(sum(matrix[r][k] * (c[k] - pivot[k]) for k in range(3))
                          + pivot[r] for r in range(3))
                    for c in coords
                ]

        shift = _sub(target, coords[reactive_idx])
        for i, c in enumerate(coords):
            conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(
                c[0] + shift[0], c[1] + shift[1], c[2] + shift[2]))

    moved = mol.GetConformer().GetAtomPosition(reactive_idx)
    final_distance = _length(_sub((moved.x, moved.y, moved.z), anchor))
    return Placement(
        moved=True,
        start_distance=round(start_distance, 3),
        final_distance=round(final_distance, 3),
        translation=tuple(round(v, 3) for v in _sub(target, (start.x, start.y, start.z))),
        rotated=rotated,
        note=(f"Reactive atom started {start_distance:.1f} Å from the receptor "
              f"atom; moved to {final_distance:.2f} Å as a docking start point."),
    )


def required_box_size(mol: Chem.Mol, anchor_xyz: Sequence[float],
                      margin: float = 3.0) -> float:
    """
    Smallest cubic box centred on the receptor atom that the ligand fits in.

    A covalent ligand is tethered at one end, so it sweeps a sphere centred on
    the receptor atom as it rotates — the box has to cover that radius, not the
    ligand's half-length. That is why a box which looks generous next to the
    ligand's size can still be too small: the ligand hangs off one side of the
    centre rather than straddling it.
    """
    if mol.GetNumConformers() == 0:
        return 0.0
    conf = mol.GetConformer()
    reach = 0.0
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        reach = max(reach, _length(_sub((p.x, p.y, p.z), anchor_xyz)))
    return round(2.0 * (reach + margin), 1)


def write_sdf(mol: Chem.Mol, path: str, name: Optional[str] = None,
              props: Optional[dict] = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mol = Chem.Mol(mol)
    if name:
        mol.SetProp("_Name", name)
    for key, value in (props or {}).items():
        mol.SetProp(str(key), str(value))
    writer = Chem.SDWriter(path)
    if mol.GetNumConformers() == 0:
        writer.write(mol)
    else:
        for conf in mol.GetConformers():
            writer.write(mol, confId=conf.GetId())
    writer.close()
    return path


def prepare_with_rdkit(
    mol: Chem.Mol,
    out_path: str,
    name: str = "ligand",
    n_conformers: int = 1,
    random_seed: int = 42,
    optimise: bool = True,
    reactive_idx: Optional[int] = None,
    anchor_xyz: Optional[Sequence[float]] = None,
    anchor_element: str = "S",
    protein_xyz: Sequence[Sequence[float]] = (),
    orient: bool = True,
) -> LigandPrep:
    warnings: List[str] = []
    work = Chem.Mol(mol)
    work.SetProp("_Name", name)
    added_hydrogens = 0

    needs_3d = work.GetNumConformers() == 0 or not work.GetConformer().Is3D()
    if needs_3d:
        work = Chem.AddHs(work, addCoords=True)
        try:
            params = AllChem.ETKDGv3()
        except AttributeError:
            params = AllChem.ETKDG()
        params.randomSeed = random_seed
        if n_conformers > 1:
            ids = AllChem.EmbedMultipleConfs(work, numConfs=n_conformers, params=params)
            if not ids:
                warnings.append("Multi-conformer embedding failed; trying a single conformer.")
                if AllChem.EmbedMolecule(work, params) != 0:
                    warnings.append("3-D embedding failed.")
        else:
            if AllChem.EmbedMolecule(work, params) != 0:
                if AllChem.EmbedMolecule(work, AllChem.ETKDG()) != 0:
                    warnings.append("3-D embedding failed.")
        if optimise and work.GetNumConformers():
            # MMFF needs a sanitised molecule with ring perception. A bound form
            # whose sanitisation failed after the bond edit arrives without it,
            # and MMFF then raises "RingInfo not initialized" instead of
            # optimising — so the geometry silently goes out unrefined.
            try:
                Chem.SanitizeMol(work)
            except Exception:                                 # noqa: BLE001
                work.UpdatePropertyCache(strict=False)
                Chem.FastFindRings(work)
            try:
                AllChem.MMFFOptimizeMoleculeConfs(work, mmffVariant="MMFF94", maxIters=2000)
            except Exception as exc:                          # noqa: BLE001
                warnings.append(f"MMFF94 optimisation skipped: {exc}")
    else:
        # The supplied geometry is kept, but any hydrogen the file left implicit
        # still has to become a real atom. The test used to be "has no hydrogens
        # at all", which silently passed through a partially protonated ligand —
        # polar hydrogens only is a common way to ship a prepared SDF — so that
        # one input path reached gnina less complete than every other, and its
        # scores were not comparable with them.
        #
        # AddHs only fills in what is missing, so an already complete ligand
        # comes back untouched, and it appends, so reactive_idx still refers to
        # the same atom afterwards.
        before = work.GetNumAtoms()
        try:
            work = Chem.AddHs(work, addCoords=True)
        except Exception as exc:                              # noqa: BLE001
            warnings.append(f"Could not add missing hydrogens: {exc}")
        added_hydrogens = work.GetNumAtoms() - before

    # Put the reactive atom at the covalent site before writing, so gnina's
    # search box actually contains the ligand it is asked to bond.
    placement: Optional[Placement] = None
    if anchor_xyz is not None and reactive_idx is not None:
        placement = place_reactive_atom(
            work, reactive_idx, anchor_xyz, anchor_element=anchor_element,
            protein_xyz=protein_xyz, orient=orient)
        if not placement.moved:
            warnings.append(placement.note)
        else:
            placement.required_box = required_box_size(work, anchor_xyz)

    if needs_3d:
        method = "RDKit ETKDGv3 + MMFF94"
    elif added_hydrogens:
        method = f"input geometry kept, {added_hydrogens} hydrogens added"
    else:
        method = "input geometry kept"

    write_sdf(work, out_path, name=name)
    return LigandPrep(
        path=out_path,
        name=name,
        method=method,
        n_atoms=work.GetNumAtoms(),
        n_conformers=max(work.GetNumConformers(), 1),
        reactive_idx=reactive_idx,
        warnings=warnings,
        placement=placement,
    )


def prepare_with_obabel(smiles: str, out_path: str, name: str = "ligand") -> LigandPrep:
    """Fallback when RDKit cannot embed the molecule."""
    obabel = shutil.which("obabel")
    if obabel is None:
        raise RuntimeError("obabel is not on PATH and RDKit embedding failed.")

    flag_sets = [
        ("gen3d + MMFF94", ["--gen3d", "--minimize", "--ff", "MMFF94"]),
        ("gen3d", ["--gen3d"]),
        ("gen3d Best", ["--gen3d", "Best"]),
        ("gen3d Fast", ["--gen3d", "Fast"]),
    ]
    warnings: List[str] = []
    for desc, flags in flag_sets:
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp:
            tmp_path = tmp.name
        cmd = [obabel, f"-:{smiles}", "-osdf", "-O", tmp_path, "--title", name] + flags
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            os.replace(tmp_path, out_path)
            mol = Chem.MolFromMolFile(out_path, removeHs=False)
            return LigandPrep(
                path=out_path, name=name, method=f"OpenBabel {desc}",
                n_atoms=mol.GetNumAtoms() if mol else 0, n_conformers=1,
                warnings=warnings,
            )
        warnings.append(f"obabel '{desc}' failed: {proc.stderr.strip()[:160]}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    raise RuntimeError("Every obabel flag set failed: " + " | ".join(warnings))


def prepare_ligand(
    mol: Chem.Mol,
    out_path: str,
    name: str = "ligand",
    n_conformers: int = 1,
    reactive_idx: Optional[int] = None,
    random_seed: int = 42,
    anchor_xyz: Optional[Sequence[float]] = None,
    anchor_element: str = "S",
    protein_xyz: Sequence[Sequence[float]] = (),
    orient: bool = True,
) -> LigandPrep:
    """RDKit first, OpenBabel second — the order prep_ligands.py uses."""
    try:
        prep = prepare_with_rdkit(mol, out_path, name=name,
                                  n_conformers=n_conformers,
                                  random_seed=random_seed,
                                  reactive_idx=reactive_idx,
                                  anchor_xyz=anchor_xyz,
                                  anchor_element=anchor_element,
                                  protein_xyz=protein_xyz,
                                  orient=orient)
        if os.path.getsize(out_path) > 0:
            return prep
    except Exception as exc:                                  # noqa: BLE001
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        prep = prepare_with_obabel(smiles, out_path, name=name)
        prep.warnings.insert(0, f"RDKit failed ({exc}); used OpenBabel.")
        prep.reactive_idx = None
        prep.warnings.append(
            "Atom order may differ after the OpenBabel round-trip — "
            "re-check the SMARTS against the written file."
        )
        if anchor_xyz is not None:
            prep.warnings.append(
                "The ligand was not moved to the covalent site: the OpenBabel "
                "fallback does not preserve the reactive atom index. Widen the "
                "box, or fix the RDKit failure above."
            )
        return prep
    raise RuntimeError("Ligand file was written but is empty.")


def verify_pattern_against_file(sdf_path: str, smarts: str) -> dict:
    """
    The check that matters: does the pattern match the file gnina will read?
    """
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        return {"ok": False, "error": "Could not read the written SDF back."}
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return {"ok": False, "error": "SMARTS does not compile."}
    matches = [m[0] for m in mol.GetSubstructMatches(patt, uniquify=True)]
    return {
        "ok": bool(matches),
        "matches": matches,
        "n_matches": len(matches),
        "error": None if matches else "The pattern matches nothing in this file.",
    }
