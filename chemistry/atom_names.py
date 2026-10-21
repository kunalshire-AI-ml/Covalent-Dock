"""
atom_names.py — give ligand atoms their deposited crystal names.

`atom_label` already prefers an atom's PDB name and only falls back to the
'#'-marked index form when there is none. An SDF carries no atom names, so a
ligand loaded from one is labelled C#20 while the same atom is C36 in the
structure the user is looking at in PyMOL. Those two numbers are unrelated —
RDKit orders atoms its own way — and comparing them has already caused a
correct pick to be read as a wrong one.

This module closes that gap by copying the names across from the HETATM records
of the structure itself. Matching is by position: an atom is given a crystal
name only when it is sitting on the crystal atom. That restricts the feature to
a ligand actually taken from the structure, which is the only case where a
deposited name means anything — a conformer generated from SMILES has no
defensible mapping, and guessing one by graph symmetry would invent the very
correspondence the user is trying to check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from rdkit import Chem

from .protein_reactivity import WATERS, Residue, Structure

# A crystal atom and a ligand atom are the same atom when they are on top of
# each other. Coordinates agree to the third decimal when the ligand came out
# of the file; this leaves room for a round-trip through another program.
DEFAULT_TOLERANCE = 0.60

# Single-atom hetero groups are ions and buffer components, never the ligand.
MIN_HETERO_ATOMS = 6


@dataclass
class NameTransfer:
    names: Dict[int, str] = field(default_factory=dict)
    matched: int = 0
    total: int = 0
    max_distance: float = 0.0
    source: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.matched > 0

    @property
    def complete(self) -> bool:
        return self.total > 0 and self.matched == self.total

    def report(self) -> str:
        if not self.ok:
            return "No ligand atom lines up with that residue."
        bits = [f"{self.matched} of {self.total} atoms named from {self.source}"]
        if self.max_distance:
            bits.append(f"worst offset {self.max_distance:.2f} Å")
        return " · ".join(bits)


def _element_of(atom) -> str:
    """The element of a structure atom, inferred from its name if absent."""
    if atom.element:
        return atom.element.strip().upper()
    name = atom.name.strip()
    # PDB atom names put a two-letter element flush left ("CL1"), a one-letter
    # element in the second column (" CA "). Without the original spacing all
    # that is safe is the leading alphabetic run.
    letters = "".join(c for c in name if c.isalpha())
    return letters[:1].upper() if letters else ""


def hetero_candidates(structure: Structure,
                      min_atoms: int = MIN_HETERO_ATOMS) -> List[Residue]:
    """Hetero residues big enough to be the ligand, largest first."""
    out = []
    for chain in structure.chains:
        for res in chain.residues:
            if res.kind != "hetero":
                continue
            if res.resname.strip().upper() in WATERS:
                continue
            if len(res.atoms) < min_atoms:
                continue
            out.append(res)
    out.sort(key=lambda r: (-len(r.atoms), r.chain, r.resseq))
    return out


def residue_label(res: Residue) -> str:
    return f"{res.resname.strip()} {res.chain}:{res.resseq} ({len(res.atoms)} atoms)"


def transfer_from_residue(mol: Chem.Mol, residue: Residue,
                          tolerance: float = DEFAULT_TOLERANCE,
                          conf_id: int = -1) -> NameTransfer:
    """
    Map ligand atoms onto a hetero residue's atoms by position.

    Greedy over all candidate pairs in distance order, so the closest pairing is
    settled first and no crystal atom is claimed twice. Elements must agree: a
    carbon is never given an oxygen's name, however close it sits.
    """
    result = NameTransfer(source=residue_label(residue))
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    result.total = len(heavy)

    if mol.GetNumConformers() == 0:
        result.warnings.append(
            "The ligand has no coordinates, so it cannot be matched to the "
            "structure. Upload it as SDF or PDB taken from the same entry.")
        return result

    conf = mol.GetConformer(conf_id)
    targets = [a for a in residue.atoms if _element_of(a) != "H"]
    if not targets:
        result.warnings.append("That residue has no heavy atoms.")
        return result

    pairs = []
    for idx in heavy:
        sym = mol.GetAtomWithIdx(idx).GetSymbol().upper()
        p = conf.GetAtomPosition(idx)
        for j, atom in enumerate(targets):
            if _element_of(atom) != sym:
                continue
            d = math.dist((p.x, p.y, p.z), atom.xyz)
            if d <= tolerance:
                pairs.append((d, idx, j))

    pairs.sort()
    used_lig, used_res = set(), set()
    for d, idx, j in pairs:
        if idx in used_lig or j in used_res:
            continue
        used_lig.add(idx)
        used_res.add(j)
        result.names[idx] = targets[j].name.strip()
        result.max_distance = max(result.max_distance, d)

    result.matched = len(result.names)
    if not result.matched:
        result.warnings.append(
            "No atom of the ligand is within "
            f"{tolerance:.2f} Å of an atom of {result.source}. This is the "
            "wrong residue, or the ligand is not the one from this structure.")
    elif not result.complete:
        result.warnings.append(
            f"{result.total - result.matched} ligand atoms were left with their "
            "index-based names — they had no partner in the structure. That is "
            "expected when the deposited ligand is missing atoms, and a sign of "
            "a mismatch otherwise.")
    return result


def best_match(mol: Chem.Mol, structure: Structure,
               tolerance: float = DEFAULT_TOLERANCE) -> Optional[NameTransfer]:
    """The hetero residue that best explains this ligand, if any does."""
    best: Optional[NameTransfer] = None
    for res in hetero_candidates(structure):
        got = transfer_from_residue(mol, res, tolerance=tolerance)
        if got.ok and (best is None or got.matched > best.matched):
            best = got
    return best


def apply_names(mol: Chem.Mol, names: Dict[int, str],
                resname: str = "LIG") -> Chem.Mol:
    """
    A copy of `mol` whose atoms carry the given PDB names.

    Returns a copy: the molecule it is given is usually a cached object shared
    across reruns, and renaming it in place would leak into every later use of
    that cache.
    """
    out = Chem.Mol(mol)
    for idx, name in names.items():
        if idx >= out.GetNumAtoms():
            continue
        atom = out.GetAtomWithIdx(idx)
        info = atom.GetPDBResidueInfo()
        if info is None:
            info = Chem.AtomPDBResidueInfo()
            info.SetResidueName(resname)
            info.SetResidueNumber(1)
            info.SetIsHeteroAtom(True)
        info.SetName(name)
        atom.SetMonomerInfo(info)
    return out
