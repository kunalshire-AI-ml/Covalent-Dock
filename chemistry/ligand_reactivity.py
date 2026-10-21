"""
ligand_reactivity.py — ligand side of the covalent-docking workflow.

Load a ligand from whatever the user has, then report the atoms that could act
as electrophiles, each tagged with its warhead type and role. The UI shows
these instead of all 71 atoms.

Nothing here decides that a reaction *will* happen; it generates candidates.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

from . import warheads as wh
from .warheads import Warhead

RDLogger.DisableLog("rdApp.warning")

SUPPORTED_SUFFIXES = (".sdf", ".mol", ".mol2", ".pdb", ".smi", ".smiles", ".txt")


# ─────────────────────────────────────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoadedLigand:
    mol: Chem.Mol
    name: str
    source_format: str
    has_3d: bool
    n_atoms: int
    n_heavy: int
    formula: str = ""
    warnings: List[str] = field(default_factory=list)


def _finalize(mol: Chem.Mol, name: str, fmt: str, warns: List[str]) -> LoadedLigand:
    from rdkit.Chem import rdMolDescriptors

    if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
        name = mol.GetProp("_Name").strip()
    else:
        mol.SetProp("_Name", name)
    has_3d = mol.GetNumConformers() > 0 and mol.GetConformer().Is3D()
    return LoadedLigand(
        mol=mol,
        name=name,
        source_format=fmt,
        has_3d=has_3d,
        n_atoms=mol.GetNumAtoms(),
        n_heavy=mol.GetNumHeavyAtoms(),
        formula=rdMolDescriptors.CalcMolFormula(mol),
        warnings=warns,
    )


def load_ligand_text(text: str, fmt: str, name: str = "ligand") -> LoadedLigand:
    """fmt: sdf | mol | mol2 | pdb | smiles"""
    warns: List[str] = []
    fmt = fmt.lower().lstrip(".")
    mol = None

    if fmt in ("sdf", "mol"):
        mol = Chem.MolFromMolBlock(text, removeHs=False, sanitize=True)
        if mol is None:
            mol = Chem.MolFromMolBlock(text, removeHs=False, sanitize=False)
            if mol is not None:
                warns.append("Sanitisation failed; the structure was read as-is.")
                _try_sanitize(mol, warns)
    elif fmt == "mol2":
        mol = Chem.MolFromMol2Block(text, removeHs=False, sanitize=True)
    elif fmt == "pdb":
        mol = Chem.MolFromPDBBlock(text, removeHs=False, sanitize=True)
        if mol is not None:
            warns.append(
                "PDB carries no bond orders — double bonds may be missing. "
                "Prefer SDF/MOL, or supply a SMILES to assign bond orders."
            )
    elif fmt in ("smi", "smiles"):
        smiles = text.strip().splitlines()[0].split()[0] if text.strip() else ""
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and not mol.HasProp("_Name"):
            mol.SetProp("_Name", name)
    else:
        raise ValueError(f"Unsupported ligand format: {fmt}")

    if mol is None:
        raise ValueError(f"Could not parse the ligand as {fmt}.")
    # MolFromSmiles("") returns an empty molecule rather than None, so an empty
    # or whitespace-only input would otherwise be accepted as a 0-atom ligand
    # and only fail much later, with a confusing message.
    if mol.GetNumAtoms() == 0:
        raise ValueError(f"The {fmt} input contains no atoms.")
    return _finalize(mol, name, fmt, warns)


def _try_sanitize(mol: Chem.Mol, warns: List[str]) -> None:
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:                                  # noqa: BLE001
        warns.append(f"Partial sanitisation: {exc}")
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )


def load_ligand_file(path: str) -> LoadedLigand:
    ext = os.path.splitext(path)[1].lower()
    name = os.path.splitext(os.path.basename(path))[0]
    if ext not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported ligand file type: {ext}")
    with open(path, "r", errors="replace") as fh:
        text = fh.read()
    fmt = {".smi": "smiles", ".smiles": "smiles", ".txt": "smiles"}.get(ext, ext.lstrip("."))
    return load_ligand_text(text, fmt, name=name)


def assign_bond_orders_from_smiles(mol: Chem.Mol, smiles: str) -> Chem.Mol:
    """Fix a PDB/coordinate-only ligand using a reference SMILES."""
    ref = Chem.MolFromSmiles(smiles)
    if ref is None:
        raise ValueError("Reference SMILES could not be parsed.")
    return AllChem.AssignBondOrdersFromTemplate(ref, mol)


# ─────────────────────────────────────────────────────────────────────────────
#  Electrophile detection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Electrophile:
    """One candidate reactive atom on the ligand."""
    atom_idx: int                       # 0-based RDKit index
    element: str
    label: str                          # e.g. "C25"
    warhead_key: str
    warhead_label: str
    role: str
    role_label: str
    mechanism: str
    mechanism_label: str
    nucleophiles: Tuple[str, ...]
    bond_order: int
    reversible: bool
    confidence: str                     # high | medium | low
    match_atoms: Tuple[int, ...]        # every atom of the matched motif
    leaving_atom: Optional[int] = None
    transform: str = "none"
    note: str = ""
    alternates: List[str] = field(default_factory=list)   # other warhead keys hit

    @property
    def serial(self) -> int:
        """1-based index, which is what the atom label shows."""
        return self.atom_idx + 1

    def describe(self) -> str:
        bits = [f"{self.warhead_label} {self.role_label}"]
        if self.reversible:
            bits.append("reversible")
        return " · ".join(bits)


def atom_label(mol: Chem.Mol, idx: int) -> str:
    """
    The ligand's own PDB atom name where it has one, otherwise 'C#25'.

    The '#' matters. Without it an index-derived label like 'C25' is
    indistinguishable from a real PDB atom name, and the two are unrelated:
    RDKit numbers atoms in its own order, so 'C14' from a SMILES is almost
    never the crystal's C14. Comparing them has already caused a correct pick
    to be read as a wrong one. A label with no '#' is the deposited name and
    can be compared with the structure; one with '#' cannot.
    """
    atom = mol.GetAtomWithIdx(idx)
    info = atom.GetPDBResidueInfo()
    if info is not None and info.GetName().strip():
        return info.GetName().strip()
    return f"{atom.GetSymbol()}#{idx + 1}"


def heavy_atom_labels(mol: Chem.Mol) -> Dict[int, str]:
    """
    Atom index -> the name that atom is shown under, hydrogens left out.

    One source of truth for the flat depiction and the 3-D panel, so the two
    never disagree about which atom is which.
    """
    return {atom.GetIdx(): atom_label(mol, atom.GetIdx())
            for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1}


def _confidence(w: Warhead) -> str:
    if w.promiscuous:
        return "low"
    if w.priority <= 15:
        return "high"
    if w.priority <= 30:
        return "medium"
    return "medium" if not w.key.endswith("_generic") else "low"


def find_electrophiles(
    mol: Chem.Mol,
    include_promiscuous: bool = True,
    include_reversible: bool = True,
) -> List[Electrophile]:
    """
    Run the warhead library over the molecule.

    One entry per reactive atom. If several patterns hit the same atom the most
    specific wins and the others are listed in `alternates`.
    """
    best: Dict[int, Tuple[Warhead, Tuple[int, ...]]] = {}
    alternates: Dict[int, List[str]] = {}

    for w in wh.WARHEAD_ORDER:
        if w.promiscuous and not include_promiscuous:
            continue
        if w.reversible and not include_reversible:
            continue
        patt = Chem.MolFromSmarts(w.smarts)
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt, uniquify=True):
            if w.reactive_index >= len(match):
                continue
            idx = match[w.reactive_index]
            current = best.get(idx)
            if current is None or w.priority < current[0].priority:
                if current is not None:
                    alternates.setdefault(idx, []).append(current[0].key)
                best[idx] = (w, tuple(match))
            elif w.key != current[0].key:
                alternates.setdefault(idx, []).append(w.key)

    out: List[Electrophile] = []
    for idx, (w, match) in best.items():
        leaving = None
        if w.leaving_index is not None and w.leaving_index < len(match):
            leaving = match[w.leaving_index]
        atom = mol.GetAtomWithIdx(idx)
        out.append(
            Electrophile(
                atom_idx=idx,
                element=atom.GetSymbol(),
                label=atom_label(mol, idx),
                warhead_key=w.key,
                warhead_label=w.label,
                role=w.role,
                role_label=wh.role_label(w.role),
                mechanism=w.mechanism,
                mechanism_label=wh.mechanism_label(w.mechanism),
                nucleophiles=w.nucleophiles,
                bond_order=w.bond_order,
                reversible=w.reversible,
                confidence=_confidence(w),
                match_atoms=match,
                leaving_atom=leaving,
                transform=w.transform,
                note=w.note,
                alternates=sorted(set(alternates.get(idx, []))),
            )
        )

    rank = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda e: (rank[e.confidence], e.atom_idx))
    return out


def group_by_warhead(electrophiles: Sequence[Electrophile]) -> Dict[str, List[Electrophile]]:
    groups: Dict[str, List[Electrophile]] = {}
    for e in electrophiles:
        groups.setdefault(e.warhead_label, []).append(e)
    return groups


def atom_environment(mol: Chem.Mol, idx: int) -> Dict[str, object]:
    """Chemical context of one atom — what the app knows after a click."""
    atom = mol.GetAtomWithIdx(idx)
    neighbours = []
    for nb in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(idx, nb.GetIdx())
        neighbours.append({
            "label": atom_label(mol, nb.GetIdx()),
            "element": nb.GetSymbol(),
            "bond": str(bond.GetBondType()).replace("BondType.", "").lower(),
            "aromatic": nb.GetIsAromatic(),
        })
    return {
        "index": idx,
        "label": atom_label(mol, idx),
        "element": atom.GetSymbol(),
        "hybridization": str(atom.GetHybridization()),
        "aromatic": atom.GetIsAromatic(),
        "in_ring": atom.IsInRing(),
        "formal_charge": atom.GetFormalCharge(),
        "explicit_hs": atom.GetTotalNumHs(),
        "degree": atom.GetDegree(),
        "neighbours": neighbours,
    }


def describe_ligand(mol: Chem.Mol) -> Dict[str, object]:
    return {
        "smiles": Chem.MolToSmiles(Chem.RemoveHs(mol)),
        "mw": round(Descriptors.MolWt(mol), 2),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "rings": mol.GetRingInfo().NumRings(),
        "conformers": mol.GetNumConformers(),
    }
