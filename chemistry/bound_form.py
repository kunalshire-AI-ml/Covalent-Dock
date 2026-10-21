"""
bound_form.py — build the ligand in its *reacted* state.

gnina does not model the reaction. It expects the ligand already in the form it
takes after the bond forms: the acrylamide double bond saturated, the epoxide
opened, the chloride gone. Docking the unreacted form is the mistake that
produces a pattern like [CH2]=[CH]C(=O)N that then fails to match the file
being docked.

Each transform edits the molecule, keeps track of the reactive atom across the
edit, and regenerates 3-D coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

from .warheads import WARHEADS

_TAG = "_covalent_reactive_atom"


@dataclass
class BoundForm:
    mol: Chem.Mol
    reactive_idx: int
    changed: bool
    description: str
    smiles: str
    removed_atoms: int = 0
    warnings: List[str] = field(default_factory=list)
    embedded: bool = False
    #: Atoms the transform turned into stereocentres without being able to say
    #: which configuration they take. See `new_stereocentres` below.
    undefined_stereocentres: List[int] = field(default_factory=list)

    @property
    def stereo_ambiguous(self) -> bool:
        return bool(self.undefined_stereocentres)


# ─────────────────────────────────────────────────────────────────────────────
#  helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unassigned_centres(mol: Chem.Mol) -> set:
    """Atom indices that are stereocentres with no configuration assigned."""
    try:
        found = Chem.FindMolChiralCenters(mol, includeUnassigned=True,
                                          useLegacyImplementation=False)
    except Exception:                                         # noqa: BLE001
        return set()
    return {idx for idx, label in found if label == "?"}


def new_stereocentres(before: Chem.Mol, after: Chem.Mol) -> List[int]:
    """
    Stereocentres the transform created and could not assign.

    Reducing a carbonyl to a carbinol makes the carbon tetrahedral, and which
    face the nucleophile added to is a property of the enzyme, not of the
    ligand — so there is no configuration to deduce here. Left undefined, the
    embedding picks one arbitrarily: reproducible for a given RDKit build, and
    liable to flip to the other diastereomer on a different one. The wrong
    diastereomer docks worse for a reason nothing in the output explains, so
    the ambiguity has to be visible rather than silently resolved.
    """
    try:
        was = _unassigned_centres(before)
        now = _unassigned_centres(after)
    except Exception:                                         # noqa: BLE001
        return []
    return sorted(now - was)


def _reset_hydrogens(atom: Chem.Atom) -> None:
    """Let RDKit recompute the hydrogen count for an atom we just changed."""
    atom.SetNumExplicitHs(0)
    atom.SetNoImplicit(False)


def _tag(mol: Chem.Mol, idx: int) -> Chem.Mol:
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.ClearProp(_TAG) if atom.HasProp(_TAG) else None
    mol.GetAtomWithIdx(idx).SetBoolProp(_TAG, True)
    return mol


def _find_tag(mol: Chem.Mol) -> int:
    for atom in mol.GetAtoms():
        if atom.HasProp(_TAG):
            return atom.GetIdx()
    return -1


def _strip_tag(mol: Chem.Mol) -> None:
    for atom in mol.GetAtoms():
        if atom.HasProp(_TAG):
            atom.ClearProp(_TAG)


def _delete_fragment_of(rw: Chem.RWMol, seed: int, keep: int) -> int:
    """Delete the connected fragment containing `seed` but not `keep`."""
    seen, stack = {seed}, [seed]
    while stack:
        cur = stack.pop()
        for nb in rw.GetAtomWithIdx(cur).GetNeighbors():
            j = nb.GetIdx()
            if j not in seen:
                seen.add(j)
                stack.append(j)
    if keep in seen:
        return 0
    for idx in sorted(seen, reverse=True):
        rw.RemoveAtom(idx)
    return len(seen)


# ─────────────────────────────────────────────────────────────────────────────
#  transforms
# ─────────────────────────────────────────────────────────────────────────────

def _saturate_alkene(rw: Chem.RWMol, idx: int) -> Tuple[bool, str, int]:
    atom = rw.GetAtomWithIdx(idx)
    for bond in atom.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            other = bond.GetOtherAtom(atom)
            if other.GetSymbol() != "C":
                continue
            bond.SetBondType(Chem.BondType.SINGLE)
            _reset_hydrogens(atom)
            _reset_hydrogens(other)
            return True, "C=C saturated to C–C (conjugate addition)", 0
    return False, "No C=C double bond found at this atom", 0


def _reduce_alkyne(rw: Chem.RWMol, idx: int) -> Tuple[bool, str, int]:
    atom = rw.GetAtomWithIdx(idx)
    for bond in atom.GetBonds():
        if bond.GetBondType() == Chem.BondType.TRIPLE:
            other = bond.GetOtherAtom(atom)
            bond.SetBondType(Chem.BondType.DOUBLE)
            _reset_hydrogens(atom)
            _reset_hydrogens(other)
            return True, "C≡C reduced to C=C (vinyl adduct)", 0
    return False, "No C≡C triple bond found at this atom", 0


def _drop_leaving_group(rw: Chem.RWMol, idx: int,
                        leaving: Optional[int]) -> Tuple[bool, str, int]:
    atom = rw.GetAtomWithIdx(idx)
    target = leaving
    if target is None or not rw.GetBondBetweenAtoms(idx, target):
        halides = [nb.GetIdx() for nb in atom.GetNeighbors()
                   if nb.GetSymbol() in ("F", "Cl", "Br", "I")]
        target = halides[0] if halides else None
    if target is None:
        for nb in atom.GetNeighbors():
            if nb.GetSymbol() == "O" and nb.GetDegree() == 2:
                target = nb.GetIdx()
                break
    if target is None:
        return False, "No leaving group found on this atom", 0

    symbol = rw.GetAtomWithIdx(target).GetSymbol()
    rw.RemoveBond(idx, target)
    removed = _delete_fragment_of(rw, target, keep=idx)
    _reset_hydrogens(rw.GetAtomWithIdx(idx))
    return True, f"{symbol} leaving group removed ({removed} atom(s))", removed


def _open_strained_ring(rw: Chem.RWMol, idx: int,
                        leaving: Optional[int]) -> Tuple[bool, str, int]:
    atom = rw.GetAtomWithIdx(idx)
    hetero = None
    if leaving is not None and rw.GetBondBetweenAtoms(idx, leaving):
        hetero = leaving
    if hetero is None:
        for nb in atom.GetNeighbors():
            if nb.GetSymbol() in ("O", "N") and nb.IsInRing():
                hetero = nb.GetIdx()
                break
    if hetero is None:
        return False, "No ring heteroatom found next to this atom", 0

    het_atom = rw.GetAtomWithIdx(hetero)
    symbol = het_atom.GetSymbol()
    rw.RemoveBond(idx, hetero)
    _reset_hydrogens(atom)
    _reset_hydrogens(het_atom)
    label = "hydroxyl" if symbol == "O" else "amine"
    return True, f"Ring opened at C–{symbol}; the {symbol} becomes a {label}", 0


def _reduce_carbonyl(rw: Chem.RWMol, idx: int) -> Tuple[bool, str, int]:
    atom = rw.GetAtomWithIdx(idx)
    for bond in atom.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE and \
                bond.GetOtherAtom(atom).GetSymbol() == "O":
            other = bond.GetOtherAtom(atom)
            bond.SetBondType(Chem.BondType.SINGLE)
            _reset_hydrogens(atom)
            _reset_hydrogens(other)
            return True, "C=O reduced to C–OH (tetrahedral adduct)", 0
    return False, "No C=O found at this atom", 0


def _reduce_nitrile(rw: Chem.RWMol, idx: int) -> Tuple[bool, str, int]:
    atom = rw.GetAtomWithIdx(idx)
    for bond in atom.GetBonds():
        if bond.GetBondType() == Chem.BondType.TRIPLE and \
                bond.GetOtherAtom(atom).GetSymbol() == "N":
            other = bond.GetOtherAtom(atom)
            bond.SetBondType(Chem.BondType.DOUBLE)
            _reset_hydrogens(atom)
            _reset_hydrogens(other)
            return True, "C≡N reduced to C=N (thioimidate adduct)", 0
    return False, "No C≡N found at this atom", 0


def _reduce_cumulene(rw: Chem.RWMol, idx: int) -> Tuple[bool, str, int]:
    atom = rw.GetAtomWithIdx(idx)
    for bond in atom.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE and \
                bond.GetOtherAtom(atom).GetSymbol() == "N":
            other = bond.GetOtherAtom(atom)
            bond.SetBondType(Chem.BondType.SINGLE)
            _reset_hydrogens(atom)
            _reset_hydrogens(other)
            return True, "C=N reduced to C–NH (addition across the cumulene)", 0
    return False, "No C=N found at this atom", 0


def _cleave_disulfide(rw: Chem.RWMol, idx: int,
                      leaving: Optional[int]) -> Tuple[bool, str, int]:
    atom = rw.GetAtomWithIdx(idx)
    partner = leaving
    if partner is None or not rw.GetBondBetweenAtoms(idx, partner):
        for nb in atom.GetNeighbors():
            if nb.GetSymbol() == "S":
                partner = nb.GetIdx()
                break
    if partner is None:
        return False, "No S–S bond found at this atom", 0
    rw.RemoveBond(idx, partner)
    removed = _delete_fragment_of(rw, partner, keep=idx)
    _reset_hydrogens(atom)
    return True, f"S–S cleaved; leaving thiol removed ({removed} atom(s))", removed


# ─────────────────────────────────────────────────────────────────────────────
#  public API
# ─────────────────────────────────────────────────────────────────────────────

def _demote_broken_aromatics(mol: Chem.Mol) -> None:
    """Clear aromatic flags left behind on atoms and bonds no longer in a ring."""
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
    except Exception:                                         # noqa: BLE001
        return
    for bond in mol.GetBonds():
        if bond.GetIsAromatic() and not bond.IsInRing():
            bond.SetIsAromatic(False)
            bond.SetBondType(Chem.BondType.SINGLE)
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic() and not atom.IsInRing():
            atom.SetIsAromatic(False)
            _reset_hydrogens(atom)


def transform_description(transform: str) -> str:
    return {
        "saturate_alkene": "saturate the C=C so the β-carbon is sp³",
        "reduce_alkyne": "reduce the C≡C to C=C",
        "drop_leaving_group": "remove the leaving group",
        "open_strained_ring": "open the strained ring",
        "reduce_carbonyl": "reduce C=O to C–OH",
        "reduce_nitrile": "reduce C≡N to C=N",
        "reduce_cumulene": "add across the cumulene",
        "cleave_disulfide": "cleave the S–S bond",
        "none": "no change needed",
    }.get(transform, transform)


def build_bound_form(
    mol: Chem.Mol,
    reactive_idx: int,
    warhead_key: Optional[str] = None,
    transform: Optional[str] = None,
    leaving_atom: Optional[int] = None,
    embed: bool = True,
    random_seed: int = 42,
) -> BoundForm:
    """
    Return the ligand as it looks *after* the covalent bond forms.

    The reactive atom index is tracked across the edit, so the returned
    `reactive_idx` is the atom to hand to the SMARTS generator.
    """
    warnings: List[str] = []
    if transform is None:
        transform = WARHEADS[warhead_key].transform if warhead_key in WARHEADS else "none"
    if leaving_atom is None and warhead_key in WARHEADS:
        leaving_atom = None  # resolved from topology inside the transforms

    tagged = _tag(mol, reactive_idx)
    work = Chem.RemoveHs(tagged, sanitize=False)
    work.UpdatePropertyCache(strict=False)
    idx = _find_tag(work)
    if idx < 0:
        idx = reactive_idx
        warnings.append("Could not track the reactive atom through hydrogen removal.")

    if transform == "none":
        _strip_tag(work)
        return BoundForm(
            mol=work, reactive_idx=idx, changed=False,
            description="No modification needed for this warhead.",
            smiles=Chem.MolToSmiles(work), warnings=warnings,
        )

    rw = Chem.RWMol(work)
    handlers = {
        "saturate_alkene": lambda: _saturate_alkene(rw, idx),
        "reduce_alkyne": lambda: _reduce_alkyne(rw, idx),
        "drop_leaving_group": lambda: _drop_leaving_group(rw, idx, leaving_atom),
        "open_strained_ring": lambda: _open_strained_ring(rw, idx, leaving_atom),
        "reduce_carbonyl": lambda: _reduce_carbonyl(rw, idx),
        "reduce_nitrile": lambda: _reduce_nitrile(rw, idx),
        "reduce_cumulene": lambda: _reduce_cumulene(rw, idx),
        "cleave_disulfide": lambda: _cleave_disulfide(rw, idx, leaving_atom),
    }
    handler = handlers.get(transform)
    if handler is None:
        _strip_tag(work)
        return BoundForm(mol=work, reactive_idx=idx, changed=False,
                         description=f"Unknown transform '{transform}'.",
                         smiles=Chem.MolToSmiles(work),
                         warnings=warnings + ["Ligand left unchanged."])

    changed, description, removed = handler()
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:                                         # noqa: BLE001
        # Opening an aromatic ring — ebselen's Se–N, for instance — leaves its
        # atoms still flagged aromatic while no longer belonging to any ring,
        # and the molecule can then never be kekulised. Drop the flags that the
        # edit invalidated and try again before giving up.
        _demote_broken_aromatics(out)
        try:
            Chem.SanitizeMol(out)
        except Exception as exc:                              # noqa: BLE001
            warnings.append(f"Sanitisation after the edit: {exc}")
            out.UpdatePropertyCache(strict=False)

    new_idx = _find_tag(out)
    if new_idx < 0:
        new_idx = idx
        warnings.append("Reactive atom index could not be re-located after the edit.")
    _strip_tag(out)

    # Which face the nucleophile added to is the enzyme's doing, so a transform
    # that makes a carbon tetrahedral cannot say what configuration it takes.
    # Detected before embedding, because the embedding is what silently picks
    # one. (`_tag` is a plain bool property and does not perturb stereo
    # perception, so `work` can be compared as it stands.)
    stereocentres = new_stereocentres(work, out) if changed else []
    if stereocentres:
        names = ", ".join(f"atom {i + 1}" for i in stereocentres)
        warnings.append(
            f"This transform makes {names} a stereocentre, and which "
            "configuration it takes depends on the enzyme, not the ligand. One "
            "is chosen arbitrarily when the 3-D coordinates are generated — a "
            "different RDKit build may choose the other. If the pose or the "
            "score looks wrong, the other diastereomer is worth docking too.")

    embedded = False
    if changed and embed:
        out, ok, msg = embed_3d(out, random_seed=random_seed)
        embedded = ok
        if not ok:
            warnings.append(msg)

    return BoundForm(
        mol=out,
        reactive_idx=new_idx,
        changed=changed,
        description=description,
        smiles=Chem.MolToSmiles(Chem.RemoveHs(out)),
        removed_atoms=removed,
        warnings=warnings,
        embedded=embedded,
        undefined_stereocentres=stereocentres,
    )


def both_epimers(bound: "BoundForm", random_seed: int = 42) -> Tuple[Chem.Mol, int]:
    """
    The bound form carrying one conformer per configuration of its undefined
    stereocentre. Returns (molecule, number of conformers).

    Opt-in, and deliberately so: it changes what gets docked. When the adduct
    stereochemistry is unknown — which it is, prospectively — both
    diastereomers are legitimate hypotheses, and docking both lets the score
    choose instead of the embedding choosing by accident.

    They are returned as conformers of one molecule rather than two molecules
    because epimers share a graph and differ only in 3-D, so the existing
    multi-conformer path writes and places them unchanged.

    If there is nothing ambiguous, the molecule comes back as it was.
    """
    centre = bound.undefined_stereocentres[0] if bound.undefined_stereocentres else None
    if centre is None:
        return bound.mol, max(bound.mol.GetNumConformers(), 1)

    conformers = []
    for tag in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
        trial = Chem.Mol(bound.mol)
        trial.RemoveAllConformers()
        if centre >= trial.GetNumAtoms():
            continue
        trial.GetAtomWithIdx(centre).SetChiralTag(tag)
        try:
            Chem.AssignStereochemistry(trial, cleanIt=True, force=True)
        except Exception:                                     # noqa: BLE001
            pass
        built, ok, _ = embed_3d(trial, random_seed=random_seed)
        if ok and built.GetNumConformers():
            conformers.append(built)

    if len(conformers) < 2:
        return bound.mol, max(bound.mol.GetNumConformers(), 1)

    merged = Chem.Mol(conformers[0])
    for extra in conformers[1:]:
        # Same graph and same atom order — both come from the same molecule —
        # so a conformer can be carried straight across.
        for conf in extra.GetConformers():
            merged.AddConformer(Chem.Conformer(conf), assignId=True)
    return merged, merged.GetNumConformers()


def sdf_text(bound: "BoundForm", random_seed: int = 42) -> str:
    """
    A mol block for the bound form, for the 3-D panel. Empty string if it
    cannot be produced.

    `build_bound_form` embeds only when it actually edited the molecule, so a
    warhead whose transform is 'none' arrives here with no conformer. The 3-D
    view needs coordinates either way, so embed on demand — on a copy, since
    the caller's molecule is what the SMARTS indices refer to.
    """
    mol = bound.mol
    if mol.GetNumConformers() == 0:
        mol, ok, _ = embed_3d(Chem.Mol(mol), random_seed=random_seed)
        if not ok:
            return ""
    try:
        return Chem.MolToMolBlock(mol)
    except Exception:                                         # noqa: BLE001
        return ""


def embed_3d(mol: Chem.Mol, random_seed: int = 42,
             optimise: bool = True) -> Tuple[Chem.Mol, bool, str]:
    """ETKDGv3 embedding + MMFF94 optimisation, same recipe as prep_ligands.py."""
    mol = Chem.AddHs(mol, addCoords=True)
    # MMFF needs ring perception, and not every caller arrives sanitised: the
    # 'none' transform returns the molecule straight out of a RemoveHs with
    # sanitize=False, and MMFF then aborts on "RingInfo not initialized"
    # instead of returning a status. Sanitise where we can, and fall back to
    # ring perception alone where the molecule will not sanitise.
    try:
        Chem.SanitizeMol(mol)
    except Exception:                                         # noqa: BLE001
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
    try:
        params = AllChem.ETKDGv3()
    except AttributeError:
        params = AllChem.ETKDG()
    params.randomSeed = random_seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            return mol, False, "3-D embedding failed; coordinates may be unusable."
    if optimise:
        status = AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94", maxIters=2000)
        if status == -1:
            return mol, True, "MMFF94 setup failed; using the unoptimised geometry."
        if status == 1:
            return mol, True, "MMFF94 did not fully converge."
    return mol, True, ""
