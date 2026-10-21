"""
smarts_generator.py — turn "the user clicked this atom" into a gnina pattern.

gnina takes the covalent ligand atom as a SMARTS expression and bonds through
the *first atom* of the pattern, evaluating every atom that matches. So the
pattern has to (a) be rooted at the selected atom and (b) ideally match that
atom only.

Both are handled by wrapping a rooted fragment in a recursive SMARTS:

    [$([CH3][CH2][C](=O))]

A recursive SMARTS is a single-atom query that matches the root of the
expression inside it, which is exactly the semantics we need. The environment
radius grows until the pattern matches one atom, so the generated pattern is
derived from the chemistry around the click, never from an atom number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from rdkit import Chem

from . import warheads as wh


@dataclass
class SmartsResult:
    smarts: str
    matches: Tuple[int, ...]         # atoms in this molecule the pattern hits
    target: int
    strategy: str                    # warhead | environment-r{N} | element
    radius: Optional[int] = None
    unique: bool = False
    hits_target: bool = True
    alternatives: List[Tuple[str, str, int]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def n_matches(self) -> int:
        return len(self.matches)

    @property
    def quoted(self) -> str:
        return f"'{self.smarts}'"

    def summary(self) -> str:
        if self.unique:
            return "Matches only the selected atom."
        return (
            f"Matches {self.n_matches} atoms in this ligand — gnina will try "
            f"a covalent bond at each of them."
        )


def _fragment_smarts(mol: Chem.Mol, atoms: Sequence[int], root: int,
                     explicit_h: bool = True) -> Optional[str]:
    """Rooted SMILES for a subset of atoms, used as the body of [$(...)]."""
    atom_set = sorted(set(atoms))
    if root not in atom_set:
        return None
    bonds = []
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set:
            bonds.append(bond.GetIdx())
    try:
        frag = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=atom_set,
            bondsToUse=bonds or None,
            rootedAtAtom=root,
            canonical=False,
            isomericSmiles=False,
            allHsExplicit=explicit_h,
            allBondsExplicit=False,
            kekuleSmiles=False,
        )
    except Exception:                                        # noqa: BLE001
        return None
    return frag or None


def _environment_atoms(mol: Chem.Mol, idx: int, radius: int) -> List[int]:
    if radius <= 0:
        return [idx]
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, idx, useHs=False)
    atoms = {idx}
    for bidx in env:
        bond = mol.GetBondWithIdx(bidx)
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())
    return sorted(atoms)


def _match_atoms(mol: Chem.Mol, smarts: str) -> Optional[Tuple[int, ...]]:
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return None
    return tuple(m[0] for m in mol.GetSubstructMatches(patt, uniquify=True))


def _wrap(fragment: str) -> str:
    return f"[$({fragment})]"


_TAG = "_smarts_src_idx"


def _collapse_hydrogens(
    mol: Chem.Mol,
    atom_idx: int,
    match_atoms: Optional[Sequence[int]] = None,
) -> Tuple[Chem.Mol, int, Optional[List[int]]]:
    """
    Work on a hydrogen-suppressed copy, with indices remapped.

    A 3-D ligand carries its hydrogens as real atoms, so every heavy atom
    reports an implicit-H count of zero and `allHsExplicit` writes a bare
    `[C]` instead of `[CH3]`. Removing the hydrogens first puts the counts
    back where the SMARTS writer can see them, and the H atoms are never
    part of the pattern anyway.
    """
    work = Chem.Mol(mol)
    if not any(a.GetAtomicNum() == 1 for a in work.GetAtoms()):
        return work, atom_idx, (list(match_atoms) if match_atoms else None)

    for atom in work.GetAtoms():
        atom.SetIntProp(_TAG, atom.GetIdx())
    try:
        stripped = Chem.RemoveHs(work, sanitize=False)
        stripped.UpdatePropertyCache(strict=False)
    except Exception:                                        # noqa: BLE001
        return work, atom_idx, (list(match_atoms) if match_atoms else None)

    remap = {a.GetIntProp(_TAG): a.GetIdx()
             for a in stripped.GetAtoms() if a.HasProp(_TAG)}
    if atom_idx not in remap:            # the reactive atom was a hydrogen
        return work, atom_idx, (list(match_atoms) if match_atoms else None)

    for atom in stripped.GetAtoms():
        if atom.HasProp(_TAG):
            atom.ClearProp(_TAG)
    new_match = ([remap[i] for i in match_atoms if i in remap]
                 if match_atoms else None)
    return stripped, remap[atom_idx], (new_match or None)


def generate_smarts(
    mol: Chem.Mol,
    atom_idx: int,
    warhead_key: Optional[str] = None,
    match_atoms: Optional[Sequence[int]] = None,
    min_radius: int = 3,
    max_radius: int = 6,
    explicit_h: bool = True,
) -> SmartsResult:
    """
    Build a pattern for `atom_idx` in `mol`.

    `mol` must be the molecule that will actually be docked — for an
    irreversible warhead that is the bound form (see bound_form.py), because
    gnina matches the pattern against the ligand file it is given.

    `min_radius` keeps the pattern chemically meaningful. A two-bond
    environment can be unique in a small fragment while saying nothing about
    the chemistry; three bonds reaches the carbonyl of a saturated acrylamide,
    which is what makes the pattern recognisable and portable.
    """
    mol, atom_idx, match_atoms = _collapse_hydrogens(mol, atom_idx, match_atoms)
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL
                     ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE, catchErrors=True)
    attempts: List[Tuple[str, str, Optional[int], Tuple[int, ...]]] = []

    # 1. The warhead motif itself, rooted at the reactive atom.
    if match_atoms:
        frag = _fragment_smarts(mol, match_atoms, atom_idx, explicit_h=explicit_h)
        if frag:
            smarts = _wrap(frag)
            hits = _match_atoms(mol, smarts)
            if hits is not None:
                attempts.append(("warhead", smarts, None, hits))
    elif warhead_key and warhead_key in wh.WARHEADS:
        w = wh.WARHEADS[warhead_key]
        patt = Chem.MolFromSmarts(w.smarts)
        if patt is not None:
            for m in mol.GetSubstructMatches(patt, uniquify=True):
                if w.reactive_index < len(m) and m[w.reactive_index] == atom_idx:
                    frag = _fragment_smarts(mol, m, atom_idx, explicit_h=explicit_h)
                    if frag:
                        smarts = _wrap(frag)
                        hits = _match_atoms(mol, smarts)
                        if hits is not None:
                            attempts.append(("warhead", smarts, None, hits))
                    break

    # 2. Grow the local environment until the pattern is unique.
    seen_smarts = set()
    for radius in range(1, max_radius + 1):
        atoms = _environment_atoms(mol, atom_idx, radius)
        frag = _fragment_smarts(mol, atoms, atom_idx, explicit_h=explicit_h)
        if not frag:
            continue
        smarts = _wrap(frag)
        if smarts in seen_smarts:
            continue
        seen_smarts.add(smarts)
        hits = _match_atoms(mol, smarts)
        if hits is None:
            continue
        attempts.append((f"environment-r{radius}", smarts, radius, hits))
        if radius >= min_radius and len(hits) == 1 and hits[0] == atom_idx:
            break

    # 3. Last resort: the atom's own query features.
    atom = mol.GetAtomWithIdx(atom_idx)
    fallback = (
        f"[#{atom.GetAtomicNum()};H{atom.GetTotalNumHs()};"
        f"D{atom.GetDegree()};{'a' if atom.GetIsAromatic() else 'A'}]"
    )
    hits = _match_atoms(mol, fallback) or ()
    attempts.append(("element", fallback, None, hits))

    # Prefer, in order: patterns that hit the selected atom; patterns that hit
    # only it; patterns carrying at least `min_radius` of chemical context;
    # then the smallest such environment.
    def score(item):
        strategy, _smarts, radius, hits = item
        unique = len(hits) == 1 and hits and hits[0] == atom_idx
        # The bare-element query carries no chemistry, so it never counts as
        # enough context however few atoms it happens to match.
        enough_context = (strategy == "warhead"
                          or (radius is not None and radius >= min_radius))
        strategy_rank = {"warhead": 0, "element": 2}.get(strategy, 1)
        return (
            0 if atom_idx in hits else 1,
            0 if unique else 1,
            0 if enough_context else 1,
            strategy_rank,
            len(hits) if hits else 999,
            radius if radius is not None else 0,
        )

    attempts.sort(key=score)
    best_strategy, best_smarts, best_radius, best_hits = attempts[0]

    result = SmartsResult(
        smarts=best_smarts,
        matches=best_hits,
        target=atom_idx,
        strategy=best_strategy,
        radius=best_radius,
        unique=(len(best_hits) == 1 and best_hits[0] == atom_idx),
        hits_target=atom_idx in best_hits,
        alternatives=[(s, sm, len(h)) for s, sm, _r, h in attempts[1:5]],
    )

    if not result.hits_target:
        result.warnings.append(
            "The generated pattern does not match the selected atom. "
            "Check the ligand structure, or set the pattern manually."
        )
    elif not result.unique:
        others = [i for i in best_hits if i != atom_idx]
        result.warnings.append(
            "The pattern also matches atom index "
            + ", ".join(str(i) for i in others[:6])
            + ". gnina will evaluate a covalent bond at each match."
        )
    return result


def verify_smarts(mol: Chem.Mol, smarts: str, atom_idx: Optional[int] = None) -> dict:
    """Check a hand-written pattern before it goes on the command line."""
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return {"valid": False, "error": "SMARTS does not compile.", "matches": []}
    matches = [m[0] for m in mol.GetSubstructMatches(patt, uniquify=True)]
    info = {
        "valid": True,
        "matches": matches,
        "n_matches": len(matches),
        "first_atom_is_query_root": True,
        "hits_target": (atom_idx in matches) if atom_idx is not None else None,
    }
    if not matches:
        info["error"] = "Valid SMARTS, but it matches nothing in this ligand."
    return info
