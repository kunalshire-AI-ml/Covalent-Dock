"""
attachment.py — rank every ligand atom as a covalent attachment point.

`find_electrophiles` answers "which atoms are electrophiles". This answers the
question a user actually has when overriding it: *of all the atoms, which one
should I bond?*

The two are not the same question. A ligand taken straight out of a crystal
structure is already in its reacted form — the attachment carbon of a 2-acyl
thiazole, say, arrives as an sp3 carbinol — so it is no longer an electrophile
and no warhead pattern can match it. The atom is still the right one to bond;
it just cannot be found by looking for electrophiles. Everything here is a
ranking to guide a manual choice, never a claim that a reaction occurs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from rdkit import Chem

from .compatibility import STRENGTH_ORDER, Verdict, check
from .ligand_reactivity import Electrophile, atom_label
from .protein_reactivity import Nucleophile

STRONG, PLAUSIBLE, WEAK, UNSUITABLE = "strong", "plausible", "weak", "unsuitable"

BAND_ORDER = {STRONG: 0, PLAUSIBLE: 1, WEAK: 2, UNSUITABLE: 3}

# Score a detected warhead atom from how well its mechanism suits the chosen
# nucleophile. These are the compatibility verdicts, mapped onto the same 0-100
# scale as the heuristics below so one ranking can hold both.
_VERDICT_SCORE = {"strong": 90, "plausible": 72, "weak": 48, "incompatible": 22}
_CONFIDENCE_ADJUST = {"high": 5, "medium": 0, "low": -12}

HALOGENS = {"F", "Cl", "Br", "I"}


@dataclass
class AtomAssessment:
    """How suitable one ligand atom is as the covalent attachment point."""
    atom_idx: int
    label: str
    element: str
    score: int                                  # 0-100
    band: str                                   # strong | plausible | weak | unsuitable
    headline: str
    reasons: List[str] = field(default_factory=list)
    electrophile: Optional[Electrophile] = None
    verdict: Optional[Verdict] = None

    @property
    def detected(self) -> bool:
        """True when a warhead pattern matched this atom."""
        return self.electrophile is not None

    @property
    def badge(self) -> str:
        return {STRONG: "Strong", PLAUSIBLE: "Plausible",
                WEAK: "Weak", UNSUITABLE: "Unsuitable"}[self.band]

    def describe(self) -> str:
        return f"{self.label} · {self.score}/100 · {self.headline}"


def _band(score: int) -> str:
    """
    'Unsuitable' is reserved for atoms that physically cannot carry the bond —
    a hydrogen, a halogen leaving group, an atom with no free valence. An
    ordinary unactivated carbon is 'weak': nothing recommends it, but a user
    who knows something the library does not is not being told it is impossible.
    """
    if score >= 75:
        return STRONG
    if score >= 50:
        return PLAUSIBLE
    if score >= 20:
        return WEAK
    return UNSUITABLE


def _has_capacity(atom: Chem.Atom) -> bool:
    """
    Can this atom take one more bond?

    The protein bond either displaces a hydrogen or fills the valence freed by
    the bound-form transform. An atom with neither — a quaternary carbon, say —
    would come out hypervalent.
    """
    if atom.GetTotalNumHs() > 0:
        return True
    return any(b.GetBondTypeAsDouble() > 1.0 for b in atom.GetBonds())


def _neighbour_flags(mol: Chem.Mol, atom: Chem.Atom) -> dict:
    flags = {"hydroxyl": False, "carbonyl": False, "aromatic": False,
             "halogen": False, "withdrawing": False}
    for nb in atom.GetNeighbors():
        sym = nb.GetSymbol()
        if sym in HALOGENS:
            flags["halogen"] = True
        if nb.GetIsAromatic():
            flags["aromatic"] = True
            flags["withdrawing"] = True
        if sym == "O" and nb.GetTotalNumHs() > 0 and \
                mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx()).GetBondTypeAsDouble() == 1.0:
            flags["hydroxyl"] = True
        # a carbonyl or nitrile hanging off the neighbour stabilises this centre
        for b in nb.GetBonds():
            other = b.GetOtherAtom(nb)
            if other.GetIdx() == atom.GetIdx():
                continue
            if b.GetBondTypeAsDouble() >= 2.0 and other.GetSymbol() in ("O", "N"):
                flags["withdrawing"] = True
        if sum(1 for n2 in nb.GetNeighbors() if n2.GetSymbol() in HALOGENS) >= 2:
            flags["withdrawing"] = True          # CF3 and friends
    for b in atom.GetBonds():
        other = b.GetOtherAtom(atom)
        if b.GetBondTypeAsDouble() >= 2.0 and other.GetSymbol() == "O":
            flags["carbonyl"] = True
    return flags


def _assess_undetected(mol: Chem.Mol, atom: Chem.Atom) -> tuple:
    """(score, headline, reasons) for an atom no warhead pattern matched."""
    sym = atom.GetSymbol()
    flags = _neighbour_flags(mol, atom)
    reasons: List[str] = []

    if sym in ("S", "Se"):
        if atom.GetIsAromatic():
            # The sulfur of a thiophene or thiazole is a ring heteroatom with
            # its lone pairs in the aromatic system. It does not exchange.
            return (20, f"aromatic ring {sym}",
                    ["Part of the aromatic ring, not an exchangeable sulfur."])
        disulfide = any(nb.GetSymbol() in ("S", "Se") for nb in atom.GetNeighbors())
        if atom.GetTotalNumHs() > 0 or disulfide:
            return (62 if sym == "S" else 66,
                    f"{sym} — can exchange with a protein thiol",
                    ["Disulfide or selenosulfide exchange rather than attack "
                     "on carbon."])
        return (40, f"{sym} in a thioether",
                ["A thioether sulfur exchanges far less readily than a thiol "
                 "or a disulfide."])
    if sym == "P":
        return 60, "phosphorus — phosphorylation centre", []
    if sym in ("N", "O"):
        return (22, f"{sym} is normally the nucleophile, not the electrophile",
                ["Ligand nitrogen and oxygen rarely accept the protein atom; "
                 "check this is really the attachment point."])

    if sym != "C":
        return 18, f"{sym} — not a usual attachment element", []

    # ── carbon ───────────────────────────────────────────────────────────────
    if atom.GetIsAromatic():
        return (20, "aromatic carbon",
                ["Only reactive when strongly activated for SNAr — an ortho/para "
                 "nitro or an azine ring nitrogen with a leaving group."])

    if flags["carbonyl"]:
        # An amide or ester carbonyl is a far poorer electrophile than a ketone:
        # the nitrogen or oxygen lone pair delocalises into it. Not making this
        # distinction is what let a trifluoroacetamide cap be reported as a
        # trifluoromethyl ketone warhead, so it must not be repeated here.
        donors = [nb.GetSymbol() for nb in atom.GetNeighbors()
                  if nb.GetSymbol() in ("N", "O", "S")
                  and mol.GetBondBetweenAtoms(
                      atom.GetIdx(), nb.GetIdx()).GetBondTypeAsDouble() == 1.0]
        if donors:
            return (30, "amide or ester carbonyl",
                    ["A neighbouring nitrogen or oxygen feeds this carbonyl, "
                     "making it much less electrophilic than a ketone.",
                     "Acylation does happen here, but needs real activation."])
        return (56, "ketone or aldehyde carbonyl that no pattern matched",
                ["Open to reversible addition. No library pattern claimed it, so "
                 "confirm the activation yourself."])

    if flags["hydroxyl"] and atom.GetHybridization() == Chem.HybridizationType.SP3:
        # The product of a thiolate or hydroxide adding to a ketone is exactly
        # this: an sp3 carbon carrying a new hydroxyl, next to whatever group
        # stabilised the carbonyl. A ligand lifted out of a crystal structure
        # arrives in this state, and it is the single most useful thing to be
        # able to recognise when the automatic search has found nothing.
        if flags["withdrawing"] or flags["aromatic"]:
            return (78, "looks like an already-formed adduct",
                    ["An sp3 carbon bearing a hydroxyl next to an activating "
                     "group is what a ketone becomes after a nucleophile adds.",
                     "If this ligand came from a crystal structure it is already "
                     "the bound form — leave 'build the bound form' unticked."])
        return (52, "carbon bearing a hydroxyl",
                ["Could be a reacted carbonyl, or just an alcohol."])

    if flags["halogen"]:
        return (64, "carbon carrying a halogen leaving group",
                ["Direct displacement of the halide."])

    if any(b.GetBondTypeAsDouble() == 2.0 for b in atom.GetBonds()):
        return (46, "alkene carbon",
                ["Conjugate addition needs a withdrawing group on the far side."])

    if atom.GetTotalNumHs() == 0:
        return (10, "fully substituted carbon",
                ["No hydrogen to displace and no open valence."])

    return (24, "saturated carbon with nothing activating it",
            ["No leaving group, carbonyl or conjugation nearby."])


def assess_atom(
    mol: Chem.Mol,
    atom_idx: int,
    nucleophile: Optional[Nucleophile] = None,
    electrophiles: Sequence[Electrophile] = (),
) -> AtomAssessment:
    """Score one atom. `electrophiles` is the output of find_electrophiles."""
    atom = mol.GetAtomWithIdx(atom_idx)
    sym = atom.GetSymbol()
    label = atom_label(mol, atom_idx)
    found = next((e for e in electrophiles if e.atom_idx == atom_idx), None)

    if atom.GetAtomicNum() == 1:
        return AtomAssessment(atom_idx, label, sym, 0, UNSUITABLE,
                              "hydrogen", ["Hydrogens cannot carry the bond."])
    if sym in HALOGENS:
        return AtomAssessment(
            atom_idx, label, sym, 0, UNSUITABLE, "halogen",
            ["A halogen is the leaving group; bond the carbon it sits on."])

    if found is not None:
        verdict = check(nucleophile, found) if nucleophile is not None else None
        if verdict is not None:
            score = _VERDICT_SCORE[verdict.strength]
            headline = f"{found.warhead_label} · {verdict.badge.lower()}"
            reasons = [verdict.reason] + list(verdict.caveats)
        else:
            score = {"high": 80, "medium": 62, "low": 44}[found.confidence]
            headline = f"{found.warhead_label} (no protein atom chosen yet)"
            reasons = [found.note] if found.note else []
        score += _CONFIDENCE_ADJUST[found.confidence]
    else:
        score, headline, reasons = _assess_undetected(mol, atom)
        verdict = None

    if not _has_capacity(atom):
        score = min(score, 18)
        reasons.append("No hydrogen to displace and no multiple bond to open — "
                       "bonding here would leave the atom hypervalent.")

    score = max(0, min(100, score))
    return AtomAssessment(atom_idx, label, sym, score, _band(score),
                          headline, [r for r in reasons if r], found, verdict)


def rank_atoms(
    mol: Chem.Mol,
    nucleophile: Optional[Nucleophile] = None,
    electrophiles: Sequence[Electrophile] = (),
    include_unsuitable: bool = True,
) -> List[AtomAssessment]:
    """Every heavy atom, best candidate first."""
    out = [assess_atom(mol, a.GetIdx(), nucleophile, electrophiles)
           for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not include_unsuitable:
        out = [a for a in out if a.band != UNSUITABLE]
    out.sort(key=lambda a: (-a.score, a.atom_idx))
    return out


def manual_electrophile(mol: Chem.Mol, atom_idx: int,
                        bond_order: int = 1) -> Electrophile:
    """
    An Electrophile record for an atom the user chose themselves.

    `transform` is "none": a manually chosen atom is taken to be the ligand as
    it should be docked, which is what a structure-derived ligand already is.
    The rest of the pipeline only needs these fields to be present and honest
    about where they came from.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    return Electrophile(
        atom_idx=atom_idx,
        element=atom.GetSymbol(),
        label=atom_label(mol, atom_idx),
        warhead_key="manual",
        warhead_label="Manual selection",
        role="attachment_point",
        role_label="attachment point",
        mechanism="manual",
        mechanism_label="chosen by hand",
        nucleophiles=(),
        bond_order=bond_order,
        reversible=False,
        confidence="medium",
        match_atoms=(atom_idx,),
        leaving_atom=None,
        transform="none",
        note="Chosen by hand; the library did not propose this atom. The ligand "
             "is used as supplied, so it must already be in its bound form.",
    )
