"""
compatibility.py — which protein nucleophile can plausibly attack which warhead.

This is a candidate-generation filter, used to shorten the list the user has to
look at. A "compatible" verdict means the pairing is chemically sensible, not
that the reaction happens in this protein: that depends on geometry,
accessibility, protonation state and electronic activation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .ligand_reactivity import Electrophile
from .protein_reactivity import Nucleophile
from .warheads import (AMINE, CARBOXYLATE, HYDROXYL, IMIDAZOLE, SELENOL, THIOL)

STRONG, PLAUSIBLE, WEAK, INCOMPATIBLE = "strong", "plausible", "weak", "incompatible"

STRENGTH_ORDER = {STRONG: 0, PLAUSIBLE: 1, WEAK: 2, INCOMPATIBLE: 3}

# (nucleophile type, mechanism) -> strength
_MATRIX: Dict[Tuple[str, str], str] = {}


def _fill(nu: str, table: Dict[str, str]) -> None:
    for mech, strength in table.items():
        _MATRIX[(nu, mech)] = strength


_fill(THIOL, {
    "phosphorylation": PLAUSIBLE,
    "michael_addition": STRONG,
    "sn2_alkylation": STRONG,
    "ring_opening": STRONG,
    "snar": STRONG,
    "disulfide_exchange": STRONG,
    "reversible_addition": STRONG,
    "addition": STRONG,
    "acylation": PLAUSIBLE,
    "sulfonylation": PLAUSIBLE,
})
_fill(SELENOL, {
    "phosphorylation": PLAUSIBLE,
    "michael_addition": STRONG,
    "sn2_alkylation": STRONG,
    "ring_opening": STRONG,
    "snar": STRONG,
    "disulfide_exchange": STRONG,
    "reversible_addition": STRONG,
    "addition": STRONG,
    "acylation": PLAUSIBLE,
    "sulfonylation": PLAUSIBLE,
})
_fill(AMINE, {
    "phosphorylation": PLAUSIBLE,
    "acylation": STRONG,
    "addition": STRONG,
    "sulfonylation": PLAUSIBLE,
    "michael_addition": PLAUSIBLE,
    "sn2_alkylation": PLAUSIBLE,
    "ring_opening": PLAUSIBLE,
    "snar": PLAUSIBLE,
    "reversible_addition": WEAK,
    "disulfide_exchange": INCOMPATIBLE,
})
_fill(HYDROXYL, {
    "phosphorylation": STRONG,
    "acylation": STRONG,
    "sulfonylation": STRONG,
    "reversible_addition": STRONG,
    "ring_opening": PLAUSIBLE,
    "addition": PLAUSIBLE,
    "michael_addition": WEAK,
    "sn2_alkylation": WEAK,
    "snar": WEAK,
    "disulfide_exchange": INCOMPATIBLE,
})
_fill(IMIDAZOLE, {
    "phosphorylation": PLAUSIBLE,
    "acylation": STRONG,
    "sulfonylation": STRONG,
    "sn2_alkylation": PLAUSIBLE,
    "michael_addition": PLAUSIBLE,
    "ring_opening": PLAUSIBLE,
    "addition": PLAUSIBLE,
    "snar": WEAK,
    "reversible_addition": WEAK,
    "disulfide_exchange": INCOMPATIBLE,
})
_fill(CARBOXYLATE, {
    "phosphorylation": WEAK,
    "ring_opening": PLAUSIBLE,
    "sn2_alkylation": PLAUSIBLE,
    "acylation": WEAK,
    "michael_addition": WEAK,
    "snar": WEAK,
    "sulfonylation": WEAK,
    "addition": WEAK,
    "reversible_addition": INCOMPATIBLE,
    "disulfide_exchange": INCOMPATIBLE,
})

BOND_SYMBOL = {
    "S": "S–C", "SE": "Se–C", "N": "N–C", "O": "O–C",
}

_CONTEXT_NOTES = {
    (HYDROXYL, "acylation"):
        "Serine/threonine acylation usually needs a catalytic triad to activate the hydroxyl.",
    (CARBOXYLATE, "ring_opening"):
        "Carboxylate attack on epoxides is known for glycosidase inactivators but is rare elsewhere.",
    (AMINE, "michael_addition"):
        "Lysine conjugate addition needs a lowered pKa; it is slower than thiol addition.",
    (THIOL, "acylation"):
        "Thioester adducts are formed but can hydrolyse or transfer.",
}


@dataclass
class Verdict:
    compatible: bool
    strength: str
    bond: str
    bond_order: int
    reason: str
    caveats: List[str]

    @property
    def badge(self) -> str:
        return {STRONG: "Strong match", PLAUSIBLE: "Plausible",
                WEAK: "Weak", INCOMPATIBLE: "Not compatible"}[self.strength]


def bond_label(nucleophile: Nucleophile, electrophile: Electrophile) -> str:
    left = nucleophile.element.capitalize()
    right = electrophile.element.capitalize()
    return f"{left}–{right}"


def check(nucleophile: Nucleophile, electrophile: Electrophile) -> Verdict:
    """Is this protein atom / ligand atom pair worth docking?"""
    nu_type = nucleophile.nucleophile_type
    mech = electrophile.mechanism

    listed = nu_type in electrophile.nucleophiles
    strength = _MATRIX.get((nu_type, mech), WEAK if listed else INCOMPATIBLE)
    if not listed and strength in (STRONG, PLAUSIBLE):
        strength = WEAK

    caveats: List[str] = []
    if nucleophile.disulfide_partner:
        caveats.append(
            f"This cysteine is in a disulfide with {nucleophile.disulfide_partner} "
            "and is unlikely to be free to react."
        )
        if strength == STRONG:
            strength = PLAUSIBLE
    if nucleophile.neighbor_count > 60:
        caveats.append("The atom is deeply buried; check that the ligand can reach it.")
    if electrophile.confidence == "low":
        caveats.append(
            f"{electrophile.warhead_label} is a common, weakly activated group — "
            "confirm it is really the intended warhead."
        )
    if electrophile.reversible:
        caveats.append("This warhead forms a reversible adduct.")
    note = _CONTEXT_NOTES.get((nu_type, mech))
    if note:
        caveats.append(note)

    reason = (
        f"{nucleophile.description} attacking the "
        f"{electrophile.role_label} of a {electrophile.warhead_label.lower()} "
        f"({electrophile.mechanism_label.lower()})"
    )

    return Verdict(
        compatible=strength != INCOMPATIBLE,
        strength=strength,
        bond=bond_label(nucleophile, electrophile),
        bond_order=electrophile.bond_order,
        reason=reason,
        caveats=caveats,
    )


def filter_for(
    nucleophile: Nucleophile,
    electrophiles: Sequence[Electrophile],
    min_strength: str = WEAK,
) -> List[Tuple[Electrophile, Verdict]]:
    """Electrophiles worth showing for this protein atom, best first."""
    limit = STRENGTH_ORDER[min_strength]
    rows = []
    for e in electrophiles:
        v = check(nucleophile, e)
        if v.compatible and STRENGTH_ORDER[v.strength] <= limit:
            rows.append((e, v))
    rows.sort(key=lambda t: (STRENGTH_ORDER[t[1].strength],
                             {"high": 0, "medium": 1, "low": 2}[t[0].confidence],
                             t[0].atom_idx))
    return rows


def nucleophiles_for(electrophile: Electrophile,
                     nucleophiles: Sequence[Nucleophile],
                     min_strength: str = WEAK) -> List[Tuple[Nucleophile, Verdict]]:
    """The reverse filter: protein atoms worth showing for a chosen warhead."""
    limit = STRENGTH_ORDER[min_strength]
    rows = []
    for n in nucleophiles:
        v = check(n, electrophile)
        if v.compatible and STRENGTH_ORDER[v.strength] <= limit:
            rows.append((n, v))
    rows.sort(key=lambda t: (STRENGTH_ORDER[t[1].strength], t[0].neighbor_count))
    return rows
