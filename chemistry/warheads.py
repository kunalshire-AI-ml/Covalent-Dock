"""
warheads.py — the warhead rule database.

Every entry describes one electrophilic motif:

  smarts          SMARTS that matches the whole motif
  reactive_index  which atom of that match is the electrophilic centre
  leaving_index   which atom (if any) leaves when the bond forms
  role            human label for the reactive atom ("beta_carbon", ...)
  nucleophiles    protein nucleophile *types* that can plausibly attack it
  transform       how to build the covalently bound (adduct) form
                  -> see chemistry/bound_form.py

Priority matters: a chloroacetamide also matches the generic alkyl-halide
rule, so the more specific pattern must win. Lower `priority` = more specific.

Nothing here proves a reaction happens. These are candidate-generation rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Nucleophile type vocabulary (see chemistry/compatibility.py)
THIOL = "thiol"            # Cys SG
AMINE = "amine"            # Lys NZ, N-terminus
HYDROXYL = "hydroxyl"      # Ser OG, Thr OG1, Tyr OH
IMIDAZOLE = "imidazole"    # His ND1 / NE2
CARBOXYLATE = "carboxylate"  # Asp OD1/OD2, Glu OE1/OE2
SELENOL = "selenol"        # Sec SE

ALL_NU = (THIOL, AMINE, HYDROXYL, IMIDAZOLE, CARBOXYLATE, SELENOL)


@dataclass(frozen=True)
class Warhead:
    key: str
    label: str
    smarts: str
    reactive_index: int
    role: str
    mechanism: str
    nucleophiles: Tuple[str, ...]
    bond_order: int = 1
    leaving_index: Optional[int] = None
    transform: str = "none"
    reversible: bool = False
    priority: int = 50
    # A promiscuous pattern matches lots of unreactive chemistry (plain
    # nitriles, ketones...). The UI shows these in a separate, quieter group.
    promiscuous: bool = False
    note: str = ""
    aliases: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def family(self) -> str:
        return self.mechanism


# ─────────────────────────────────────────────────────────────────────────────
#  Michael acceptors  (conjugate addition — the β-carbon is attacked)
# ─────────────────────────────────────────────────────────────────────────────
_MICHAEL: List[Warhead] = [
    Warhead(
        key="acrylamide",
        label="Acrylamide",
        smarts="[CH2X3]=[CHX3][CX3](=[OX1])[NX3]",
        reactive_index=0,
        role="beta_carbon",
        mechanism="michael_addition",
        nucleophiles=(THIOL, SELENOL, AMINE, IMIDAZOLE),
        transform="saturate_alkene",
        priority=10,
        note="Workhorse covalent warhead (afatinib, ibrutinib, sotorasib).",
        aliases=("michael_acceptor",),
    ),
    Warhead(
        key="acrylate_ester",
        label="Acrylate ester",
        smarts="[CH2X3]=[CHX3][CX3](=[OX1])[OX2][#6]",
        reactive_index=0,
        role="beta_carbon",
        mechanism="michael_addition",
        nucleophiles=(THIOL, SELENOL, AMINE),
        transform="saturate_alkene",
        priority=12,
        aliases=("michael_acceptor",),
    ),
    Warhead(
        key="vinyl_ketone",
        label="Vinyl ketone",
        smarts="[CH2X3]=[CHX3][CX3](=[OX1])[#6]",
        reactive_index=0,
        role="beta_carbon",
        mechanism="michael_addition",
        nucleophiles=(THIOL, SELENOL, AMINE),
        transform="saturate_alkene",
        priority=12,
        note="Highly reactive; often too promiscuous for a drug candidate.",
        aliases=("michael_acceptor",),
    ),
    Warhead(
        key="vinyl_sulfone",
        label="Vinyl sulfone",
        smarts="[CH2X3]=[CHX3][SX4](=[OX1])(=[OX1])",
        reactive_index=0,
        role="beta_carbon",
        mechanism="michael_addition",
        nucleophiles=(THIOL, SELENOL, AMINE),
        transform="saturate_alkene",
        priority=12,
        aliases=("michael_acceptor",),
    ),
    Warhead(
        key="acrylonitrile",
        label="Acrylonitrile",
        smarts="[CH2X3]=[CHX3][CX2]#[NX1]",
        reactive_index=0,
        role="beta_carbon",
        mechanism="michael_addition",
        nucleophiles=(THIOL, SELENOL, AMINE),
        transform="saturate_alkene",
        priority=12,
        aliases=("michael_acceptor",),
    ),
    Warhead(
        key="maleimide",
        label="Maleimide",
        smarts="[CX3H1]1=[CX3H1][CX3](=[OX1])[NX3][CX3]1=[OX1]",
        reactive_index=0,
        role="alkene_carbon",
        mechanism="michael_addition",
        nucleophiles=(THIOL, SELENOL, AMINE),
        transform="saturate_alkene",
        priority=10,
        note="Fast and highly thiol-selective; standard bioconjugation warhead.",
        aliases=("michael_acceptor",),
    ),
    Warhead(
        key="propiolamide",
        label="Propiolamide / ynone",
        smarts="[CX2H1]#[CX2][CX3](=[OX1])[NX3,#6]",
        reactive_index=0,
        role="beta_carbon",
        mechanism="michael_addition",
        nucleophiles=(THIOL, SELENOL, AMINE),
        transform="reduce_alkyne",
        priority=12,
        note="Activated alkyne; addition gives a vinyl thioether.",
    ),
    Warhead(
        key="nitroalkene",
        label="Nitroalkene",
        smarts="[CX3]=[CX3][NX3+](=[OX1])[OX1-]",
        reactive_index=0,
        role="beta_carbon",
        mechanism="michael_addition",
        nucleophiles=(THIOL, SELENOL, AMINE),
        transform="saturate_alkene",
        priority=15,
    ),
    # Fallback: any electron-poor alkene not caught above.
    Warhead(
        key="michael_acceptor_generic",
        label="Michael acceptor (generic)",
        smarts=(
            "[CX3;!$(C=[O,N,S])]=[CX3]"
            "[$([CX3]=[OX1]),$([SX4](=[OX1])=[OX1]),$([CX2]#[NX1]),"
            "$([NX3+](=[OX1])[OX1-]),$([PX4]=[OX1])]"
        ),
        reactive_index=0,
        role="beta_carbon",
        mechanism="michael_addition",
        nucleophiles=(THIOL, SELENOL, AMINE),
        transform="saturate_alkene",
        priority=40,
        note="Alkene conjugated to an electron-withdrawing group.",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
#  Alkylating agents  (SN2 — leaving group departs)
# ─────────────────────────────────────────────────────────────────────────────
_ALKYLATORS: List[Warhead] = [
    Warhead(
        key="anomeric_fluorosugar",
        label="Anomeric fluorosugar (glycosidase)",
        # A mechanism-based glycosidase inactivator. The catalytic carboxylate
        # attacks the anomeric carbon and displaces its leaving group, giving a
        # glycosyl-enzyme ester; the fluorine next door withdraws enough
        # density to stall the hydrolysis that would normally clear it, so the
        # intermediate accumulates and can be crystallised.
        #
        # The ring fluorine is the whole of the specificity. An anomeric carbon
        # on its own is present in every sugar — glucose included, and CovPDB
        # is full of them as substrates — so a pattern without the fluorine
        # would claim the entire carbohydrate content of the database. With it,
        # glucose, methyl glucoside and ribose are all left alone.
        #
        # Matching starts at the fluorine but bonds through atom 2, the
        # anomeric carbon: the nucleophile attacks there, not at the C-F.
        smarts="[F][CX4;r6][CX4;r6]([OX2;r6])[F,OX2]",
        reactive_index=2,
        leaving_index=4,
        role="anomeric_carbon",
        mechanism="sn2_alkylation",
        nucleophiles=(CARBOXYLATE, HYDROXYL, THIOL),
        transform="drop_leaving_group",
        priority=19,
        note="Glycosidase inactivator; the catalytic Asp/Glu forms a "
             "glycosyl-enzyme ester at the anomeric carbon.",
        aliases=("fluorosugar", "2_deoxy_2_fluoro_glycoside"),
    ),
    Warhead(
        key="haloacetamide",
        label="Haloacetamide (chloro/bromo/iodo)",
        smarts="[CX4H2]([F,Cl,Br,I])[CX3](=[OX1])[NX3]",
        reactive_index=0,
        leaving_index=1,
        role="alpha_carbon",
        mechanism="sn2_alkylation",
        nucleophiles=(THIOL, SELENOL, AMINE, IMIDAZOLE, CARBOXYLATE),
        transform="drop_leaving_group",
        priority=10,
        note="Chloroacetamide is the classic irreversible Cys alkylator.",
    ),
    Warhead(
        key="alpha_halo_ketone",
        label="α-Halo ketone",
        smarts="[CX4H2]([F,Cl,Br,I])[CX3](=[OX1])[#6]",
        reactive_index=0,
        leaving_index=1,
        role="alpha_carbon",
        mechanism="sn2_alkylation",
        nucleophiles=(THIOL, SELENOL, AMINE, IMIDAZOLE, CARBOXYLATE),
        transform="drop_leaving_group",
        priority=12,
    ),
    Warhead(
        key="diazo_ketone",
        label="α-Diazo ketone",
        smarts="[CX3](=[OX1])[CX3H1]=[NX2+]=[NX1-]",
        reactive_index=2,          # the diazo carbon, not the carbonyl
        leaving_index=3,           # N2 departs
        role="diazo_carbon",
        mechanism="sn2_alkylation",
        nucleophiles=(THIOL, SELENOL, CARBOXYLATE, AMINE, IMIDAZOLE),
        transform="drop_leaving_group",
        priority=12,
        note="DON / azaserine type: loses N2 and alkylates the nucleophile. "
             "Classic glutamine-amidotransferase inactivator.",
    ),
    Warhead(
        key="alkyl_halide",
        label="Alkyl halide",
        smarts="[CX4;H1,H2]([Cl,Br,I])",
        reactive_index=0,
        leaving_index=1,
        role="alkylation_carbon",
        mechanism="sn2_alkylation",
        nucleophiles=(THIOL, SELENOL, AMINE),
        transform="drop_leaving_group",
        priority=60,
        promiscuous=True,
        note="Weakly activated unless adjacent to an EWG.",
    ),
    Warhead(
        key="sulfonate_ester",
        label="Sulfonate ester (mesylate/tosylate)",
        smarts="[CX4][OX2][SX4](=[OX1])(=[OX1])[#6]",
        reactive_index=0,
        leaving_index=1,
        role="alkylation_carbon",
        mechanism="sn2_alkylation",
        nucleophiles=(THIOL, SELENOL, AMINE),
        transform="drop_leaving_group",
        priority=30,
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
#  Strained rings  (ring opening)
# ─────────────────────────────────────────────────────────────────────────────
_STRAINED: List[Warhead] = [
    Warhead(
        key="epoxide",
        label="Epoxide",
        smarts="[CX4;r3]1[OX2;r3][CX4;r3]1",
        reactive_index=0,
        leaving_index=1,
        role="epoxide_carbon",
        mechanism="ring_opening",
        nucleophiles=(THIOL, SELENOL, AMINE, IMIDAZOLE, CARBOXYLATE),
        transform="open_strained_ring",
        priority=10,
        note="Attack is usually at the less hindered carbon; both are offered.",
    ),
    Warhead(
        key="aziridine",
        label="Aziridine",
        smarts="[CX4;r3]1[NX3;r3][CX4;r3]1",
        reactive_index=0,
        leaving_index=1,
        role="aziridine_carbon",
        mechanism="ring_opening",
        nucleophiles=(THIOL, SELENOL, AMINE, CARBOXYLATE),
        transform="open_strained_ring",
        priority=10,
    ),
    Warhead(
        key="beta_lactam",
        label="β-Lactam",
        smarts="[CX3;r4]1(=[OX1])[NX3;r4][CX4;r4][CX4;r4]1",
        reactive_index=0,
        leaving_index=2,
        role="carbonyl_carbon",
        mechanism="acylation",
        nucleophiles=(HYDROXYL, THIOL, AMINE),
        transform="open_strained_ring",
        priority=10,
        note="Acylates the catalytic Ser of transpeptidases/β-lactamases.",
    ),
    Warhead(
        key="lactone",
        label="Lactone (5-/6-membered)",
        # The leading [#6] is what makes this a lactone rather than anything
        # with a ring C(=O)-O. Without it the pattern also matches cyclic
        # carbamates and carbonates, where a nitrogen or a second oxygen on
        # the acyl side kills the reactivity: it called linezolid's
        # oxazolidinone a warhead, and that is a stable scaffold.
        smarts="[#6][CX3;r5,r6](=[OX1])[OX2;r5,r6]",
        reactive_index=1,          # the carbonyl carbon, now atom 1 not 0
        leaving_index=3,           # the ring oxygen, now atom 3 not 2
        role="carbonyl_carbon",
        mechanism="acylation",
        nucleophiles=(HYDROXYL, THIOL, SELENOL, AMINE),
        transform="open_strained_ring",
        # beta_lactone (r4) is the strained, far more reactive case and keeps
        # its own higher priority; this covers the larger rings.
        priority=24,
        note="Acylates a catalytic serine by ring opening (statine/orlistat "
             "type). Less reactive than the strained β-lactone.",
    ),
    Warhead(
        key="diazabicyclooctanone",
        label="Diazabicyclooctanone (avibactam-type)",
        # The bicyclic urea of avibactam / relebactam / durlobactam. The ring
        # nitrogen carries an O-sulfate, which is what makes the neighbouring
        # carbonyl electrophilic enough to carbamoylate a catalytic serine.
        smarts="[NX3]C(=[OX1])[NX3][OX2][SX4](=[OX1])(=[OX1])",
        reactive_index=1,          # the carbonyl carbon
        leaving_index=3,           # the N-O-sulfate nitrogen; the ring opens here
        role="carbonyl_carbon",
        mechanism="acylation",
        nucleophiles=(HYDROXYL, THIOL, AMINE),
        transform="open_strained_ring",
        priority=10,
        note="Non-β-lactam β-lactamase inhibitor; carbamoylates the catalytic "
             "Ser by opening the C–N bond of the cyclic urea.",
    ),
    Warhead(
        key="beta_lactone",
        label="β-Lactone",
        smarts="[CX3;r4]1(=[OX1])[OX2;r4][CX4;r4][CX4;r4]1",
        reactive_index=0,
        leaving_index=2,
        role="carbonyl_carbon",
        mechanism="acylation",
        nucleophiles=(HYDROXYL, THIOL, AMINE),
        transform="open_strained_ring",
        priority=10,
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
#  Acylating / sulfonylating / heteroatom-transfer
# ─────────────────────────────────────────────────────────────────────────────
_ACYLATORS: List[Warhead] = [
    Warhead(
        key="sulfonyl_fluoride",
        label="Sulfonyl fluoride",
        smarts="[SX4](=[OX1])(=[OX1])[F]",
        reactive_index=0,
        leaving_index=3,
        role="sulfur_center",
        mechanism="sulfonylation",
        nucleophiles=(HYDROXYL, IMIDAZOLE, AMINE, THIOL),
        transform="drop_leaving_group",
        priority=10,
        note="SuFEx probe; favours Tyr-OH, Ser-OG and His.",
    ),
    Warhead(
        key="fluorosulfate",
        label="Aryl fluorosulfate",
        smarts="[OX2][SX4](=[OX1])(=[OX1])[F]",
        reactive_index=1,
        leaving_index=4,
        role="sulfur_center",
        mechanism="sulfonylation",
        nucleophiles=(HYDROXYL, IMIDAZOLE, AMINE),
        transform="drop_leaving_group",
        priority=10,
    ),
    Warhead(
        key="nhs_ester",
        label="Activated ester (NHS)",
        smarts="[CX3](=[OX1])[OX2][NX3]1[CX3](=[OX1])[CX4][CX4][CX3]1=[OX1]",
        reactive_index=0,
        leaving_index=2,
        role="carbonyl_carbon",
        mechanism="acylation",
        nucleophiles=(AMINE, HYDROXYL, THIOL),
        transform="drop_leaving_group",
        priority=10,
    ),
    Warhead(
        key="acyl_halide",
        label="Acyl halide",
        smarts="[CX3](=[OX1])[F,Cl,Br,I]",
        reactive_index=0,
        leaving_index=2,
        role="carbonyl_carbon",
        mechanism="acylation",
        nucleophiles=(AMINE, HYDROXYL, THIOL, IMIDAZOLE),
        transform="drop_leaving_group",
        priority=10,
    ),
    Warhead(
        key="carbamate_activated",
        label="Activated carbamate",
        smarts="[NX3][CX3](=[OX1])[OX2][c]",
        reactive_index=1,
        leaving_index=3,
        role="carbonyl_carbon",
        mechanism="acylation",
        nucleophiles=(HYDROXYL, THIOL, AMINE),
        transform="drop_leaving_group",
        priority=20,
        note="Carbamoylates catalytic Ser (rivastigmine-like).",
    ),
    Warhead(
        key="aryl_ester",
        label="Aryl ester (activated)",
        smarts="[CX3](=[OX1])[OX2][c]",
        reactive_index=0,
        leaving_index=2,
        role="carbonyl_carbon",
        mechanism="acylation",
        nucleophiles=(HYDROXYL, THIOL, SELENOL, AMINE, IMIDAZOLE),
        transform="drop_leaving_group",
        # Below carbamate_activated (20), which is the more specific rule for
        # an aryl ester that also carries a nitrogen on the carbonyl.
        priority=22,
        note="The aryloxide leaving group makes this an acylating agent; a "
             "plain alkyl ester is not and is deliberately not matched.",
    ),
    Warhead(
        key="isothiocyanate",
        label="Isothiocyanate",
        smarts="[NX2]=[CX2]=[SX1]",
        reactive_index=1,
        role="central_carbon",
        mechanism="addition",
        nucleophiles=(THIOL, AMINE, HYDROXYL),
        transform="reduce_cumulene",
        reversible=True,
        priority=10,
        note="Thiol adducts are reversible; amine adducts are stable thioureas.",
    ),
    Warhead(
        key="isocyanate",
        label="Isocyanate",
        smarts="[NX2]=[CX2]=[OX1]",
        reactive_index=1,
        role="central_carbon",
        mechanism="addition",
        nucleophiles=(AMINE, HYDROXYL, THIOL),
        transform="reduce_cumulene",
        priority=10,
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
#  Phosphorylation — the nucleophile attacks phosphorus, a leaving group goes
#
#  Only phosphorus carrying a genuine leaving group belongs here. A plain
#  phosphate ester or a free phosphate is not an electrophile at P — it is a
#  substituent — and matching those would nominate the phosphate of every
#  sugar phosphate and nucleotide as a warhead.
# ─────────────────────────────────────────────────────────────────────────────
_PHOSPHORUS: List[Warhead] = [
    Warhead(
        key="phosphoryl_fluoride",
        label="Phosphoryl fluoride",
        smarts="[PX4](=[OX1])[F]",
        reactive_index=0,
        leaving_index=2,
        role="phosphorus_center",
        mechanism="phosphorylation",
        nucleophiles=(HYDROXYL, SELENOL, THIOL, IMIDAZOLE, AMINE),
        transform="drop_leaving_group",
        priority=10,
        note="DFP / sarin-type. Phosphorylates the catalytic Ser of serine "
             "hydrolases; the fluoride is the leaving group.",
    ),
    Warhead(
        key="phosphoryl_aryl_ester",
        label="Aryl phosphate / phosphonate ester",
        smarts="[PX4](=[OX1])[OX2][c]",
        reactive_index=0,
        leaving_index=2,
        role="phosphorus_center",
        mechanism="phosphorylation",
        nucleophiles=(HYDROXYL, SELENOL, THIOL, IMIDAZOLE, AMINE),
        transform="drop_leaving_group",
        priority=12,
        note="Paraoxon-type: the aryloxide is a good enough leaving group to "
             "phosphorylate a catalytic serine. An alkyl ester is not.",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
#  Aromatic substitution (SNAr) — activated aryl/heteroaryl halides
# ─────────────────────────────────────────────────────────────────────────────
_SNAR: List[Warhead] = [
    Warhead(
        key="snar_heteroaryl_halide",
        label="Heteroaryl halide (SNAr)",
        smarts="[c;$(c[F,Cl,Br]);$(c:n)]",
        reactive_index=0,
        role="ipso_carbon",
        mechanism="snar",
        nucleophiles=(THIOL, SELENOL, AMINE, HYDROXYL),
        transform="drop_leaving_group",
        priority=20,
        note="2-halo-pyridine/pyrimidine type; activated by the ring nitrogen.",
    ),
    Warhead(
        key="snar_nitroaryl_halide",
        label="Nitroaryl halide (SNAr)",
        smarts=(
            "[c;$(c[F,Cl,Br]);"
            "$(c1ccc([NX3+](=[OX1])[OX1-])cc1),"
            "$(c1cc([NX3+](=[OX1])[OX1-])ccc1)]"
        ),
        reactive_index=0,
        role="ipso_carbon",
        mechanism="snar",
        nucleophiles=(THIOL, SELENOL, AMINE, HYDROXYL),
        transform="drop_leaving_group",
        priority=20,
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
#  Reversible covalent warheads
# ─────────────────────────────────────────────────────────────────────────────
_REVERSIBLE: List[Warhead] = [
    Warhead(
        key="nitrile",
        label="Nitrile",
        smarts="[CX2]#[NX1]",
        reactive_index=0,
        role="nitrile_carbon",
        mechanism="reversible_addition",
        nucleophiles=(THIOL, SELENOL, HYDROXYL),
        transform="reduce_nitrile",
        reversible=True,
        priority=35,
        promiscuous=True,
        note="Reversible thioimidate with Cys (saxagliptin, odanacatib).",
    ),
    Warhead(
        key="cyanamide",
        label="Cyanamide",
        smarts="[NX3][CX2]#[NX1]",
        reactive_index=1,
        role="nitrile_carbon",
        mechanism="reversible_addition",
        nucleophiles=(THIOL, SELENOL),
        transform="reduce_nitrile",
        reversible=True,
        priority=15,
    ),
    Warhead(
        key="aldehyde",
        label="Aldehyde",
        smarts="[CX3H1](=[OX1])[#6]",
        reactive_index=0,
        role="carbonyl_carbon",
        mechanism="reversible_addition",
        nucleophiles=(THIOL, SELENOL, AMINE, HYDROXYL),
        transform="reduce_carbonyl",
        reversible=True,
        priority=25,
        promiscuous=True,
        note="Hemithioacetal / hemiacetal adduct.",
    ),
    Warhead(
        key="hydroxymethyl_ketone",
        label="α-Hydroxy ketone",
        # [CH2] only caught the unsubstituted case; the peptidomimetic form
        # carries a substituent on that carbon, which is the common one.
        smarts="[CX3](=[OX1])[CX4;H1,H2][OX2H1]",
        reactive_index=0,
        role="carbonyl_carbon",
        mechanism="reversible_addition",
        nucleophiles=(THIOL, SELENOL, AMINE, HYDROXYL),
        transform="reduce_carbonyl",
        reversible=True,
        priority=18,
        note="Cysteine-protease warhead; the thiolate adds to the ketone to "
             "give a hemithioketal.",
    ),
    # A generic ketone — [#6][CX3](=[OX1])[#6], 253 sites in CovPDB — was tried
    # and rejected. `promiscuous` does not contain it, because filter_for ranks
    # by strength before confidence: for a serine, hydroxyl+reversible_addition
    # is STRONG while hydroxyl+ring_opening is only PLAUSIBLE, so a plain ketone
    # anywhere in the molecule outranks the epoxide that is the real warhead. It
    # displaced 446 correct picks, 144 of them epoxides and 120 α-halo ketones.
    # Adding it needs the compatibility matrix reworked first, not another flag.
    Warhead(
        key="alkyne_generic",
        label="Alkyne (generic)",
        smarts="[CX2]#[CX2][#6]",
        reactive_index=0,
        role="beta_carbon",
        mechanism="michael_addition",
        nucleophiles=(THIOL, SELENOL, AMINE),
        transform="reduce_alkyne",
        # Unactivated alkynes are poor electrophiles; propiolamide above holds
        # the activated case.
        promiscuous=True,
        priority=58,
        note="Unactivated alkyne. Only reactive when an adjacent withdrawing "
             "group activates it, which propiolamide covers specifically.",
    ),
    Warhead(
        key="trifluoromethyl_ketone",
        label="Trifluoromethyl ketone",
        # The carbon on the far side of the carbonyl has to be pinned down.
        # Without it this also matched CF3-C(=O)-N, a trifluoroacetamide, which
        # is an amide: the nitrogen lone pair feeds the carbonyl and it is not
        # an electrophile at all. It is a routine N-terminal cap on
        # peptidomimetics, so the library was reporting the cap of a covalent
        # inhibitor as its warhead — at high confidence, and often as the only
        # candidate offered.
        smarts="[#6][CX3](=[OX1])[CX4]([F])([F])[F]",
        reactive_index=1,
        role="carbonyl_carbon",
        mechanism="reversible_addition",
        nucleophiles=(HYDROXYL, THIOL, AMINE),
        transform="reduce_carbonyl",
        reversible=True,
        priority=15,
    ),
    Warhead(
        key="bisulfite_aldehyde",
        label="Bisulfite-masked aldehyde",
        # An alpha-hydroxysulfonate: the bisulfite adduct of an aldehyde, used
        # to ship a reactive aldehyde warhead as a stable solid. In solution it
        # reverts, and the freed aldehyde is what the cysteine adds to, giving
        # the hemithioacetal. Dropping the sulfonate is therefore the right
        # bound form — the same carbon, carrying the protein instead.
        #
        # Every CovPDB site bearing this motif bonds a cysteine, which is why
        # the nucleophile list is narrow.
        smarts="[CX4;H1]([OX2H1])[SX4](=[OX1])(=[OX1])[OX2H1,OX1-]",
        reactive_index=0,
        leaving_index=2,
        role="carbonyl_carbon",
        mechanism="reversible_addition",
        nucleophiles=(THIOL, SELENOL),
        transform="drop_leaving_group",
        reversible=True,
        priority=17,
        note="A masked aldehyde. The sulfonate leaves in solution and the "
             "cysteine adds to the aldehyde, giving a hemithioacetal.",
        aliases=("alpha_hydroxysulfonate", "masked_aldehyde"),
    ),
    Warhead(
        key="keto_azole",
        label="Keto-azole (ketobenzothiazole)",
        # A ketone sitting on C2 of a 1,3-azole — thiazole or oxazole, plain or
        # benzo-fused. The ring nitrogen and the adjacent S or O drain the
        # carbonyl, so a serine or cysteine adds to it and is held as the
        # tetrahedral hemiketal. This is the telaprevir / HCV-protease
        # ketobenzothiazole family and the FAAH α-ketooxazoles.
        #
        # The ring nitrogen is what makes it a warhead, and requiring it is what
        # keeps this narrow: a 2-acylthiophene or 2-acylfuran has no ring N and
        # is not activated, and a plain aryl ketone — 150 of them in CovPDB —
        # stays out entirely. That is the distinction a generic ketone pattern
        # could not draw, which is why the generic one was rejected.
        #
        # [#6] on the far side keeps amides and esters out, the same constraint
        # the trifluoromethyl ketone needed.
        smarts="[#6][CX3](=[OX1])[c;r5;$(c(:[n]):[s,o])]",
        reactive_index=1,
        role="carbonyl_carbon",
        mechanism="reversible_addition",
        nucleophiles=(THIOL, SELENOL, HYDROXYL, AMINE),
        transform="reduce_carbonyl",
        reversible=True,
        priority=16,
        note="Serine- and cysteine-protease warhead; the nucleophile adds to "
             "the ketone to give a tetrahedral hemiketal or hemithioketal.",
        aliases=("ketobenzothiazole", "keto_oxazole", "alpha_ketoheterocycle"),
    ),
    Warhead(
        key="alpha_keto_amide",
        label="α-Keto amide",
        smarts="[CX3](=[OX1])[CX3](=[OX1])[NX3]",
        reactive_index=0,
        role="carbonyl_carbon",
        mechanism="reversible_addition",
        nucleophiles=(THIOL, SELENOL, HYDROXYL),
        transform="reduce_carbonyl",
        reversible=True,
        priority=15,
        note="Nirmatrelvir-type; the keto carbon is attacked.",
    ),
    Warhead(
        key="boronic_acid",
        label="Boronic acid / ester",
        smarts="[BX3]([OX2])[OX2]",
        reactive_index=0,
        role="boron_center",
        mechanism="reversible_addition",
        nucleophiles=(HYDROXYL, THIOL, AMINE),
        transform="none",
        reversible=True,
        priority=10,
        note="Forms a tetrahedral boronate with catalytic Ser/Thr.",
    ),
    Warhead(
        key="ligand_thiol",
        label="Thiol (disulfide formation)",
        smarts="[#6][SX2H1]",
        reactive_index=1,
        role="sulfur_center",
        mechanism="disulfide_exchange",
        nucleophiles=(THIOL, SELENOL),
        # Nothing leaves: the two sulfurs simply join.
        transform="none",
        reversible=True,
        priority=18,
        note="A free thiol oxidises to a disulfide with a cysteine. Distinct "
             "from the tethering warhead below, which already carries its S–S.",
    ),
    Warhead(
        key="ebselen",
        label="Benzisoselenazolone (ebselen)",
        # Se and N sit in an aromatic ring, so the aliphatic [Se][NX3] form
        # does not match these at all.
        smarts="[Se,se;R][N,n;R]",
        reactive_index=0,
        leaving_index=1,
        role="selenium_center",
        mechanism="addition",
        nucleophiles=(THIOL, SELENOL),
        transform="open_strained_ring",
        reversible=True,
        priority=10,
        note="The nucleophile attacks selenium and the Se–N bond opens, giving "
             "a selenenyl sulfide with cysteine.",
    ),
    Warhead(
        key="disulfide",
        label="Disulfide (tethering)",
        smarts="[SX2][SX2]",
        reactive_index=0,
        leaving_index=1,
        role="sulfur_center",
        mechanism="disulfide_exchange",
        nucleophiles=(THIOL, SELENOL),
        transform="cleave_disulfide",
        reversible=True,
        priority=15,
        note="Thiol–disulfide exchange; used in fragment tethering.",
    ),
]

WARHEADS: Dict[str, Warhead] = {
    w.key: w
    for w in (_MICHAEL + _ALKYLATORS + _STRAINED + _ACYLATORS + _PHOSPHORUS
              + _SNAR + _REVERSIBLE)
}

# Ordered most-specific-first so the classifier can stop at the best hit.
WARHEAD_ORDER: List[Warhead] = sorted(WARHEADS.values(), key=lambda w: (w.priority, w.key))

MECHANISM_LABELS = {
    "michael_addition": "Michael addition",
    "sn2_alkylation": "SN2 alkylation",
    "ring_opening": "Ring opening",
    "acylation": "Acylation",
    "sulfonylation": "Sulfonylation",
    "addition": "Addition",
    "phosphorylation": "Phosphorylation",
    "snar": "Nucleophilic aromatic substitution",
    "reversible_addition": "Reversible addition",
    "disulfide_exchange": "Disulfide exchange",
}

ROLE_LABELS = {
    "beta_carbon": "β-carbon",
    "alpha_carbon": "α-carbon",
    "alkene_carbon": "alkene carbon",
    "alkylation_carbon": "alkylation carbon",
    "epoxide_carbon": "epoxide carbon",
    "aziridine_carbon": "aziridine carbon",
    "carbonyl_carbon": "carbonyl carbon",
    "nitrile_carbon": "nitrile carbon",
    "central_carbon": "central carbon",
    "ipso_carbon": "ipso carbon",
    "sulfur_center": "sulfur centre",
    "boron_center": "boron centre",
    "selenium_center": "selenium centre",
    "phosphorus_center": "phosphorus centre",
    "diazo_carbon": "diazo carbon",
}


def get(key: str) -> Warhead:
    return WARHEADS[key]


def role_label(role: str) -> str:
    return ROLE_LABELS.get(role, role.replace("_", " "))


def mechanism_label(mech: str) -> str:
    return MECHANISM_LABELS.get(mech, mech.replace("_", " "))


def validate_library() -> List[str]:
    """Compile every SMARTS. Returns a list of problems (empty == all good)."""
    from rdkit import Chem

    problems = []
    for key, w in WARHEADS.items():
        patt = Chem.MolFromSmarts(w.smarts)
        if patt is None:
            problems.append(f"{key}: SMARTS does not compile: {w.smarts}")
            continue
        n = patt.GetNumAtoms()
        if w.reactive_index >= n:
            problems.append(f"{key}: reactive_index {w.reactive_index} >= {n} atoms")
        if w.leaving_index is not None and w.leaving_index >= n:
            problems.append(f"{key}: leaving_index {w.leaving_index} >= {n} atoms")
    return problems
