"""
prepare_receptor.py — write a receptor PDB that gnina will accept.

gnina resolves --covalent_rec_atom as CHAIN:RESNUM:ATOMNAME against the file
it is given, so whatever is written here has to keep the chain ID, residue
number and atom name of the selected atom intact. Every filter below checks
that the selected atom survives.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from chemistry.protein_reactivity import AA3_TO_1, WATERS, Structure, parse_pdb


@dataclass
class ReceptorPrep:
    path: str
    kept_chains: List[str]
    n_atoms: int
    removed_waters: int
    removed_hetero: int
    removed_hydrogens: int
    selector_ok: Optional[bool] = None
    warnings: List[str] = field(default_factory=list)

    def report(self) -> str:
        bits = [f"{self.n_atoms} atoms", f"chains {', '.join(self.kept_chains) or '—'}"]
        if self.removed_waters:
            bits.append(f"{self.removed_waters} waters removed")
        if self.removed_hetero:
            bits.append(f"{self.removed_hetero} hetero atoms removed")
        if self.removed_hydrogens:
            bits.append(f"{self.removed_hydrogens} hydrogens removed")
        return " · ".join(bits)


def prepare_receptor(
    pdb_text: str,
    out_path: str,
    keep_chains: Optional[Sequence[str]] = None,
    remove_waters: bool = True,
    remove_hetero: bool = False,
    keep_hetero_resnames: Sequence[str] = (),
    remove_hydrogens: bool = True,
    selector: Optional[str] = None,
    title: str = "",
) -> ReceptorPrep:
    """
    Filter a PDB and write it out.

    `selector` is the gnina CHAIN:RESNUM:ATOMNAME string; if given, the result
    records whether that atom is still present in the written file.
    """
    keep_hetero = {h.strip().upper() for h in keep_hetero_resnames}
    kept_lines: List[str] = []
    chains: List[str] = []
    n_water = n_het = n_h = 0
    found_selector = False

    want_chain = want_resseq = want_atom = None
    if selector:
        parts = selector.split(":")
        if len(parts) == 3:
            want_chain, want_resseq, want_atom = parts[0], parts[1], parts[2]

    for raw in pdb_text.splitlines():
        rec = raw[:6]
        if rec.startswith(("ATOM", "HETATM")):
            line = raw.ljust(80)
            resname = line[17:20].strip().upper()
            chain = line[21].strip() or "A"
            element = line[76:78].strip().upper()
            atom_name = line[12:16].strip()

            if keep_chains and chain not in keep_chains:
                continue
            if resname in WATERS:
                if remove_waters:
                    n_water += 1
                    continue
            elif rec.startswith("HETATM") and resname not in AA3_TO_1:
                if remove_hetero and resname not in keep_hetero:
                    n_het += 1
                    continue
            if remove_hydrogens and (element in ("H", "D")
                                     or (not element and atom_name[:1] == "H")):
                n_h += 1
                continue

            if line[21] == " ":
                line = line[:21] + "A" + line[22:]
                chain = "A"
            if chain not in chains:
                chains.append(chain)
            if want_chain is not None and not found_selector:
                if (chain == want_chain and line[22:26].strip() == want_resseq
                        and atom_name == want_atom):
                    found_selector = True
            kept_lines.append(line.rstrip())
        # CONECT is deliberately NOT carried over. This file is gnina's input,
        # and gnina reads it through OpenBabel: with explicit connectivity
        # present OpenBabel perceives bond orders from it rather than from
        # geometry, fails to kekulise aromatic hetero groups, and mis-types
        # the atoms Vina scores on. Keeping them moved a measured affinity by
        # several kcal/mol. Connectivity for viewing belongs in the complex
        # file, not in the docking input.
        elif rec.startswith(("TER", "SSBOND", "LINK", "HELIX", "SHEET")):
            kept_lines.append(raw.rstrip())

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(f"REMARK   Prepared for covalent docking{': ' + title if title else ''}\n")
        if selector:
            fh.write(f"REMARK   Covalent receptor atom: {selector}\n")
        for line in kept_lines:
            fh.write(line + "\n")
        fh.write("END\n")

    prep = ReceptorPrep(
        path=out_path,
        kept_chains=chains,
        n_atoms=sum(1 for l in kept_lines if l.startswith(("ATOM", "HETATM"))),
        removed_waters=n_water,
        removed_hetero=n_het,
        removed_hydrogens=n_h,
        selector_ok=found_selector if selector else None,
    )
    if selector and not found_selector:
        prep.warnings.append(
            f"The receptor atom {selector} is not in the written file. "
            "gnina will not be able to resolve it — check the chain filter."
        )
    if not chains:
        prep.warnings.append("No atoms survived the filters.")
    return prep


def structure_from_text(pdb_text: str, source: str = "") -> Structure:
    return parse_pdb(pdb_text, source=source)


def binding_site_pdb(structure: Structure, center: Sequence[float],
                     out_path: str, radius: float = 12.0) -> str:
    """
    Write the residues around a point as their own PDB.

    Useful as a gnina --autobox_ligand argument when you would rather let
    gnina size the box than give explicit dimensions.
    """
    import math

    cx, cy, cz = center
    keep = set()
    for atom in structure.atoms:
        d = math.dist((atom.x, atom.y, atom.z), (cx, cy, cz))
        if d <= radius:
            keep.add((atom.chain, atom.resseq, atom.icode))

    lines = []
    for atom in structure.atoms:
        if (atom.chain, atom.resseq, atom.icode) not in keep:
            continue
        lines.append(
            f"ATOM  {atom.serial:>5} {atom.name:<4}{'':1}{atom.resname:>3} "
            f"{atom.chain}{atom.resseq:>4}{atom.icode}   "
            f"{atom.x:>8.3f}{atom.y:>8.3f}{atom.z:>8.3f}"
            f"{1.00:>6.2f}{atom.bfactor:>6.2f}          {atom.element:>2}"
        )
    with open(out_path, "w") as fh:
        fh.write("REMARK   Binding-site shell for autobox\n")
        fh.write("\n".join(lines) + "\nEND\n")
    return out_path
