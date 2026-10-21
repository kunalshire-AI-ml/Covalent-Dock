"""
molecule_2d.py — flat depiction of the ligand with its reactive atoms marked.

The ligand-side counterpart to the sequence grid: candidate electrophiles in
amber, the selected atom in the same indigo the sequence card uses, so the two
panels read as one interface.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from chemistry.ligand_reactivity import atom_label

SELECTED = (0.870, 0.890, 0.992)     # #dfe3fd
SELECTED_EDGE = (0.310, 0.341, 0.769)
CANDIDATE = (0.992, 0.965, 0.890)    # #fdf6e3
CANDIDATE_EDGE = (0.878, 0.714, 0.176)


_ORIG_IDX = "_covdockOrigIdx"


def _prepare(mol: Chem.Mol) -> Tuple[Chem.Mol, Dict[int, int]]:
    """
    The flat molecule, plus a map from the caller's atom indices to it.

    RemoveHs renumbers: it compacts the atoms that survive, so an index is only
    unchanged when every hydrogen already sat after every heavy atom. That is
    true of a bound form built here, but not of an arbitrary uploaded SDF with
    its hydrogens interleaved — and there the highlighted atom would silently
    become a different one. Tag the atoms first and read the tags back, so the
    mapping is whatever actually happened rather than an assumption about it.
    """
    flat = Chem.Mol(mol)
    for atom in flat.GetAtoms():
        atom.SetIntProp(_ORIG_IDX, atom.GetIdx())
    try:
        flat = Chem.RemoveHs(flat)
    except Exception:                                         # noqa: BLE001
        pass

    # Lay the molecule out flat, rather than drawing whatever conformer it
    # arrived with. The bound form comes back from embed_3d with 3-D
    # coordinates, and RDKit will happily draw those by projecting them onto
    # the page: rings end up edge-on as a bundle of overlapping lines and
    # distant atoms land on top of each other. A real 2-D depiction has to be
    # computed, so drop the conformers first and let the depictor do it.
    flat.RemoveAllConformers()
    try:
        rdDepictor.SetPreferCoordGen(True)
    except Exception:                                         # noqa: BLE001
        pass                                                  # older RDKit
    rdDepictor.Compute2DCoords(flat)
    rdMolDraw2D.PrepareMolForDrawing(flat, kekulize=True)

    mapping: Dict[int, int] = {}
    for atom in flat.GetAtoms():
        if atom.HasProp(_ORIG_IDX):
            mapping[atom.GetIntProp(_ORIG_IDX)] = atom.GetIdx()
    return flat, mapping


def molecule_svg(
    mol: Chem.Mol,
    candidate_atoms: Sequence[int] = (),
    selected_atom: Optional[int] = None,
    labels: Optional[Dict[int, str]] = None,
    width: int = 520,
    height: int = 360,
    title: str = "",
    number_atoms: bool = False,
) -> str:
    """
    SVG string. Atom indices are the caller's — indices into `mol` as given,
    which is the indexing the rest of the app uses. They are translated to the
    hydrogen-stripped depiction internally.

    `number_atoms` annotates every heavy atom with the name the rest of the app
    shows it under — the deposited PDB name where there is one, otherwise the
    '#'-marked index form. It comes from `ligand_reactivity.atom_label` rather
    than RDKit's own `addAtomIndices`, which numbers from 0 and would disagree
    with every other label in the interface.
    """
    flat, index_of = _prepare(mol)
    highlight_atoms = []
    colors: Dict[int, Tuple[float, float, float]] = {}
    radii: Dict[int, float] = {}

    for idx in candidate_atoms:
        drawn = index_of.get(idx)
        if drawn is not None:
            highlight_atoms.append(drawn)
            colors[drawn] = CANDIDATE
            radii[drawn] = 0.45
    selected_drawn = index_of.get(selected_atom) if selected_atom is not None else None
    if selected_drawn is not None:
        if selected_drawn not in highlight_atoms:
            highlight_atoms.append(selected_drawn)
        colors[selected_drawn] = SELECTED
        radii[selected_drawn] = 0.58

    # Numbering goes on first so an explicit `labels` entry wins the atom it
    # names: the reactive atom should read "attachment point", not "C#36".
    # Names come from the caller's molecule, where the index in a '#'-marked
    # name is the one the rest of the app quotes.
    notes: Dict[int, str] = {}
    if number_atoms:
        notes = {i: atom_label(mol, i) for i in index_of}
    notes.update(labels or {})
    for idx, text in notes.items():
        drawn = index_of.get(idx)
        if drawn is not None:
            flat.GetAtomWithIdx(drawn).SetProp("atomNote", text)

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.addStereoAnnotation = False
    # Numbering every atom puts a note on each one; at the shared 0.7 scale they
    # collide on a crowded skeleton, so shrink the notes and let the larger
    # canvas carry the legibility instead.
    options.annotationFontScale = 0.55 if number_atoms else 0.7
    options.highlightBondWidthMultiplier = 12
    options.fillHighlights = True
    if title:
        options.legendFontSize = 14

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, flat,
        legend=title,
        highlightAtoms=highlight_atoms or None,
        highlightAtomColors=colors or None,
        highlightAtomRadii=radii or None,
        highlightBonds=[],
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


_SVG_SIZE = re.compile(r"width='\d+px' height='\d+px'")


def _responsive(svg: str) -> str:
    """
    Let the drawing fill its column.

    RDKit stamps a fixed pixel width and height on the <svg>, so a depiction
    sized for a wide canvas still renders at those pixels and a large molecule
    comes out as an unreadable thumbnail. The viewBox it also writes carries
    the aspect ratio, so dropping the fixed size scales the drawing to whatever
    width Streamlit gives it without distorting it.
    """
    return _SVG_SIZE.sub("width='100%' height='auto'", svg, count=1)


def render_molecule_2d(mol: Chem.Mol, **kwargs) -> None:
    import streamlit as st

    svg = _responsive(molecule_svg(mol, **kwargs))
    st.markdown(
        f'<div style="border:1px solid rgba(49,51,63,0.10);border-radius:8px;'
        f'padding:8px;background:#fff;text-align:center">{svg}</div>',
        unsafe_allow_html=True,
    )


def reaction_svg(before: Chem.Mol, after: Chem.Mol,
                 before_atom: Optional[int] = None,
                 after_atom: Optional[int] = None,
                 width: int = 260, height: int = 200) -> Tuple[str, str]:
    """Side-by-side depiction of the ligand before and after the bond forms."""
    left = molecule_svg(before, selected_atom=before_atom,
                        width=width, height=height, title="as supplied")
    right = molecule_svg(after, selected_atom=after_atom,
                         width=width, height=height, title="bound form")
    return left, right
