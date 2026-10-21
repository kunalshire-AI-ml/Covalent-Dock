#!/usr/bin/env python3
"""
app.py — covalent docking, set up by clicking.

The user picks a protein atom and a ligand atom. The app works out the warhead,
builds the bound form, generates the SMARTS and assembles the gnina command.
SMARTS stays behind an "advanced" panel.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import dataclasses
import os
import sys
from typing import Dict, List, Optional

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import plip as plip_mod
from analysis import rmsd as rmsd_mod
from analysis import scores as scores_mod
from chemistry import atom_names
from chemistry import attachment as attach
from chemistry import bound_form as bf_mod
from chemistry import compatibility as compat
from chemistry import ligand_reactivity as lig_chem
from chemistry import protein_reactivity as prot_chem
from chemistry import smarts_generator as smarts_mod
from chemistry.protein_reactivity import (NUCLEOPHILE_ELEMENT,
                                          NUCLEOPHILE_LABELS, WATERS)
from core.session import WorkSession, list_runs, prune
from docking import complex_builder, gnina, prepare_ligand, prepare_receptor
from docking import batch as batch_mod
from visualization import molecule_2d, sequence_view, viewer

st.set_page_config(page_title="Covalent docking", page_icon="⚗️",
                   layout="wide", initial_sidebar_state="expanded")

EXAMPLE_FRAGMENTS = {
    "f1 · N-acryloylpiperidine": "C=CC(N1CCCCC1)=O",
    "f2 · phenyl vinyl ketone": "C=CC(C1=CC=CC=C1)=O",
    "f3 · acrylanilide": "C=CC(NC1=CC=CC=C1)=O",
}

PROTEIN_FILE = "protein_input.pdb"
RECEPTOR_FILE = "receptor.pdb"
LIGAND_FILE = "ligand_docked.sdf"
POSES_FILE = "poses.sdf"
FLEX_FILE = "gnina_complex.sdf"

STRENGTH_COLOR = {"strong": "#1a7f5a", "plausible": "#8a6d1f",
                  "weak": "#8a4b1f", "incompatible": "#8a1f1f",
                  "unsuitable": "#8a1f1f"}

# Anything derived from the chosen atoms. A new selection invalidates all of it:
# a ligand placed at the old site with a box centred on the new one gives
# "ligand outside box" and scores that mean nothing.
STALE = ("gnina_done", "receptor_prepared", "ligand_prepared", "smarts",
         "pattern_check", "required_box", "placement_note",
         "placement_distance", "bound_reactive_idx")


# ═════════════════════════════════════════════════════════════════════════════
#  Cached chemistry
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def parse_protein(pdb_text: str) -> prot_chem.Structure:
    return prot_chem.parse_pdb(pdb_text)


@st.cache_data(show_spinner=False)
def protein_nucleophiles(pdb_text: str, chains: tuple, types: tuple
                         ) -> List[prot_chem.Nucleophile]:
    structure = prot_chem.parse_pdb(pdb_text)
    found = prot_chem.find_nucleophiles(structure, chains=list(chains) or None,
                                        types=list(types) or None)
    return prot_chem.rank_nucleophiles(found)


@st.cache_data(show_spinner=False)
def poses_rmsd(path: str, mtime: float) -> rmsd_mod.RmsdResult:
    """Pairwise heavy-atom RMSD; `mtime` makes the cache follow the file."""
    return rmsd_mod.pose_rmsd_matrix(path)


@st.cache_resource(show_spinner=False)
def ligand_mol(text: str, fmt: str, name: str):
    return lig_chem.load_ligand_text(text, fmt, name=name)


@st.cache_data(show_spinner=False)
def ligand_electrophiles(text: str, fmt: str, name: str, promiscuous: bool,
                         reversible: bool) -> List[lig_chem.Electrophile]:
    loaded = lig_chem.load_ligand_text(text, fmt, name=name)
    return lig_chem.find_electrophiles(loaded.mol,
                                       include_promiscuous=promiscuous,
                                       include_reversible=reversible)


# ═════════════════════════════════════════════════════════════════════════════
#  Session plumbing
# ═════════════════════════════════════════════════════════════════════════════

def get_session() -> WorkSession:
    params = st.query_params
    run_id = params.get("s")
    session = WorkSession.load_or_create(run_id)
    if params.get("s") != session.run_id:
        st.query_params["s"] = session.run_id
    return session


def handle_residue_click(session: WorkSession, pdb_text: str) -> None:
    """A click in the sequence grid arrives as ?res=<residue key>."""
    key = st.query_params.get("res")
    if not key:
        return
    del st.query_params["res"]
    candidates = protein_nucleophiles(
        pdb_text, tuple(session.get("chains", [])),
        tuple(session.get("nu_types", [])))
    match = [n for n in candidates if n.residue_key == key]
    if match:
        select_nucleophile(session, match[0])
    else:
        st.session_state["residue_click_miss"] = key


def select_nucleophile(session: WorkSession, nu: prot_chem.Nucleophile) -> None:
    session.set("nucleophile", {
        "selector": nu.selector, "chain": nu.chain, "resseq": nu.resseq,
        "atom": nu.atom_name, "resname": nu.resname, "label": nu.label,
        "type": nu.nucleophile_type, "residue_key": nu.residue_key,
        "xyz": list(nu.xyz), "description": nu.description,
        "disulfide": nu.disulfide_partner, "exposure": nu.exposure,
    })
    session.clear(*STALE)


def select_electrophile(session: WorkSession, el: lig_chem.Electrophile) -> None:
    session.set("electrophile", {
        "atom_idx": el.atom_idx, "label": el.label, "element": el.element,
        "warhead": el.warhead_key, "warhead_label": el.warhead_label,
        "role": el.role_label, "mechanism": el.mechanism,
        "mechanism_label": el.mechanism_label, "transform": el.transform,
        "bond_order": el.bond_order, "reversible": el.reversible,
        "leaving_atom": el.leaving_atom, "confidence": el.confidence,
    })
    session.clear(*STALE)


# ═════════════════════════════════════════════════════════════════════════════
#  Small UI helpers
# ═════════════════════════════════════════════════════════════════════════════

def badge(text: str, color: str = "#4f57c4", background: str = "#eef0ff") -> str:
    return (f'<span style="background:{background};color:{color};padding:2px 8px;'
            f'border-radius:10px;font-size:0.74rem;white-space:nowrap">{text}</span>')


def caption_row(items: Dict[str, str]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items.items()):
        col.metric(label, value)


def status_strip(session: WorkSession) -> None:
    nu = session.get("nucleophile")
    el = session.get("electrophile")
    bits = []
    bits.append(badge(session.get("protein_name") or "no protein",
                      "#31333F", "#f0f2f6"))
    bits.append(badge(session.get("ligand_name") or "no ligand",
                      "#31333F", "#f0f2f6"))
    if nu:
        bits.append(badge(f"protein: {nu['label']}"))
    if el:
        bits.append(badge(f"ligand: {el['label']} · {el['warhead_label']}",
                          "#6b5a22", "#fdf6e3"))
    st.markdown(
        '<div style="margin:-6px 0 14px 0;display:flex;gap:8px;flex-wrap:wrap">'
        + "".join(bits) + "</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ═════════════════════════════════════════════════════════════════════════════

def sidebar(session: WorkSession) -> Dict:
    with st.sidebar:
        st.markdown("### Covalent docking")
        st.caption("Pick a protein atom and a ligand atom. The chemistry, the "
                   "SMARTS and the gnina command follow from the pair.")

        binary = gnina.find_binary()
        if binary:
            st.success(f"gnina found: {os.path.basename(binary)}", icon="✅")
        else:
            st.warning("gnina is not on PATH. You can still set everything up "
                       "and copy the command.", icon="⚠️")

        st.divider()
        st.markdown("**Which residues to offer**")
        types = st.multiselect(
            "Nucleophile types", list(NUCLEOPHILE_LABELS.keys()),
            default=session.get("nu_types", ["thiol", "amine", "imidazole",
                                             "hydroxyl", "selenol"]),
            format_func=lambda t: NUCLEOPHILE_LABELS[t],
            help="Carboxylates (Asp/Glu) are off by default: chemically "
                 "possible, rarely the intended site.")
        hide_disulfide = st.checkbox(
            "Hide cysteines already in a disulfide", value=True)

        st.markdown("**Sequence view**")
        per_row = st.slider("Residues per row", 10, 40,
                            session.get("per_row", 20), step=5)
        show_numbers = st.checkbox("Show residue numbers",
                                   value=session.get("show_numbers", False))
        clickable = st.checkbox("Click a residue to select it", value=True,
                                help="Clicking reloads the page; the run is "
                                     "restored from disk.")

        st.divider()
        st.markdown("**Ligand scan**")
        # On by default: a plain aldehyde or nitrile really is the warhead in a
        # large share of deposited covalent complexes (peptide aldehydes,
        # saxagliptin, odanacatib). These rank last and carry a "confirm this is
        # really the warhead" caveat, so switching them off hides a real option
        # rather than protecting the user from a wrong one.
        promiscuous = st.checkbox("Include weakly activated groups", value=True,
                                  help="Plain nitriles, aldehydes, alkyl halides. "
                                       "Offered last, with a low-confidence note. "
                                       "Untick to see only strongly activated "
                                       "warheads.")
        reversible = st.checkbox("Include reversible warheads", value=True)

        st.divider()
        st.markdown("**Docking**")
        exhaustiveness = st.slider("Exhaustiveness", 1, 64,
                                   session.get("exhaustiveness", 16))
        num_modes = st.slider("Poses to keep", 1, 20, session.get("num_modes", 20))
        box_size = st.slider("Box size (Å)", 10, 60, session.get("box_size", 20))
        autofit_box = st.checkbox(
            "Grow the box to fit the placed ligand", value=True,
            help="A covalent ligand hangs off one side of the receptor atom "
                 "and pivots about it, so the box has to cover its whole reach. "
                 "Too small a box and gnina reports 'ligand outside box' and "
                 "returns scores that mean nothing.")
        optimize_lig = st.checkbox("UFF-optimise the covalent complex", value=True)
        seed = st.number_input("Random seed", value=42, step=1)
        # This deployment has no GPU, so the CNN rescoring is off and gnina is
        # pinned to the CPU. Neither is offered as a choice.
        cnn_scoring = "none"
        no_gpu = True
        st.caption("Runs CPU-only with the CNN off "
                   "(`--cnn_scoring none --no_gpu`): there is no GPU on this "
                   "host, so CNN rescoring would only cost time.")

        st.divider()
        with st.expander("Run directory"):
            st.code(session.dir, language=None)
            st.caption(f"{len(session.files())} file(s) · {session.size_mb()} MB")
            if st.button("Start a new run", use_container_width=True):
                st.query_params.clear()
                st.rerun()
            if st.button("Delete this run", use_container_width=True):
                session.delete()
                st.query_params.clear()
                st.rerun()

    settings = dict(nu_types=types, hide_disulfide=hide_disulfide,
                    per_row=per_row, show_numbers=show_numbers,
                    clickable=clickable, promiscuous=promiscuous,
                    reversible=reversible, exhaustiveness=exhaustiveness,
                    num_modes=num_modes, box_size=box_size,
                    autofit_box=autofit_box,
                    cnn_scoring=cnn_scoring, optimize_lig=optimize_lig,
                    no_gpu=no_gpu, seed=int(seed))
    session.data.update({k: settings[k] for k in
                         ("nu_types", "per_row", "show_numbers", "exhaustiveness",
                          "num_modes", "box_size")})
    session.save()
    return settings


# ═════════════════════════════════════════════════════════════════════════════
#  Step 1 — upload
# ═════════════════════════════════════════════════════════════════════════════

def step_upload(session: WorkSession) -> None:
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### Protein")
        uploaded = st.file_uploader("PDB or mmCIF file",
                                    type=["pdb", "ent", "txt", "cif", "mmcif"],
                                    key="protein_upload")
        pasted = st.text_area("or paste PDB / mmCIF text", height=90,
                              key="protein_paste",
                              placeholder="ATOM      1  N   MET A   1 ...")

        def accept_structure(text: str, name: str) -> None:
            """
            Store the upload as PDB, converting mmCIF on the way in.

            Recent PDB entries exist only as mmCIF, because a 5-character
            ligand code or a multi-character chain ID does not fit the fixed
            columns of a .pdb file. Those are exactly the names that have to be
            renamed to store it here, so the user is told what changed — the
            chain they see in the gnina selector will be the new one.
            """
            pdb_text, chain_map, het_map = prot_chem.as_pdb_text(text, name)
            session.write_text(PROTEIN_FILE, pdb_text)
            session.update(protein_name=name)
            renamed = {k: v for k, v in chain_map.items() if k != v}
            if renamed or het_map:
                bits = []
                if renamed:
                    bits.append("chains " + ", ".join(f"{k}→{v}" for k, v in
                                                      list(renamed.items())[:4]))
                if het_map:
                    bits.append("ligands " + ", ".join(f"{k}→{v}" for k, v in
                                                       list(het_map.items())[:4]))
                st.info("mmCIF read and converted to PDB. Renamed " +
                        "; ".join(bits) + " — these are too wide for the PDB "
                        "format. The new names are what the gnina selector "
                        "will use.", icon="ℹ️")

        if uploaded is not None:
            accept_structure(uploaded.getvalue().decode("utf-8", errors="replace"),
                             uploaded.name)
            st.success(f"Loaded {uploaded.name}")
        elif pasted.strip() and st.button("Use pasted structure"):
            accept_structure(pasted, "pasted structure")
            st.rerun()

        if session.has(PROTEIN_FILE):
            structure = parse_protein(session.read_text(PROTEIN_FILE))
            summary = structure.summary()
            st.caption(structure.title[:120] or "—")
            caption_row({"Atoms": str(summary["atoms"]),
                         "Residues": str(summary["residues"]),
                         "Chains": str(len(summary["chains"]))})
            if summary["hetero"]:
                st.caption("Hetero groups: " + ", ".join(summary["hetero"][:12]))

    with right:
        st.markdown("#### Ligand")
        tab_smiles, tab_file, tab_example = st.tabs(["SMILES", "File", "Examples"])

        with tab_smiles:
            smiles = st.text_input("SMILES", key="ligand_smiles",
                                   placeholder="C=CC(N1CCCCC1)=O")
            name = st.text_input("Name", value="ligand", key="ligand_smiles_name")
            if smiles.strip() and st.button("Use this SMILES"):
                try:
                    lig_chem.load_ligand_text(smiles, "smiles", name)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    session.write_text("ligand_input.smi", smiles.strip())
                    session.update(ligand_text=smiles.strip(), ligand_format="smiles",
                                   ligand_name=name or "ligand")
                    session.clear("electrophile", "smarts")
                    st.rerun()

        with tab_file:
            lig_file = st.file_uploader("SDF / MOL / MOL2 / PDB",
                                        type=["sdf", "mol", "mol2", "pdb"],
                                        key="ligand_upload")
            if lig_file is not None:
                text = lig_file.getvalue().decode("utf-8", errors="replace")
                fmt = os.path.splitext(lig_file.name)[1].lstrip(".").lower()
                try:
                    loaded = lig_chem.load_ligand_text(text, fmt, lig_file.name)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    session.write_text(f"ligand_input.{fmt}", text)
                    session.update(ligand_text=text, ligand_format=fmt,
                                   ligand_name=loaded.name)
                    session.clear("electrophile", "smarts")
                    for warning in loaded.warnings:
                        st.warning(warning)
                    st.success(f"Loaded {lig_file.name}")

        with tab_example:
            st.caption("The Michael-acceptor fragments from prep_ligands.py.")
            for label, smi in EXAMPLE_FRAGMENTS.items():
                col_a, col_b = st.columns([3, 1])
                col_a.code(smi, language=None)
                if col_b.button("Use", key=f"ex_{label}"):
                    session.write_text("ligand_input.smi", smi)
                    session.update(ligand_text=smi, ligand_format="smiles",
                                   ligand_name=label.split(" · ")[0])
                    session.clear("electrophile", "smarts")
                    st.rerun()

        if session.get("ligand_text"):
            loaded = ligand_mol(session.get("ligand_text"),
                                session.get("ligand_format"),
                                session.get("ligand_name"))
            info = lig_chem.describe_ligand(loaded.mol)
            caption_row({"Heavy atoms": str(info["heavy_atoms"]),
                         "MW": f"{info['mw']:.1f}",
                         "Rotatable": str(info["rotatable_bonds"])})

    if session.has(PROTEIN_FILE) and session.get("ligand_text"):
        # No `icon=`: Streamlit validates it as a single emoji, and the "→"
        # that used to be here is a typographic arrow, not one. The default
        # success icon says the same thing without the risk.
        st.success("Both inputs are in. Move on to **2 · Protein site**.")


# ═════════════════════════════════════════════════════════════════════════════
#  Step 2 — protein site
# ═════════════════════════════════════════════════════════════════════════════

def step_protein(session: WorkSession, settings: Dict) -> None:
    if not session.has(PROTEIN_FILE):
        st.info("Upload a protein in step 1 first.")
        return

    pdb_text = session.read_text(PROTEIN_FILE)
    structure = parse_protein(pdb_text)
    polymer_chains = [c for c in structure.chains if c.polymer]
    if not polymer_chains:
        st.error("No polymer chains found in this file.")
        return

    chain_ids = [c.chain_id for c in polymer_chains]
    default = session.get("chain") if session.get("chain") in chain_ids else chain_ids[0]
    chain_id = st.selectbox(
        "Chain", chain_ids, index=chain_ids.index(default),
        format_func=lambda cid: sequence_view.chain_picker_label(
            structure.chain(cid)))
    session.set("chain", chain_id)

    nucleophiles = protein_nucleophiles(pdb_text, (), tuple(settings["nu_types"]))
    if settings["hide_disulfide"]:
        nucleophiles = [n for n in nucleophiles if not n.disulfide_partner]
    in_chain = [n for n in nucleophiles if n.chain == chain_id]

    chain = structure.chain(chain_id)
    residues = chain.polymer
    candidate_keys = {n.residue_key for n in in_chain}
    selected = session.get("nucleophile")
    selected_keys = {selected["residue_key"]} if selected else set()

    tooltips = {}
    by_residue: Dict[str, List[prot_chem.Nucleophile]] = {}
    for n in in_chain:
        by_residue.setdefault(n.residue_key, []).append(n)
    for res in residues:
        hits = by_residue.get(res.key)
        if hits:
            atoms = "/".join(h.atom_name for h in hits)
            tooltips[res.key] = f"{res.label} · {atoms} · {hits[0].context_note()}"
        else:
            tooltips[res.key] = res.label

    link_for = None
    if settings["clickable"]:
        def link_for(res):                                    # noqa: F811
            if res.key not in candidate_keys:
                return None
            return f"?s={session.run_id}&res={res.key}"

    st.markdown(sequence_view.sequence_html(
        residues,
        selected_keys=selected_keys,
        candidate_keys=candidate_keys,
        per_row=settings["per_row"],
        show_numbers=settings["show_numbers"],
        tooltips=tooltips,
        link_for=link_for,
        caption=("Highlighted = selected interacting residues &nbsp;·&nbsp; "
                 + sequence_view.legend("selected", "candidate nucleophile")),
    ), unsafe_allow_html=True)

    if st.session_state.pop("residue_click_miss", None):
        st.warning("That residue has no nucleophilic atom in the current filter.")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("##### Candidate covalent residues")
        if not in_chain:
            st.info("No candidate nucleophiles in this chain with the current "
                    "filters. Widen the nucleophile types in the sidebar.")
        else:
            labels = {
                n.selector: (f"{n.label}  ·  {NUCLEOPHILE_LABELS[n.nucleophile_type]}"
                             f"  ·  {n.context_note()}")
                for n in in_chain
            }
            options = list(labels.keys())
            current = selected["selector"] if selected else None
            index = options.index(current) if current in options else 0
            chosen = st.radio("Pick the atom to bond to", options, index=index,
                              format_func=lambda s: labels[s],
                              label_visibility="collapsed")
            if chosen and (not selected or selected["selector"] != chosen):
                match = next(n for n in in_chain if n.selector == chosen)
                select_nucleophile(session, match)
                st.rerun()
            st.caption(f"{len(in_chain)} candidate atom(s) shown · "
                       f"{len(nucleophiles)} in the whole structure")

    with right:
        st.markdown("##### Structure")
        show_3d = st.checkbox("Show the 3-D view", value=False,
                              help="Loads 3Dmol.js from a CDN.")
        if show_3d:
            markers = [viewer.nucleophile_marker(n) for n in in_chain[:60]]
            chosen_marker = None
            if selected:
                chosen_marker = {"chain": selected["chain"],
                                 "resi": selected["resseq"],
                                 "atom": selected["atom"],
                                 "label": selected["label"]}
            viewer.render_structure(pdb_text, selected=chosen_marker,
                                    candidates=markers, height=380)
        elif selected:
            st.markdown(
                f"**{selected['label']}** · {selected['description']}  \n"
                f"gnina selector `{selected['selector']}`  \n"
                f"{selected['exposure']}"
                + (f" · disulfide with {selected['disulfide']}"
                   if selected.get("disulfide") else ""))
            if selected.get("disulfide"):
                st.warning("This cysteine is disulfide-bonded, so it is "
                           "probably not free to react.", icon="⚠️")
        else:
            st.caption("Nothing selected yet.")


def crystal_names_panel(session: WorkSession, loaded) -> None:
    """
    Offer the ligand the atom names it has in the uploaded structure.

    Without this, a ligand loaded from SDF is labelled by RDKit's own atom
    order — 'C#20' — while the same atom is 'C36' in the structure the user is
    comparing against. The two numbers are unrelated, so the '#' marks a name
    that must not be compared with a crystal one. Adopting the deposited names
    removes the '#' and makes the comparison valid.
    """
    text = session.read_text(PROTEIN_FILE)
    if not text:
        return
    try:
        structure = parse_protein(text)
    except Exception:                                         # noqa: BLE001
        return
    candidates = atom_names.hetero_candidates(structure)
    if not candidates:
        return

    active = session.get("atom_names") or {}
    with st.expander("Atom names from the structure"
                     + (f" · using {session.get('atom_names_source', '')}" if active else ""),
                     expanded=False):
        st.caption(
            "Matches the ligand onto a hetero group in the uploaded structure "
            "by position, and takes its deposited atom names. Only a ligand "
            "that came from this structure will match — which is the only case "
            "where its crystal names mean anything.")

        labels = {atom_names.residue_label(r): r for r in candidates}
        pick = st.selectbox("Hetero group", list(labels))
        residue = labels[pick]
        preview = atom_names.transfer_from_residue(loaded.mol, residue)
        st.caption(preview.report())
        for warning in preview.warnings:
            st.caption("⚠ " + warning)

        if preview.ok:
            sample = list(preview.names.items())[:8]
            st.caption("For example: " + ", ".join(
                f"{lig_chem.atom_label(loaded.mol, i)} → **{n}**" for i, n in sample))

        a, b = st.columns(2)
        if a.button("Use these names", use_container_width=True,
                    disabled=not preview.ok):
            session.set("atom_names", {str(k): v for k, v in preview.names.items()})
            session.set("atom_names_source", preview.source)
            st.rerun()
        if b.button("Back to index names", use_container_width=True,
                    disabled=not active):
            session.set("atom_names", {})
            session.set("atom_names_source", "")
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  Step 3 — ligand warhead
# ═════════════════════════════════════════════════════════════════════════════

def step_ligand(session: WorkSession, settings: Dict) -> None:
    if not session.get("ligand_text"):
        st.info("Add a ligand in step 1 first.")
        return

    loaded = ligand_mol(session.get("ligand_text"), session.get("ligand_format"),
                        session.get("ligand_name"))
    electrophiles = ligand_electrophiles(
        session.get("ligand_text"), session.get("ligand_format"),
        session.get("ligand_name"), settings["promiscuous"], settings["reversible"])

    # Deposited atom names, if the user has adopted them. Applied to a copy —
    # `ligand_mol` is cached and shared, so renaming its molecule in place would
    # leak into every later use of that cache. Both the molecule and the already
    # detected electrophiles are relabelled, or the radio would still be
    # offering 'C#20' for an atom the rest of the page now calls 'C36'.
    stored_names = session.get("atom_names") or {}
    if stored_names:
        names = {int(k): v for k, v in stored_names.items()}
        loaded = dataclasses.replace(
            loaded, mol=atom_names.apply_names(loaded.mol, names))
        electrophiles = [
            dataclasses.replace(e, label=lig_chem.atom_label(loaded.mol, e.atom_idx))
            for e in electrophiles
        ]

    crystal_names_panel(session, loaded)

    # No detection does not end the step. A ligand taken from a crystal
    # structure is already in its reacted form — its attachment carbon is a
    # plain carbinol by then — so there is nothing left for the warhead library
    # to find, and that is exactly when choosing the atom by hand matters most.
    if not electrophiles:
        st.warning("No electrophilic atoms matched the warhead library. Turn on "
                   "**Include weakly activated groups** in the sidebar, or check "
                   "that the ligand has bond orders (PDB input often does not). "
                   "If this ligand came out of a crystal structure it is already "
                   "the reacted form, and there is nothing to detect — choose the "
                   "attachment atom by hand below.", icon="⚠️")

    nu_state = session.get("nucleophile")
    nu = None
    rows: List = []
    if nu_state:
        # Built whenever a protein atom is chosen, detections or not: the
        # by-hand ranking scores against this nucleophile too.
        nu = prot_chem.Nucleophile(
            chain=nu_state["chain"], resseq=nu_state["resseq"], icode=" ",
            resname=nu_state["resname"], atom_name=nu_state["atom"],
            element="S" if nu_state["type"] == "thiol" else nu_state["atom"][0],
            nucleophile_type=nu_state["type"], description=nu_state["description"],
            xyz=tuple(nu_state["xyz"]), bfactor=0.0,
            residue_key=nu_state["residue_key"],
            disulfide_partner=nu_state.get("disulfide"),
        )
    if nu is not None and electrophiles:
        rows = compat.filter_for(nu, electrophiles)
        st.caption(f"Filtered for **{nu_state['label']}** "
                   f"({NUCLEOPHILE_LABELS[nu_state['type']]}). "
                   f"{len(rows)} of {len(electrophiles)} detected atoms are "
                   "compatible.")
    elif electrophiles:
        st.info("No protein atom chosen yet, so every detected electrophile is "
                "listed. Pick a residue in step 2 to filter this list.")
        rows = [(e, None) for e in electrophiles]

    left, right = st.columns([1, 1], gap="large")
    selected_el = session.get("electrophile")
    manual_now = bool(selected_el) and selected_el.get("warhead") == "manual"

    with left:
        st.markdown("##### Detected warheads")
        if not rows:
            st.warning("None of the detected warheads is compatible with that "
                       "nucleophile.", icon="⚠️")
        else:
            options = [e.atom_idx for e, _ in rows]
            current = selected_el["atom_idx"] if selected_el else None
            index = options.index(current) if current in options else 0

            def describe(idx: int) -> str:
                el, verdict = next((e, v) for e, v in rows if e.atom_idx == idx)
                text = f"{el.label} · {el.warhead_label} {el.role_label}"
                if verdict:
                    text += f"  ·  {verdict.badge.lower()}"
                return text

            # The radio writes the selection from its on_change callback, not
            # from its value on every rerun. An atom chosen by hand is usually
            # not in this list, so the widget falls back to its first entry —
            # and acting on that value unprompted would overwrite the by-hand
            # choice on the very next rerun, every time.
            radio_key = "warhead_radio"
            if st.session_state.get(radio_key) not in options:
                st.session_state[radio_key] = options[index]

            def choose_detected() -> None:
                picked = st.session_state[radio_key]
                select_electrophile(
                    session, next(e for e, _ in rows if e.atom_idx == picked))

            st.radio("Pick the reacting atom", options, format_func=describe,
                     label_visibility="collapsed", key=radio_key,
                     on_change=choose_detected)
            chosen = st.session_state[radio_key]

            el, verdict = next((e, v) for e, v in rows if e.atom_idx == chosen)
            if verdict:
                colour = STRENGTH_COLOR[verdict.strength]
                st.markdown(
                    badge(verdict.badge, colour, colour + "18")
                    + f" &nbsp; {verdict.reason}", unsafe_allow_html=True)
                for caveat in verdict.caveats:
                    st.caption("· " + caveat)
            if el.note:
                st.caption(el.note)
            if el.alternates:
                st.caption("Also matches: " + ", ".join(el.alternates))
            if manual_now:
                st.caption("⚠ An atom chosen by hand is active, so this list is "
                           "not driving the selection. Click an entry here to "
                           "go back to a detected warhead.")

        # ── choose the atom yourself ─────────────────────────────────────────
        with st.expander("Choose the attachment atom by hand",
                         expanded=not rows or manual_now):
            st.caption(
                "Every heavy atom scored for how well it would serve as the "
                "attachment point. This ranks a choice you make — it is not a "
                "prediction that the reaction happens.")

            ranked = attach.rank_atoms(loaded.mol, nucleophile=nu,
                                       electrophiles=electrophiles)
            hide_bad = st.checkbox(
                "Hide atoms that cannot carry the bond", value=True,
                help="Hydrogens, halogen leaving groups, and atoms with no free "
                     "valence to bond through.")
            shown = [a for a in ranked if a.band != attach.UNSUITABLE] \
                if hide_bad else ranked
            if not shown:
                shown = ranked

            st.dataframe(
                [{"Atom": a.label, "Score": a.score, "Rating": a.badge,
                  "Why": a.headline} for a in shown],
                hide_index=True, use_container_width=True, height=210)

            idx_options = [a.atom_idx for a in shown]
            current = selected_el["atom_idx"] if selected_el else None
            start = idx_options.index(current) if current in idx_options else 0
            by_idx = {a.atom_idx: a for a in shown}

            pick = st.selectbox(
                "Attachment atom", idx_options, index=start,
                format_func=lambda i: by_idx[i].describe())
            choice = by_idx[pick]

            colour = STRENGTH_COLOR[choice.band]
            st.markdown(badge(f"{choice.badge} · {choice.score}/100", colour,
                              colour + "18") + f" &nbsp; {choice.headline}",
                        unsafe_allow_html=True)
            for reason in choice.reasons:
                st.caption("· " + reason)
            if choice.band == attach.UNSUITABLE:
                st.warning("This atom cannot carry the bond. gnina will either "
                           "refuse the pattern or produce a hypervalent adduct.",
                           icon="⚠️")

            if st.button("Use this atom", use_container_width=True,
                         disabled=choice.band == attach.UNSUITABLE):
                if choice.detected:
                    # Already a library warhead — keep its chemistry rather than
                    # flattening it to a manual pick, so the bound-form transform
                    # still runs.
                    select_electrophile(session, choice.electrophile)
                else:
                    select_electrophile(
                        session, attach.manual_electrophile(loaded.mol, pick))
                st.rerun()

            if manual_now:
                st.info("This atom was chosen by hand. The ligand is docked as "
                        "supplied, so it must already be in its bound form — "
                        "untick **Build the bound form automatically** in step 4 "
                        "if it is not already off.", icon="✋")

    with right:
        st.markdown("##### Ligand")
        candidate_atoms = [e.atom_idx for e, _ in rows]
        labels = {e.atom_idx: e.label for e, _ in rows}
        molecule_2d.render_molecule_2d(
            loaded.mol,
            candidate_atoms=candidate_atoms,
            selected_atom=selected_el["atom_idx"] if selected_el else None,
            labels=labels,
            title=loaded.name,
        )
        with st.expander("Atom environment of the selected atom"):
            if selected_el:
                env = lig_chem.atom_environment(loaded.mol,
                                                selected_el["atom_idx"])
                st.json(env, expanded=False)
            else:
                st.caption("Nothing selected yet.")


# ═════════════════════════════════════════════════════════════════════════════
#  Step 4 — chemistry and command
# ═════════════════════════════════════════════════════════════════════════════

def step_chemistry(session: WorkSession, settings: Dict) -> None:
    nu_state = session.get("nucleophile")
    el_state = session.get("electrophile")
    if not (nu_state and el_state):
        st.info("Choose a protein atom (step 2) and a ligand atom (step 3).")
        return

    loaded = ligand_mol(session.get("ligand_text"), session.get("ligand_format"),
                        session.get("ligand_name"))

    st.markdown("#### The covalent pair")
    cols = st.columns(5)
    cols[0].metric("Protein atom", nu_state["label"])
    cols[1].metric("Ligand atom", el_state["label"])
    cols[2].metric("Warhead", el_state["warhead_label"])
    cols[3].metric("Bond", f"{nu_state['atom'][0]}–{el_state['element']}")
    cols[4].metric("Bond order", str(el_state["bond_order"]))
    st.caption(f"{el_state['mechanism_label']} · "
               f"{'reversible' if el_state['reversible'] else 'irreversible'} · "
               "candidate pair, not a prediction that the reaction occurs")

    # ── bound form ───────────────────────────────────────────────────────────
    st.markdown("#### Bound form")
    st.caption("gnina does not model the reaction: it needs the ligand as it "
               "looks after the bond forms. "
               f"For this warhead that means: "
               f"{bf_mod.transform_description(el_state['transform'])}.")

    use_bound = st.checkbox("Build the bound form automatically", value=True,
                            help="Turn this off only if the ligand you uploaded "
                                 "is already the reacted form.")

    with st.spinner("Building the bound form…"):
        if use_bound:
            bound = bf_mod.build_bound_form(
                loaded.mol, el_state["atom_idx"], el_state["warhead"],
                el_state["transform"], el_state.get("leaving_atom"),
                embed=True, random_seed=settings["seed"])
        else:
            mol, ok, message = bf_mod.embed_3d(loaded.mol, settings["seed"])
            bound = bf_mod.BoundForm(mol=mol, reactive_idx=el_state["atom_idx"],
                                     changed=False,
                                     description="Using the ligand as supplied.",
                                     smiles=session.get("ligand_text", ""),
                                     embedded=ok,
                                     warnings=[message] if message else [])

    st.markdown(f"**{bound.description}**")
    st.code(bound.smiles, language=None)
    for warning in bound.warnings:
        st.caption("⚠ " + warning)

    # The depiction gets the full width: a drug-sized ligand squeezed into half
    # a column is a tangle in which no individual atom can be read, and the
    # attachment point is the one thing the user has to be able to check here.
    opt_a, opt_b, _ = st.columns([1, 1, 2])
    label_all = opt_a.checkbox(
        "Label every atom", value=True,
        help="Names follow the rest of the app: the deposited PDB name where "
             "the ligand has one, otherwise the '#'-marked index form.")
    show_h = opt_b.checkbox(
        "Show hydrogens", value=False,
        help="Hydrogens are hidden by default — the covalent bond is made by a "
             "heavy atom, and 3-D labels are unreadable with them on.")

    # Name the reactive atom from the bound form, not from el_state: a transform
    # that drops a leaving group renumbers the atoms, so the label picked on the
    # molecule as supplied can belong to a different atom by now.
    all_names = lig_chem.heavy_atom_labels(bound.mol)
    atom_names = all_names if label_all else {}
    reactive_name = all_names.get(bound.reactive_idx, el_state["label"])

    tab_3d, tab_2d = st.tabs(["3-D", "2-D"])
    with tab_3d:
        block = bf_mod.sdf_text(bound, random_seed=settings["seed"])
        if block:
            viewer.render_molecule(
                block, height=470, highlight_atom=bound.reactive_idx,
                label=f"{reactive_name} · attachment point",
                atom_labels=atom_names, show_hydrogens=show_h)
            st.caption("Drag to rotate, scroll to zoom. The highlighted atom is "
                       "the one gnina bonds to the protein.")
        else:
            st.info("No 3-D coordinates could be generated for this ligand. "
                    "The 2-D tab still shows the attachment point.")
    with tab_2d:
        # With every atom named, a long note on the reactive atom collides with
        # its neighbours' labels. The halo already marks it, so name it in the
        # legend instead, where nothing can overlap it.
        molecule_2d.render_molecule_2d(
            bound.mol, selected_atom=bound.reactive_idx,
            labels={} if label_all
                    else {bound.reactive_idx: f"{reactive_name} attachment point"},
            number_atoms=label_all,
            title=f"attachment point: {reactive_name}",
            width=1100, height=700)
        st.caption("The shaded atom is the one gnina bonds to the protein.")

    # ── SMARTS ───────────────────────────────────────────────────────────────
    st.markdown("#### Ligand atom pattern")
    with st.expander("Advanced · how the pattern is built, and overrides"):
        min_radius = st.slider(
            "Chemical context (bonds around the atom)", 1, 5, 3,
            help="The pattern grows past this until it matches one atom.")
        explicit_h = st.checkbox("Pin hydrogen counts", value=True)
        manual = st.text_input("Override the pattern", value="",
                               placeholder="[$([CH3][CH2][C](=O))]")

    result = smarts_mod.generate_smarts(
        bound.mol, bound.reactive_idx, el_state["warhead"],
        min_radius=min_radius, explicit_h=explicit_h)
    pattern = manual.strip() or result.smarts

    if manual.strip():
        check = smarts_mod.verify_smarts(bound.mol, pattern, bound.reactive_idx)
        if not check["valid"]:
            st.error(check["error"])
        elif not check["matches"]:
            st.error("That pattern matches nothing in the bound ligand.")
        elif check["hits_target"] is False:
            st.warning("That pattern matches the ligand but not the atom you "
                       "picked.", icon="⚠️")
        else:
            st.success(f"Override accepted · matches {check['n_matches']} atom(s)")

    st.code(pattern, language=None)
    if result.unique and not manual.strip():
        st.caption("Matches only the selected atom.")
    elif not manual.strip():
        for warning in result.warnings:
            st.caption("⚠ " + warning)

    with st.expander("Other patterns the generator considered"):
        st.caption(f"Chosen by: {result.strategy}")
        for strategy, alt, n in result.alternatives:
            st.markdown(f"`{alt}` — {strategy}, {n} match(es)")

    # ── prepare and run ──────────────────────────────────────────────────────
    st.markdown("#### Files and command")
    box_centre = tuple(nu_state["xyz"])

    # Off by default: this changes what is docked, and every result produced so
    # far came from the single arbitrary configuration.
    dock_both = False
    if bound.stereo_ambiguous:
        dock_both = st.checkbox(
            "Dock both configurations of the new stereocentre", value=False,
            help="The transform made the reactive atom a stereocentre and "
                 "cannot know which face the protein attacked. With this on, "
                 "both diastereomers are written to the ligand file and gnina "
                 "scores each. Doubles the search.")

    if st.button("Prepare receptor and ligand", type="secondary"):
        with st.spinner("Writing files…"):
            rec = prepare_receptor.prepare_receptor(
                session.read_text(PROTEIN_FILE),
                session.path(RECEPTOR_FILE),
                remove_waters=True, remove_hydrogens=True,
                selector=nu_state["selector"],
                title=session.get("protein_name", ""))
            # The ligand is embedded in its own frame, so it has to be brought
            # to the covalent site before gnina sees it — otherwise the search
            # box centred here simply does not contain it.
            structure = parse_protein(session.read_text(PROTEIN_FILE))
            protein_xyz = [a.xyz for a in structure.atoms
                           if a.resname not in WATERS]
            # Untouched unless the user asked for both: same molecule, same
            # single conformer, same result as before.
            dock_mol, n_configs = bound.mol, 1
            if dock_both:
                dock_mol, n_configs = bf_mod.both_epimers(
                    bound, random_seed=settings["seed"])
            lig = prepare_ligand.prepare_ligand(
                dock_mol, session.path(LIGAND_FILE),
                name=session.get("ligand_name", "ligand"),
                reactive_idx=bound.reactive_idx, random_seed=settings["seed"],
                anchor_xyz=box_centre,
                anchor_element=NUCLEOPHILE_ELEMENT.get(nu_state["type"], "S"),
                protein_xyz=protein_xyz)
            if dock_both and n_configs < 2:
                lig.warnings.append(
                    "Only one configuration could be embedded, so the ligand "
                    "file holds a single diastereomer after all.")
            check = prepare_ligand.verify_pattern_against_file(lig.path, pattern)
        placement = lig.placement
        session.update(receptor_prepared=rec.path, ligand_prepared=lig.path,
                       bound_reactive_idx=lig.reactive_idx,
                       smarts=pattern, pattern_check=check,
                       rec_report=rec.report(), lig_report=lig.report(),
                       rec_warnings=rec.warnings, lig_warnings=lig.warnings,
                       placement_note=placement.note if placement else "",
                       placement_distance=(placement.final_distance
                                           if placement else None),
                       required_box=(placement.required_box
                                     if placement else None))
        st.rerun()

    if session.get("receptor_prepared"):
        col_a, col_b = st.columns(2)
        col_a.caption("Receptor · " + session.get("rec_report", ""))
        col_b.caption("Ligand · " + session.get("lig_report", ""))
        for warning in (session.get("rec_warnings") or []):
            st.error(warning)
        for warning in (session.get("lig_warnings") or []):
            st.warning(warning)

        if session.get("placement_note"):
            st.caption("📍 " + session.get("placement_note")
                       + " gnina searches from there; it is a starting point, "
                       "not a predicted pose.")

        check = session.get("pattern_check") or {}
        if check.get("ok"):
            st.success(f"The pattern matches {check['n_matches']} atom(s) in the "
                       "written ligand file.", icon="✅")
        elif check:
            st.error(f"Pattern check failed: {check.get('error')}")

    # A box that does not hold the placed ligand is the failure that produces
    # "ligand outside box" and a score table that looks fine but is not.
    box_size = settings["box_size"]
    required = session.get("required_box")
    if required and required > box_size:
        if settings["autofit_box"]:
            st.info(f"Box grown from {box_size} Å to {required:.0f} Å so it "
                    "holds the placed ligand.", icon="📦")
            box_size = required
        else:
            st.warning(f"The placed ligand needs about {required:.0f} Å of box "
                       f"to swing in, but the box is {box_size} Å. gnina will "
                       "report 'ligand outside box' and its scores will not "
                       "mean anything.", icon="⚠️")

    config = gnina.GninaConfig(
        receptor=session.get("receptor_prepared") or session.path(RECEPTOR_FILE),
        ligand=session.get("ligand_prepared") or session.path(LIGAND_FILE),
        out=session.path(POSES_FILE),
        out_flex=session.path(FLEX_FILE),
        full_flex_output=True,
        covalent_rec_atom=nu_state["selector"],
        covalent_lig_atom_pattern=session.get("smarts", pattern),
        covalent_bond_order=el_state["bond_order"],
        covalent_optimize_lig=settings["optimize_lig"],
        center=box_centre, box_size=box_size,
        exhaustiveness=settings["exhaustiveness"],
        num_modes=settings["num_modes"], seed=settings["seed"],
        cnn_scoring=settings["cnn_scoring"], no_gpu=settings["no_gpu"],
        log=session.path("gnina.log"),
    )
    st.code(config.command_string(), language="bash")

    ready = bool(session.get("receptor_prepared") and session.get("ligand_prepared"))
    run_col, note_col = st.columns([1, 3])
    with run_col:
        run_clicked = st.button("Run docking", type="primary", disabled=not ready)
    with note_col:
        if not ready:
            st.caption("Prepare the files first.")
        elif gnina.find_binary() is None:
            st.caption("gnina is not installed here — copy the command above and "
                       "run it where it is.")

    if run_clicked:
        log_box = st.empty()
        lines: List[str] = []

        def on_line(line: str) -> None:
            lines.append(line)
            log_box.code("\n".join(lines[-14:]), language=None)

        with st.spinner("Docking…"):
            result = gnina.run(config, on_line=on_line)
        session.update(gnina_done=result.ok, gnina_error=result.error,
                       gnina_seconds=round(result.seconds, 1),
                       gnina_command=result.command, n_poses=result.n_poses)
        # New poses: the multi-model file built from the previous run no longer
        # describes them, so it must not stay on offer.
        session.clear("complex_models_path")
        if result.ok:
            st.success(f"{result.n_poses} pose(s) in {result.seconds:.1f} s. "
                       "Open **5 · Results**.", icon="✅")
        else:
            st.error(result.error or "gnina failed.")
            for tip in gnina.explain_failure(result):
                st.caption("· " + tip)


# ═════════════════════════════════════════════════════════════════════════════
#  Step 5 — results
# ═════════════════════════════════════════════════════════════════════════════

def step_results(session: WorkSession, settings: Dict) -> None:
    poses_path = session.path(POSES_FILE)
    if not os.path.isfile(poses_path):
        st.info("No results yet. Run the docking in step 4, or drop a gnina "
                "output SDF into the run directory.")
        uploaded = st.file_uploader("Load an existing gnina output", type=["sdf"])
        if uploaded is not None:
            session.write_bytes(POSES_FILE, uploaded.getvalue())
            st.rerun()
        return

    poses = scores_mod.read_poses(poses_path)
    if not poses:
        st.error("The result file has no poses in it.")
        return

    summary = scores_mod.summarise(poses)
    cols = st.columns(4)
    cols[0].metric("Poses", summary["n_poses"])
    # `is not None`, not truthiness: a CNN score of 0.000 is a real score and
    # an affinity of 0.00 is a real (bad) one, and both would read as missing.
    cols[1].metric("Best affinity",
                   f"{summary['best_affinity']:.2f}"
                   if summary["best_affinity"] is not None else "—",
                   help="kcal/mol; more negative is stronger")
    cols[2].metric("Mean affinity",
                   f"{summary['mean_affinity']:.2f}"
                   if summary["mean_affinity"] is not None else "—")
    cols[3].metric("Best CNN score",
                   f"{summary['best_cnn']:.3f}"
                   if summary["best_cnn"] is not None else "—")

    rms = poses_rmsd(poses_path, os.path.getmtime(poses_path))
    pose_aff = {p.number(): p.affinity for p in poses}
    rows = scores_mod.pose_table(poses)
    if rms.ok and len(rms.matrix) == len(rows):
        for i, row in enumerate(rows):
            row["RMSD to pose 1 (Å)"] = round(rms.matrix[0][i], 2)
    st.dataframe(rows, use_container_width=True, hide_index=True)
    # The RMSD column above is the only place these numbers now appear, so its
    # caveats belong here — an unreliable value must not read as a clean one.
    for warning in rms.warnings:
        st.warning(warning)

    export_a, export_b, export_c = st.columns(3)
    with export_a:
        with open(poses_path, "rb") as fh:
            st.download_button("Download poses (SDF)", fh.read(),
                               file_name="poses.sdf", use_container_width=True)
    with export_b:
        if st.button("Export scores (CSV)", use_container_width=True):
            path = scores_mod.write_csv(
                rows, session.path("scores.csv"),
                header_note=f"{session.get('protein_name')} / "
                            f"{session.get('ligand_name')} / "
                            f"{(session.get('nucleophile') or {}).get('selector')}")
            with open(path, "rb") as fh:
                st.download_button("Save scores.csv", fh.read(),
                                   file_name="scores.csv",
                                   use_container_width=True)
    with export_c:
        if st.button("Export scores (Excel)", use_container_width=True):
            try:
                path = scores_mod.write_xlsx(rows, session.path("scores.xlsx"))
            except ImportError:
                st.error("openpyxl is not installed: pip install openpyxl")
            else:
                with open(path, "rb") as fh:
                    st.download_button("Save scores.xlsx", fh.read(),
                                       file_name="scores.xlsx",
                                       use_container_width=True)

    # The per-reference RMSD table and the full pose-vs-pose matrix used to be
    # shown here. Both are gone: the pose table above already carries an "RMSD
    # to pose 1" column, and a second table of the same numbers only competes
    # with it. analysis/rmsd.py still computes the whole matrix — to_reference()
    # and write_matrix_csv() are untouched — so putting either back is a display
    # change, not a rebuild.

    # ── One pose at a time ───────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Pose viewer")
    nu_state = session.get("nucleophile") or {}
    pose_no = st.radio(
        "Show pose", list(range(1, len(poses) + 1)), horizontal=True,
        key="view_pose",
        format_func=lambda n: f"Pose {n}" + (f" · {pose_aff[n]:.2f}"
                                             if pose_aff.get(n) is not None else ""))
    pose = poses[pose_no - 1]

    # Poses come from poses.sdf (gnina -o). The --out_flex file holds receptor
    # residues, never the ligand, and is not read here.
    receptor = session.get("receptor_prepared") or session.path(RECEPTOR_FILE)
    if not os.path.isfile(receptor):
        receptor = session.path(PROTEIN_FILE)

    merged = None
    try:
        merged = complex_builder.merge_complex(
            receptor, poses_path, session.path(f"complex_pose{pose_no}.pdb"),
            pose_index=pose_no - 1,
            covalent_rec_atom=nu_state.get("selector"),
            lig_smarts=session.get("smarts"))
        complex_builder.write_pymol_script(
            merged.path, session.path(f"view_pose{pose_no}.pml"), merged.bonds,
            pose_numbers=[pose_no])
    except Exception as exc:                                      # noqa: BLE001
        st.error(f"Could not build the complex for pose {pose_no}: {exc}")

    bond = merged.bond if merged else None
    if merged:
        if session.get("complex_pose") != pose_no:
            # The PLIP profile belongs to the previously shown pose.
            session.clear("plip_ok", "plip_error", "plip_summary",
                          "plip_contacts", "plip_text")
        if bond:
            session.update(bond_distance=bond.distance, bond_lig_atom=bond.lig_atom,
                           bond_rec_xyz=list(bond.rec_xyz),
                           bond_lig_xyz=list(bond.lig_xyz))
        else:
            session.clear("bond_distance", "bond_lig_atom", "bond_rec_xyz",
                          "bond_lig_xyz")
        session.update(complex_path=merged.path, complex_pose=pose_no)

    cols = st.columns(4)
    cols[0].metric("Affinity (kcal/mol)",
                   f"{pose.affinity:.2f}" if pose.affinity is not None else "—")
    cols[1].metric("CNN score",
                   f"{pose.cnn_score:.3f}" if pose.cnn_score is not None else "—")
    cols[2].metric("RMSD to pose 1 (Å)",
                   f"{rms.matrix[0][pose_no - 1]:.2f}" if rms.ok else "—")
    cols[3].metric(f"{bond.rec_atom}–{bond.lig_atom} bond" if bond
                   else "Covalent bond",
                   f"{bond.distance:.2f} Å" if bond else "—")
    for warning in (merged.warnings if merged else []):
        st.warning(warning)

    pose_sdf = complex_builder.extract_sdf_block(poses_path, pose_no - 1)
    cov = None
    if bond:
        cov = {"prot": list(bond.rec_xyz), "lig": list(bond.lig_xyz),
               "label": f"{bond.rec_atom}–{bond.lig_atom}  {bond.distance:.2f} Å"}
    marker = None
    if nu_state:
        marker = {"chain": nu_state["chain"], "resi": nu_state["resseq"],
                  "atom": nu_state["atom"], "label": nu_state["label"]}
    with open(receptor) as fh:
        receptor_text = fh.read()
    # The receptor and the ONE selected pose are drawn as separate models; the
    # ligand's bonds come straight from the SDF, not from a guessed PDB.
    viewer.render_structure(receptor_text, selected=marker, ligand_sdf=pose_sdf,
                            covalent=cov, style="cartoon", height=480)
    st.caption(f"Only pose {pose_no} is drawn. Solid orange = covalent bond, "
               "measured from the receptor coordinates you docked into.")

    if merged:
        dl_a, dl_b, dl_c = st.columns(3)
        with open(merged.path) as fh:
            dl_a.download_button(f"Pose {pose_no} complex (PDB)", fh.read(),
                                 file_name=f"complex_pose{pose_no}.pdb",
                                 use_container_width=True)
        pml_text = session.read_text(f"view_pose{pose_no}.pml")
        if pml_text:
            dl_b.download_button("PyMOL script (.pml)", pml_text,
                                 file_name=f"view_pose{pose_no}.pml",
                                 use_container_width=True,
                                 help="Save next to the PDB, then run: "
                                      f"pymol view_pose{pose_no}.pml")
        dl_c.download_button(f"Pose {pose_no} ligand (SDF)", pose_sdf + "\n$$$$\n",
                             file_name=f"pose{pose_no}.sdf",
                             use_container_width=True)

        # ── all poses, one MODEL each ────────────────────────────────────────
        st.markdown(f"**All {len(poses)} poses in one file**")
        st.caption("One MODEL per pose — receptor plus that pose, each with its "
                   "own LINK and CONECT. PyMOL and Chimera read the models as "
                   "states you step through, rather than all poses overlaid in "
                   "a single frame.")
        if st.button(f"Build all {len(poses)} poses as a multi-model PDB",
                     use_container_width=True):
            with st.spinner(f"Merging {len(poses)} poses…"):
                try:
                    many = complex_builder.merge_complex_models(
                        receptor, poses_path, session.path("complex_all_poses.pdb"),
                        covalent_rec_atom=nu_state.get("selector"),
                        lig_smarts=session.get("smarts"))
                except Exception as exc:                      # noqa: BLE001
                    many = None
                    st.error(f"Could not build the multi-model complex: {exc}")
            if many is not None:
                session.set("complex_models_path", many.path)
                bonded = len(many.bonds)
                st.success(f"{many.n_poses} model(s) written; "
                           f"{bonded} carry a covalent bond.", icon="✅")
                for warning in many.warnings[:6]:
                    st.caption("⚠ " + warning)

        models_path = session.get("complex_models_path")
        if models_path and os.path.isfile(models_path):
            with open(models_path) as fh:
                st.download_button("Download all poses (multi-model PDB)",
                                   fh.read(),
                                   file_name="complex_all_poses.pdb",
                                   use_container_width=True)

    complex_path = merged.path if merged else None
    if complex_path and os.path.isfile(complex_path):
        with open(complex_path) as fh:
            complex_text = fh.read()

        def covalent_marker():
            """The covalent bond, for the PLIP view below."""
            return cov

        st.markdown("#### Interactions")
        if not plip_mod.available():
            st.caption("PLIP is not installed here. `pip install plip` to profile "
                       "the complex, or download the PDB and run it elsewhere.")
        else:
            if st.button("Run PLIP"):
                with st.spinner("Profiling…"):
                    report = plip_mod.run(complex_path, session.path("plip"))
                session.update(
                    plip_ok=report.ok, plip_error=report.error,
                    plip_summary=plip_mod.summarise(report) if report.ok else [],
                    plip_contacts=[c.as_dict() for c in report.contacts],
                    plip_text=report.text[:20000] if report.ok else "")
                st.rerun()

            if session.get("plip_ok"):
                contacts = session.get("plip_contacts") or []
                if contacts:
                    show_labels = st.checkbox("Label each interaction with its "
                                              "distance", value=True)
                    payload = viewer.contact_payload(contacts,
                                                     show_labels=show_labels)
                    marker = None
                    if nu_state:
                        marker = {"chain": nu_state["chain"],
                                  "resi": nu_state["resseq"],
                                  "atom": nu_state["atom"],
                                  "label": nu_state["label"]}
                    bond = covalent_marker()
                    viewer.render_structure(complex_text, selected=marker,
                                            style="cartoon", contacts=payload,
                                            covalent=bond, height=460)
                    st.markdown(
                        viewer.interaction_legend((c["kind"] for c in payload),
                                                  covalent=bond is not None),
                        unsafe_allow_html=True)
                    st.caption(
                        f"{len(payload)} PLIP interaction(s), dashed. The solid "
                        "orange bond is the covalent one — PLIP profiles "
                        "non-covalent contacts only, so it comes from the "
                        "complex geometry, not from the report."
                        if bond else
                        f"{len(payload)} PLIP interaction(s) drawn.")
                else:
                    st.caption("PLIP found no interactions with coordinates to "
                               "draw. The text report below still applies.")

                for line in session.get("plip_summary") or []:
                    st.text(line)
                with st.expander("Full report"):
                    st.text(session.get("plip_text") or "")
            elif session.get("plip_error"):
                st.error(session.get("plip_error"))


# ═════════════════════════════════════════════════════════════════════════════
#  Step 6 — batch / multi-ligand docking
# ═════════════════════════════════════════════════════════════════════════════

def step_batch(session: WorkSession, settings: Dict) -> None:
    if not session.has(PROTEIN_FILE):
        st.info("Upload a protein in step 1 first.")
        return

    nu_state = session.get("nucleophile")
    if not nu_state:
        st.info("Select a protein nucleophile in step 2 first.")
        return

    st.markdown(f"**Target**: {nu_state['label']} "
                f"(`{nu_state['selector']}`)")

    # ── Ligand input ──
    st.markdown("#### Ligands")
    input_method = st.radio(
        "How to provide ligands", ["Upload SDF file(s)", "Enter SMILES"],
        horizontal=True, label_visibility="collapsed")

    ligand_jobs: List[batch_mod.LigandJob] = []

    if input_method == "Upload SDF file(s)":
        uploaded = st.file_uploader(
            "Upload SDF files (each can contain multiple molecules)",
            type=["sdf"], accept_multiple_files=True)
        if uploaded:
            for f in uploaded:
                tmp_path = session.path(f"batch_{f.name}")
                session.write_bytes(f"batch_{f.name}", f.getvalue())
                jobs = batch_mod.load_ligands_from_sdf(tmp_path)
                ligand_jobs.extend(jobs)
            st.success(f"Loaded {len(ligand_jobs)} ligand(s) from "
                       f"{len(uploaded)} file(s).")
    else:
        smiles_text = st.text_area(
            "Enter SMILES (one per line, optionally: SMILES name)",
            height=200,
            placeholder="C=CC(=O)N1CCCCC1 acryloylpiperidine\n"
                        "C=CC(=O)Nc1ccccc1 acrylanilide\n"
                        "ClCC(=O)NCc1ccccc1 chloroacetamide_benzyl")
        if smiles_text.strip():
            ligand_jobs = batch_mod.load_ligands_from_smiles(
                smiles_text.strip().split("\n"))
            st.success(f"Parsed {len(ligand_jobs)} ligand(s).")

    if not ligand_jobs:
        st.caption("No ligands loaded yet.")
        return

    # Show preview table
    with st.expander(f"Preview ({len(ligand_jobs)} ligands)"):
        preview = [{"#": i + 1, "Name": j.name,
                     "SMILES": j.smiles or "(from file)"}
                   for i, j in enumerate(ligand_jobs)]
        st.dataframe(preview, use_container_width=True, hide_index=True)

    # ── Settings ──
    st.markdown("#### Docking settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        batch_exhaustiveness = st.number_input(
            "Exhaustiveness", min_value=1, max_value=64,
            value=settings["exhaustiveness"], key="batch_exh")
    with col2:
        batch_num_modes = st.number_input(
            "Num. modes", min_value=1, max_value=50,
            value=settings["num_modes"], key="batch_modes")
    with col3:
        batch_box = st.number_input(
            "Box size (Å)", min_value=10.0, max_value=60.0,
            value=float(settings["box_size"]), key="batch_box")

    # ── Run ──
    if st.button("🚀 Run batch docking", type="primary",
                  use_container_width=True):
        pdb_text = session.read_text(PROTEIN_FILE)
        structure = parse_protein(pdb_text)

        # Reconstruct Nucleophile from session state
        nuc_atom = prot_chem.find_atom(
            structure, nu_state["chain"], nu_state["resseq"],
            nu_state["atom"])
        nucleophiles = prot_chem.find_nucleophiles(structure)
        nuc = None
        for n in nucleophiles:
            if n.selector == nu_state["selector"]:
                nuc = n
                break
        if nuc is None:
            st.error(f"Could not find nucleophile {nu_state['selector']} "
                     "in the structure.")
            return

        batch_dir = session.path("batch")
        progress_bar = st.progress(0, text="Starting batch docking…")
        status_text = st.empty()
        live_table = st.empty()
        collected: List[batch_mod.LigandResult] = []

        def on_progress(current, total, msg):
            # total is 0 if the batch is empty, and Streamlit rejects a
            # fraction outside 0-1.
            fraction = min(1.0, max(0.0, current / total)) if total else 0.0
            progress_bar.progress(fraction, text=msg)

        def on_result(res):
            collected.append(res)
            rows = batch_mod.results_table(collected)
            live_table.dataframe(rows, use_container_width=True,
                                 hide_index=True)

        results = batch_mod.run_batch(
            ligand_jobs,
            pdb_text=pdb_text,
            structure=structure,
            nucleophile=nuc,
            base_dir=batch_dir,
            box_size=batch_box,
            exhaustiveness=batch_exhaustiveness,
            num_modes=batch_num_modes,
            seed=settings["seed"],
            cnn_scoring=settings["cnn_scoring"],
            no_gpu=settings["no_gpu"],
            on_progress=on_progress,
            on_result=on_result,
        )

        progress_bar.progress(1.0, text="Done!")

        # Save results to session
        session.set("batch_results",
                    [r.summary_row() for r in results])

        # Show final ranked table
        ranked = batch_mod.rank_results(results)
        final_rows = batch_mod.results_table(ranked)
        st.markdown("#### Results (ranked by affinity)")
        st.dataframe(final_rows, use_container_width=True, hide_index=True)

        # Summary stats
        done = [r for r in results if r.status == "done"]
        failed = [r for r in results if r.status == "failed"]
        st.markdown(
            f"**{len(done)}** succeeded, **{len(failed)}** failed "
            f"out of {len(results)} ligands.")

        # Download CSV of all results
        if done:
            import csv
            import io
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=final_rows[0].keys())
            w.writeheader()
            w.writerows(final_rows)
            st.download_button(
                "Download batch results (CSV)",
                buf.getvalue(),
                file_name="batch_results.csv",
                mime="text/csv")

    # Show saved results if available
    saved = session.get("batch_results")
    if saved and not st.session_state.get("_batch_running"):
        st.markdown("#### Previous batch results")
        st.dataframe(saved, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    session = get_session()
    settings = sidebar(session)
    if session.has(PROTEIN_FILE):
        handle_residue_click(session, session.read_text(PROTEIN_FILE))

    st.markdown("## Covalent docking setup")
    status_strip(session)

    tabs = st.tabs(["1 · Upload", "2 · Protein site", "3 · Ligand warhead",
                    "4 · Chemistry & command", "5 · Results", "6 · Batch"])
    with tabs[0]:
        step_upload(session)
    with tabs[1]:
        step_protein(session, settings)
    with tabs[2]:
        step_ligand(session, settings)
    with tabs[3]:
        step_chemistry(session, settings)
    with tabs[4]:
        step_results(session, settings)
    with tabs[5]:
        step_batch(session, settings)

    st.divider()
    st.caption("Candidate covalent sites are chemically plausible pairings. "
               "Whether a reaction happens depends on geometry, accessibility, "
               "protonation and electronic activation — none of which this "
               "filter establishes.")


if __name__ == "__main__":
    prune(keep=st.query_params.get("s"))
    main()
