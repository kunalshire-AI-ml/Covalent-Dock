"""
viewer.py — 3-D structure panel built on 3Dmol.js.

Shows the receptor, puts a sphere on every candidate nucleophile, highlights
the selected atom, and — once docking is done — overlays the pose and draws the
covalent bond as a dashed line.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

CDN = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"

COLORS = {
    "selected": "#4f57c4",
    "candidate": "#e0a92a",
    "ligand": "#2ca089",
    "bond": "#c2410c",
}

# One colour per PLIP interaction type. Chosen to stay distinguishable against
# the spectrum cartoon and against each other; the covalent bond keeps its own
# burnt orange so it never reads as just another contact.
INTERACTION_COLORS = {
    "Hydrogen Bonds": "#2563eb",
    "Hydrophobic Interactions": "#94a3b8",
    "Water Bridges": "#7c3aed",
    "Salt Bridges": "#dc2626",
    "pi-Stacking": "#16a34a",
    "pi-Cation Interactions": "#d97706",
    "Halogen Bonds": "#0891b2",
    "Metal Complexes": "#be185d",
}
INTERACTION_FALLBACK = "#64748b"

# Carbon colour per docked pose (pose N -> entry (N-1) % 6). Order matches the
# PyMOL script from complex_builder.write_pymol_script. Non-carbon atoms keep
# their element colours.
POSE_PALETTE = [0x22C55E, 0x06B6D4, 0xD946EF, 0xF97316, 0x8B5CF6, 0x3B82F6]
POSE_COLOR_NAMES = ["green", "cyan", "magenta", "orange", "purple", "blue"]


def interaction_color(kind: str) -> str:
    return INTERACTION_COLORS.get(kind, INTERACTION_FALLBACK)


def contact_payload(contacts, show_labels: bool = True) -> List[Dict[str, object]]:
    """
    Turn `analysis.plip.Contact` objects (or their dicts) into viewer entries.

    Accepts dicts so the app can round-trip them through the run manifest.
    """
    out: List[Dict[str, object]] = []
    for c in contacts:
        item = c if isinstance(c, dict) else c.as_dict()
        if not (item.get("prot") and item.get("lig")):
            continue
        kind = item.get("kind", "")
        distance = item.get("distance")
        out.append({
            "kind": kind,
            "prot": list(item["prot"]),
            "lig": list(item["lig"]),
            "chain": item.get("chain") or "",
            "resnr": item.get("resnr"),
            "color": interaction_color(kind),
            "label": (f"{distance:.2f} Å" if show_labels
                      and isinstance(distance, (int, float)) else ""),
        })
    return out


def interaction_legend(kinds, covalent: bool = False) -> str:
    """Swatch row naming only the interaction types actually present."""
    seen, bits = set(), []
    if covalent:
        bits.append(
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'margin:0 12px 4px 0;font-size:0.78rem;color:#495057">'
            f'<span style="width:14px;height:0;border-top:3px solid '
            f'{COLORS["bond"]};display:inline-block"></span>'
            f'covalent bond</span>')
    for kind in kinds:
        if kind in seen:
            continue
        seen.add(kind)
        bits.append(
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'margin:0 12px 4px 0;font-size:0.78rem;color:#495057">'
            f'<span style="width:14px;height:0;border-top:2px dashed '
            f'{interaction_color(kind)};display:inline-block"></span>{kind}</span>'
        )
    return ('<div style="margin:6px 0 2px 0;display:flex;flex-wrap:wrap">'
            + "".join(bits) + "</div>") if bits else ""


def structure_viewer_html(
    pdb_text: str,
    selected: Optional[Dict[str, object]] = None,
    candidates: Sequence[Dict[str, object]] = (),
    ligand_sdf: Optional[str] = None,
    height: int = 460,
    style: str = "cartoon",
    surface: bool = False,
    spin: bool = False,
    contacts: Sequence[Dict[str, object]] = (),
    contact_residues: bool = True,
    covalent: Optional[Dict[str, object]] = None,
    covalents: Sequence[Dict[str, object]] = (),
    pose_colors: bool = False,
) -> str:
    """
    `selected` and each entry of `candidates` are {"chain","resi","atom"} dicts;
    `ligand_sdf` is the text of one pose.

    `contacts` are PLIP interactions from `contact_payload()`: each is drawn as
    a dashed line between its protein and ligand points, coloured by type. With
    `contact_residues` the residues at the protein end are shown as sticks too,
    so an interaction reads as being with a side chain rather than with a
    floating point in space.

    `covalents` is a list of {"prot", "lig", "label"} bonds, one per pose, each
    drawn solid; `pose_colors` gives every LIG residue (one per pose) its own
    carbon colour so a multi-pose complex can be told apart.
    """
    payload = {
        "pdb": pdb_text,
        "selected": selected,
        "candidates": list(candidates),
        "ligand": ligand_sdf,
        "style": style,
        "surface": surface,
        "spin": spin,
        "colors": COLORS,
        "contacts": list(contacts),
        "contactResidues": contact_residues,
        "covalent": covalent,
        "covalents": list(covalents),
        "poseColors": POSE_PALETTE if pose_colors else None,
    }
    data = json.dumps(payload)

    return f"""
<div id="viewport" style="position:relative;width:100%;height:{height}px;
     border:1px solid rgba(49,51,63,0.10);border-radius:8px;overflow:hidden;"></div>
<script src="{CDN}"></script>
<script>
(function () {{
  const data = {data};
  function start() {{
    if (typeof $3Dmol === "undefined") {{ return setTimeout(start, 120); }}
    const viewer = $3Dmol.createViewer(document.getElementById("viewport"),
                                       {{backgroundColor: "white"}});
    viewer.addModel(data.pdb, "pdb");

    if (data.style === "surface") {{
      viewer.setStyle({{}}, {{cartoon: {{color: "#cfd3e3"}}}});
    }} else if (data.style === "stick") {{
      viewer.setStyle({{}}, {{stick: {{radius: 0.14, colorscheme: "whiteCarbon"}}}});
    }} else {{
      viewer.setStyle({{}}, {{cartoon: {{color: "spectrum", opacity: 0.9}}}});
    }}
    viewer.setStyle({{hetflag: true}}, {{stick: {{radius: 0.16}}}});

    // Highlight the docked ligand in the complex PDB with bold green sticks
    // so it stands out from the protein and other heteroatoms.
    viewer.setStyle({{resn: "LIG"}},
                    {{stick: {{radius: 0.19, colorscheme: "greenCarbon"}}}});

    // Multi-pose complex: pose N is LIG residue N. Recolour each one's carbons.
    if (data.poseColors) {{
      const palette = data.poseColors;
      const seen = {{}};
      viewer.getModel(0).selectedAtoms({{resn: "LIG"}}).forEach(function (a) {{
        seen[a.resi] = true;
      }});
      Object.keys(seen).map(Number).forEach(function (r) {{
        viewer.setStyle({{resn: "LIG", resi: r}}, {{stick: {{radius: 0.19,
          colorfunc: function (atom) {{
            return atom.elem === "C" ? palette[(r - 1) % palette.length]
                 : ($3Dmol.elementColors.Jmol[atom.elem] || 0xcccccc);
          }}}}}});
      }});
    }}

    (data.candidates || []).forEach(function (a) {{
      const sel = {{chain: a.chain, resi: a.resi, atom: a.atom}};
      viewer.addStyle(sel, {{sphere: {{radius: 0.45, color: data.colors.candidate}}}});
    }});

    let focus = null;
    if (data.selected) {{
      const sel = {{chain: data.selected.chain, resi: data.selected.resi,
                    atom: data.selected.atom}};
      viewer.addStyle(sel, {{sphere: {{radius: 0.75, color: data.colors.selected}}}});
      viewer.addStyle({{chain: data.selected.chain, resi: data.selected.resi}},
                      {{stick: {{radius: 0.2, colorscheme: "cyanCarbon"}}}});
      viewer.addLabel(data.selected.label || "", {{
        position: sel, backgroundColor: data.colors.selected,
        backgroundOpacity: 0.85, fontColor: "white", fontSize: 11,
        borderThickness: 0, inFront: true
      }}, sel);
      focus = sel;
    }}

    if (data.ligand) {{
      viewer.addModel(data.ligand, "sdf");
      viewer.setStyle({{model: -1}},
                      {{stick: {{radius: 0.19, colorscheme: "greenCarbon"}}}});
      if (data.selected && !data.covalent && !(data.covalents || []).length) {{
        const atoms = viewer.getModel(-1).selectedAtoms({{}});
        const anchor = viewer.getModel(0).selectedAtoms(
          {{chain: data.selected.chain, resi: data.selected.resi,
            atom: data.selected.atom}});
        if (atoms.length && anchor.length) {{
          let best = atoms[0], bestD = 1e9;
          atoms.forEach(function (at) {{
            const d = Math.hypot(at.x - anchor[0].x, at.y - anchor[0].y,
                                 at.z - anchor[0].z);
            if (d < bestD) {{ bestD = d; best = at; }}
          }});
          viewer.addCylinder({{
            start: {{x: anchor[0].x, y: anchor[0].y, z: anchor[0].z}},
            end: {{x: best.x, y: best.y, z: best.z}},
            radius: 0.12, dashed: true, fromCap: 1, toCap: 1,
            color: data.colors.bond
          }});
          viewer.addLabel(bestD.toFixed(2) + " \\u00C5", {{
            position: {{x: (anchor[0].x + best.x) / 2,
                        y: (anchor[0].y + best.y) / 2,
                        z: (anchor[0].z + best.z) / 2}},
            backgroundColor: "white", backgroundOpacity: 0.85,
            fontColor: data.colors.bond, fontSize: 11, borderThickness: 0,
            inFront: true
          }});
        }}
      }}
    }}

    // PLIP interactions: a dashed line per contact, plus the side chain it
    // comes from, so the contact is attached to something visible.
    (data.contacts || []).forEach(function (c) {{
      if (data.contactResidues && c.resnr !== null && c.resnr !== undefined) {{
        var res = {{resi: c.resnr}};
        if (c.chain) {{ res.chain = c.chain; }}
        viewer.addStyle(res, {{stick: {{radius: 0.13,
                                       colorscheme: "whiteCarbon"}}}});
      }}
      viewer.addCylinder({{
        start: {{x: c.prot[0], y: c.prot[1], z: c.prot[2]}},
        end:   {{x: c.lig[0],  y: c.lig[1],  z: c.lig[2]}},
        radius: 0.055, dashed: true, fromCap: 1, toCap: 1, color: c.color
      }});
      if (c.label) {{
        viewer.addLabel(c.label, {{
          position: {{x: (c.prot[0] + c.lig[0]) / 2,
                      y: (c.prot[1] + c.lig[1]) / 2,
                      z: (c.prot[2] + c.lig[2]) / 2}},
          backgroundColor: "white", backgroundOpacity: 0.75,
          fontColor: c.color, fontSize: 9, borderThickness: 0, inFront: true
        }});
      }}
    }});

    // The covalent bond. PLIP never reports it — it profiles non-covalent
    // contacts — so it is drawn from the complex builder's own geometry, and
    // drawn solid and thicker so it never reads as one more weak contact.
    const bonds = (data.covalents || []).concat(data.covalent ? [data.covalent] : []);
    bonds.forEach(function (cv) {{
      if (!(cv && cv.prot && cv.lig)) {{ return; }}
      viewer.addCylinder({{
        start: {{x: cv.prot[0], y: cv.prot[1], z: cv.prot[2]}},
        end:   {{x: cv.lig[0],  y: cv.lig[1],  z: cv.lig[2]}},
        radius: 0.14, dashed: false, fromCap: 2, toCap: 2,
        color: data.colors.bond
      }});
      if (cv.label) {{
        viewer.addLabel(cv.label, {{
          position: {{x: (cv.prot[0] + cv.lig[0]) / 2,
                      y: (cv.prot[1] + cv.lig[1]) / 2,
                      z: (cv.prot[2] + cv.lig[2]) / 2}},
          backgroundColor: data.colors.bond, backgroundOpacity: 0.9,
          fontColor: "white", fontSize: 11, borderThickness: 0, inFront: true
        }});
      }}
    }});

    if (data.surface) {{
      viewer.addSurface($3Dmol.SurfaceType.VDW,
                        {{opacity: 0.55, color: "#dfe3fd"}}, {{hetflag: false}});
    }}

    // Zoom to the ligand if it is present in the complex, otherwise
    // fall back to the selected nucleophile or the full structure.
    const ligAtoms = viewer.getModel(0).selectedAtoms({{resn: "LIG"}});
    if (data.ligand) {{
      viewer.zoomTo({{model: -1}});
      viewer.zoom(0.7);
    }} else if (ligAtoms.length) {{
      viewer.zoomTo({{resn: "LIG"}});
      viewer.zoom(0.6);
    }} else if (focus) {{ viewer.zoomTo(focus); viewer.zoom(0.45); }}
    else {{ viewer.zoomTo(); }}
    if (data.spin) {{ viewer.spin("y", 0.4); }}
    viewer.render();
  }}
  start();
}})();
</script>
"""


def render_structure(pdb_text: str, height: int = 460, **kwargs) -> None:
    import streamlit.components.v1 as components

    components.html(structure_viewer_html(pdb_text, height=height, **kwargs),
                    height=height + 12)


def nucleophile_marker(nu) -> Dict[str, object]:
    """Turn a Nucleophile into the dict the viewer expects."""
    return {"chain": nu.chain, "resi": nu.resseq, "atom": nu.atom_name,
            "label": nu.label}


def molecule_viewer_html(sdf_text: str, highlight_atom: Optional[int] = None,
                         height: int = 320, label: str = "",
                         atom_labels: Optional[Dict[int, str]] = None,
                         show_hydrogens: bool = False) -> str:
    """
    The ligand alone, with the reactive atom marked.

    `atom_labels` maps atom index to the text to float beside it. The caller
    supplies the text rather than the viewer deriving it, so the names here are
    the same ones the 2-D depiction and the rest of the interface use; 3Dmol
    numbers its own atoms from 0 and would otherwise disagree with all of them.

    Indices are positions in the SDF, which is RDKit's atom order. The bound
    form keeps its heavy atoms ahead of the hydrogens `AddHs` appended, so a
    heavy-atom index means the same atom here as it does in the flat drawing.
    """
    payload = json.dumps({
        "sdf": sdf_text,
        "highlight": highlight_atom,
        "label": label,
        "color": COLORS["selected"],
        "atomLabels": {str(k): v for k, v in (atom_labels or {}).items()},
        "showH": bool(show_hydrogens),
    })
    return f"""
<div id="ligview" style="width:100%;height:{height}px;
     border:1px solid rgba(49,51,63,0.10);border-radius:8px;overflow:hidden;"></div>
<script src="{CDN}"></script>
<script>
(function () {{
  const data = {payload};
  function start() {{
    if (typeof $3Dmol === "undefined") {{ return setTimeout(start, 120); }}
    const viewer = $3Dmol.createViewer(document.getElementById("ligview"),
                                       {{backgroundColor: "white"}});
    viewer.addModel(data.sdf, "sdf");
    viewer.setStyle({{}}, {{stick: {{radius: 0.17, colorscheme: "whiteCarbon"}}}});
    if (!data.showH) {{ viewer.setStyle({{elem: "H"}}, {{}}); }}

    // selectedAtoms({{}}) returns the model's atoms in parse order, so position
    // in this array is the SDF index the caller's labels are keyed on.
    const atoms = viewer.getModel(0).selectedAtoms({{}});

    Object.keys(data.atomLabels).forEach(function (key) {{
      const at = atoms[parseInt(key, 10)];
      if (!at) {{ return; }}
      if (!data.showH && at.elem === "H") {{ return; }}
      viewer.addLabel(data.atomLabels[key], {{
        position: {{x: at.x, y: at.y, z: at.z}},
        fontSize: 9, fontColor: "#1f2937", backgroundColor: "white",
        backgroundOpacity: 0.55, borderThickness: 0, inFront: true}});
    }});

    if (data.highlight !== null && data.highlight !== undefined) {{
      const at = atoms[data.highlight];
      if (at) {{
        viewer.addSphere({{center: {{x: at.x, y: at.y, z: at.z}}, radius: 0.55,
                          color: data.color, opacity: 0.75}});
        if (data.label) {{
          viewer.addLabel(data.label, {{
            position: {{x: at.x, y: at.y, z: at.z}}, backgroundColor: data.color,
            backgroundOpacity: 0.9, fontColor: "white", fontSize: 12,
            borderThickness: 0, inFront: true}});
        }}
      }}
    }}
    viewer.zoomTo();
    viewer.render();
  }}
  start();
}})();
</script>
"""


def render_molecule(sdf_text: str, height: int = 320, **kwargs) -> None:
    import streamlit.components.v1 as components

    components.html(molecule_viewer_html(sdf_text, height=height, **kwargs),
                    height=height + 12)
