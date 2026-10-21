"""
plip.py — optional interaction profiling of the finished complex.

PLIP is not a dependency. If it is installed the report is parsed and shown;
if not, the app says so and offers the complex PDB for download instead.

Two things come back from a run: the readable text report, and `Contact`
objects carrying each interaction's two endpoints in space so the viewer can
draw them. PLIP reports non-covalent interactions only — the covalent bond is
never among them, and is drawn separately from the complex geometry.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

XYZ = Tuple[float, float, float]

# PLIP names each interaction group after its table/tag; these are the ones
# that carry a protein point and a ligand point, so they can be drawn.
GROUP_LABELS = {
    "hydrophobic_interactions": "Hydrophobic Interactions",
    "hydrogen_bonds": "Hydrogen Bonds",
    "water_bridges": "Water Bridges",
    "salt_bridges": "Salt Bridges",
    "pi_stacks": "pi-Stacking",
    "pi_cation_interactions": "pi-Cation Interactions",
    "halogen_bonds": "Halogen Bonds",
    "metal_complexes": "Metal Complexes",
}

# Coordinate pairs, in the order they are looked for. Metal complexes name
# their two ends differently from every other interaction type.
_COORD_PAIRS = (("ligcoo", "protcoo"), ("metalcoo", "targetcoo"))
_DIST_TAGS = ("dist", "dist_h-a", "dist_d-a", "centdist")


@dataclass
class Contact:
    """One PLIP interaction, with both ends in space so it can be drawn."""
    kind: str                     # "Hydrogen Bonds", "pi-Stacking", …
    residue: str                  # "Cys12 A"
    chain: str
    resnr: Optional[int]
    protein_xyz: XYZ
    ligand_xyz: XYZ
    distance: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        """JSON-safe, for the session manifest and the 3-D viewer."""
        return {"kind": self.kind, "residue": self.residue, "chain": self.chain,
                "resnr": self.resnr, "prot": list(self.protein_xyz),
                "lig": list(self.ligand_xyz), "distance": self.distance}


@dataclass
class PlipResult:
    ok: bool
    report_path: Optional[str]
    text: str = ""
    interactions: Dict[str, List[str]] = field(default_factory=dict)
    contacts: List[Contact] = field(default_factory=list)
    error: Optional[str] = None


def available() -> bool:
    if shutil.which("plip"):
        return True
    try:
        import plip  # noqa: F401
        return True
    except ImportError:
        return False


def _command(complex_pdb: str, out_dir: str) -> List[str]:
    # -t for the readable report, -x for the XML one: the text tables are
    # pretty-printed and shift between PLIP versions, while the XML carries
    # the interaction coordinates in a stable shape.
    flags = ["-f", complex_pdb, "-t", "-x", "--name", "report", "-o", out_dir]
    if shutil.which("plip"):
        return ["plip"] + flags
    return ["python3", "-m", "plip.plipcmd"] + flags


def run(complex_pdb: str, out_dir: str, timeout: int = 300) -> PlipResult:
    if not available():
        return PlipResult(
            ok=False, report_path=None,
            error="PLIP is not installed. Install it with: pip install plip",
        )
    if not os.path.isfile(complex_pdb):
        return PlipResult(ok=False, report_path=None,
                          error=f"Complex not found: {complex_pdb}")

    os.makedirs(out_dir, exist_ok=True)
    try:
        proc = subprocess.run(_command(complex_pdb, out_dir), capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return PlipResult(ok=False, report_path=None,
                          error=f"PLIP exceeded {timeout}s.")
    except Exception as exc:                                  # noqa: BLE001
        return PlipResult(ok=False, report_path=None, error=str(exc))

    report = os.path.join(out_dir, "report.txt")
    if not os.path.isfile(report):
        return PlipResult(
            ok=False, report_path=None,
            error=f"PLIP produced no report. {proc.stderr.strip()[:300]}",
        )
    with open(report, errors="replace") as fh:
        text = fh.read()

    xml_path = os.path.join(out_dir, "report.xml")
    contacts = parse_xml_contacts(xml_path) if os.path.isfile(xml_path) else []
    if not contacts:
        contacts = parse_text_contacts(text)

    return PlipResult(ok=True, report_path=report, text=text,
                      interactions=parse_report(text), contacts=contacts)


def _group_label(tag: str) -> str:
    return GROUP_LABELS.get(tag, tag.replace("_", " ").title())


def _xml_xyz(node) -> Optional[XYZ]:
    if node is None:
        return None
    try:
        return (float(node.findtext("x")), float(node.findtext("y")),
                float(node.findtext("z")))
    except (AttributeError, TypeError, ValueError):
        return None


def _residue_label(restype: str, resnr: Optional[int], chain: str) -> str:
    name = (restype or "").strip().capitalize() or "?"
    label = f"{name}{resnr}" if resnr is not None else name
    return f"{label} {chain}".strip()


def parse_xml_contacts(xml_path: str) -> List[Contact]:
    """Interaction geometry from PLIP's XML report."""
    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError):
        return []

    out: List[Contact] = []
    for group in root.iter("interactions"):
        for kind_node in group:
            label = _group_label(kind_node.tag)
            for item in kind_node:
                prot = lig = None
                for lig_tag, prot_tag in _COORD_PAIRS:
                    lig = _xml_xyz(item.find(lig_tag))
                    prot = _xml_xyz(item.find(prot_tag))
                    if lig and prot:
                        break
                if not (lig and prot):
                    continue           # nothing to draw for this one

                distance = None
                for tag in _DIST_TAGS:
                    raw = item.findtext(tag)
                    if raw:
                        try:
                            distance = float(raw)
                            break
                        except ValueError:
                            pass
                try:
                    resnr = int(item.findtext("resnr"))
                except (TypeError, ValueError):
                    resnr = None
                chain = (item.findtext("reschain") or "").strip()
                out.append(Contact(
                    kind=label,
                    residue=_residue_label(item.findtext("restype"), resnr, chain),
                    chain=chain, resnr=resnr,
                    protein_xyz=prot, ligand_xyz=lig, distance=distance,
                ))
    return out


def _cell_xyz(value: str) -> Optional[XYZ]:
    parts = [p.strip() for p in (value or "").split(",")]
    if len(parts) != 3:
        return None
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return None


def parse_text_contacts(text: str) -> List[Contact]:
    """
    Same geometry, read back out of the text tables.

    Only used when the XML report is missing — older PLIP builds, or a run
    where -x was rejected. The LIGCOO/PROTCOO columns are optional in the
    text report, so this can legitimately come back empty.
    """
    out: List[Contact] = []
    kind: Optional[str] = None
    header: Optional[List[str]] = None

    for line in text.splitlines():
        heading = re.match(r"^\*\*(.+?)\*\*$", line.strip())
        if heading:
            kind, header = heading.group(1).strip(), None
            continue
        s = line.strip()
        if not kind or not s.startswith("|") or set(s) <= set("|+-= "):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if header is None:
            header = [c.upper() for c in cells]
            continue

        row = dict(zip(header, cells))
        lig = prot = None
        for lig_key, prot_key in _COORD_PAIRS:
            lig = _cell_xyz(row.get(lig_key.upper(), ""))
            prot = _cell_xyz(row.get(prot_key.upper(), ""))
            if lig and prot:
                break
        if not (lig and prot):
            continue

        distance = None
        for tag in _DIST_TAGS:
            try:
                distance = float(row[tag.upper()])
                break
            except (KeyError, ValueError):
                pass
        try:
            resnr = int(row.get("RESNR", ""))
        except ValueError:
            resnr = None
        chain = row.get("RESCHAIN", "").strip()
        out.append(Contact(
            kind=kind, residue=_residue_label(row.get("RESTYPE"), resnr, chain),
            chain=chain, resnr=resnr,
            protein_xyz=prot, ligand_xyz=lig, distance=distance,
        ))
    return out


def parse_report(text: str) -> Dict[str, List[str]]:
    """Pull the interaction sections out of a PLIP text report."""
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        heading = re.match(r"^\*\*(.+?)\*\*$", line.strip())
        if heading:
            current = heading.group(1).strip()
            sections.setdefault(current, [])
            continue
        if current and line.startswith("|") and not set(line.strip()) <= set("|+-= "):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and cells[0] and cells[0].upper() != "RESNR":
                sections[current].append(" · ".join(c for c in cells if c))
    return {k: v for k, v in sections.items() if v}


def summarise(result: PlipResult, max_rows: int = 6) -> List[str]:
    lines = []
    for name, rows in result.interactions.items():
        lines.append(f"{name}: {len(rows)}")
        for row in rows[:max_rows]:
            lines.append(f"    {row}")
    return lines
