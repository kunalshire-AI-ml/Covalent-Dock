"""
sequence_view.py — the residue grid.

A card holding one letter per residue, 20 to a row, with chosen residues
highlighted. Three cell states:

  plain      light grey tile, dark text
  candidate  residue carrying a possible nucleophilic atom
  selected   the residue the user picked

Laid out as a fixed table rather than CSS grid: columns then line up
identically in every rendering engine, including on the last row when it is
short, and every cell keeps the width of the rows above it.

`sequence_html` builds the markup with no Streamlit dependency, so it can be
unit-tested or reused; `render_sequence` drops it into a Streamlit page.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

# Palette follows Streamlit's light theme so the card sits flush with the page.
CSS = """
<style>
.seqcard {
  border: 1px solid rgba(49, 51, 63, 0.10);
  border-radius: 8px;
  padding: 16px 18px 14px 18px;
  background: #ffffff;
  margin: 4px 0 10px 0;
}
.seqcard-title { font-size: 0.875rem; color: #31333F; margin-bottom: 10px; }
.seqtable {
  width: 100%;
  table-layout: fixed;
  border-collapse: separate;
  border-spacing: 4px;
}
.seqtable td {
  font-family: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace;
  font-size: 0.92rem;
  line-height: 1.45;
  text-align: center;
  padding: 4px 0;
  border-radius: 5px;
  background: #fafafa;
  color: #31333F;
  border: 1px solid transparent;
  white-space: nowrap;
}
.seqtable td.pad { background: transparent; border-color: transparent; }
.seqtable td.num {
  width: 3.4rem;
  background: transparent;
  border-color: transparent;
  color: #a3a8b8;
  font-size: 0.72rem;
  text-align: right;
  padding-right: 8px;
}
.seqtable td.candidate {
  background: #fdf6e3;
  border-color: #f2e2b8;
  color: #6b5a22;
}
.seqtable td.selected {
  background: #dfe3fd;
  border-color: #b5bdf5;
  color: #2b2f77;
  font-weight: 600;
}
.seqtable td.clickable { cursor: pointer; }
.seqtable a { text-decoration: none; color: inherit; display: block; }
.seqtable td.clickable:hover { border-color: #9aa2e8; }
.seqcard-caption { font-size: 0.78rem; color: #808495; margin-top: 10px; }
.seq-swatch {
  display: inline-block; width: 0.7rem; height: 0.7rem;
  border-radius: 3px; vertical-align: -1px; margin-right: 5px;
}
.seq-swatch.selected { background: #dfe3fd; border: 1px solid #b5bdf5; }
.seq-swatch.candidate { background: #fdf6e3; border: 1px solid #f2e2b8; }
.seq-swatch.plain { background: #fafafa; border: 1px solid rgba(49,51,63,0.12); }
</style>
"""


@dataclass
class SeqResidue:
    """Minimal shape the grid needs — `Residue` from protein_reactivity fits."""
    one_letter: str
    label: str
    key: str
    resseq: int


def to_seq_residues(residues: Iterable) -> List[SeqResidue]:
    out: List[SeqResidue] = []
    for r in residues:
        if isinstance(r, SeqResidue):
            out.append(r)
            continue
        out.append(SeqResidue(
            one_letter=getattr(r, "one_letter", "X"),
            label=getattr(r, "label", ""),
            key=getattr(r, "key", ""),
            resseq=getattr(r, "resseq", 0),
        ))
    return out


def sequence_html(
    residues: Sequence,
    selected_keys: Optional[Iterable[str]] = None,
    candidate_keys: Optional[Iterable[str]] = None,
    per_row: int = 20,
    title: str = "Sequence visualization",
    caption: str = "Highlighted = selected interacting residues",
    show_numbers: bool = False,
    tooltips: Optional[Dict[str, str]] = None,
    link_for: Optional[Callable[[SeqResidue], Optional[str]]] = None,
    include_css: bool = True,
) -> str:
    """Build the sequence card markup."""
    rows = to_seq_residues(residues)
    selected = set(selected_keys or ())
    candidates = set(candidate_keys or ())
    tooltips = tooltips or {}

    lines: List[str] = []
    for start in range(0, len(rows), per_row):
        chunk = rows[start:start + per_row]
        cells: List[str] = []
        if show_numbers:
            cells.append(f'<td class="num">{chunk[0].resseq}</td>')
        for res in chunk:
            classes: List[str] = []
            if res.key in selected:
                classes.append("selected")
            elif res.key in candidates:
                classes.append("candidate")
            tip = tooltips.get(res.key) or res.label
            body = html.escape(res.one_letter)
            href = link_for(res) if link_for else None
            if href:
                classes.append("clickable")
                body = f'<a href="{html.escape(href)}" target="_self">{body}</a>'
            cls = f' class="{" ".join(classes)}"' if classes else ""
            cells.append(f'<td{cls} title="{html.escape(tip)}">{body}</td>')
        # Keep the final row aligned with the rows above it.
        for _ in range(per_row - len(chunk)):
            cells.append('<td class="pad"></td>')
        lines.append("<tr>" + "".join(cells) + "</tr>")

    parts = [CSS if include_css else ""]
    parts.append('<div class="seqcard">')
    if title:
        parts.append(f'<div class="seqcard-title">{html.escape(title)}</div>')
    parts.append('<table class="seqtable"><tbody>' + "".join(lines) + "</tbody></table>")
    if caption:
        parts.append(f'<div class="seqcard-caption">{caption}</div>')
    parts.append("</div>")
    return "".join(parts)


def legend(selected_text: str = "selected residue",
           candidate_text: str = "candidate nucleophile") -> str:
    return (
        f'<span class="seq-swatch selected"></span>{html.escape(selected_text)}'
        f'&nbsp;&nbsp;&nbsp;<span class="seq-swatch candidate"></span>'
        f'{html.escape(candidate_text)}'
    )


def render_sequence(residues: Sequence, **kwargs) -> None:
    """Draw the card in a Streamlit page."""
    import streamlit as st

    st.markdown(sequence_html(residues, **kwargs), unsafe_allow_html=True)


def chain_picker_label(chain) -> str:
    kind = {"protein": "protein", "nucleic": "nucleic acid"}.get(chain.kind, chain.kind)
    return f"Chain {chain.chain_id} · {len(chain.polymer)} residues · {kind}"
