"""
protein_reactivity.py — protein side of the covalent-docking workflow.

Two jobs:

  1. Parse a PDB into chains / residues / atoms (pure Python, no Biopython).
  2. Return the atoms that could plausibly act as nucleophiles, with enough
     context for a human to choose between them (burial, nearby basic
     residues, existing disulfides, distance to any bound ligand).

The output feeds both the sequence viewer and the gnina `--covalent_rec_atom`
selector, which has the form  CHAIN:RESNUM:ATOMNAME  (e.g. A:12:SG).
"""

from __future__ import annotations

import gzip
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .warheads import AMINE, CARBOXYLATE, HYDROXYL, IMIDAZOLE, SELENOL, THIOL

# ─────────────────────────────────────────────────────────────────────────────
#  Residue tables
# ─────────────────────────────────────────────────────────────────────────────

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    # common variants / modified residues
    "MSE": "M", "SEC": "U", "PYL": "O", "CYX": "C", "CYM": "C", "CSO": "C",
    "HID": "H", "HIE": "H", "HIP": "H", "HSD": "H", "HSE": "H", "HSP": "H",
    "ASH": "D", "GLH": "E", "LYN": "K", "ARN": "R", "TPO": "T", "SEP": "S",
    "PTR": "Y", "MLY": "K", "KCX": "K", "ACE": "X", "NME": "X", "UNK": "X",
}

NUCLEIC_TO_1 = {
    "DA": "A", "DC": "C", "DG": "G", "DT": "T", "DU": "U", "DI": "I",
    "A": "A", "C": "C", "G": "G", "U": "U", "I": "I", "T": "T",
    "ADE": "A", "CYT": "C", "GUA": "G", "THY": "T", "URA": "U",
}

WATERS = {"HOH", "WAT", "DOD", "H2O", "SOL", "TIP3"}

# residue -> [(atom name, nucleophile type, short description), ...]
NUCLEOPHILE_ATOMS: Dict[str, List[Tuple[str, str, str]]] = {
    "CYS": [("SG", THIOL, "thiol / thiolate")],
    "CYX": [("SG", THIOL, "thiol (disulfide-bonded)")],
    "CYM": [("SG", THIOL, "thiolate")],
    "SEC": [("SE", SELENOL, "selenol")],
    "LYS": [("NZ", AMINE, "ε-amino")],
    "SER": [("OG", HYDROXYL, "hydroxyl")],
    "THR": [("OG1", HYDROXYL, "hydroxyl")],
    "TYR": [("OH", HYDROXYL, "phenol")],
    "HIS": [("ND1", IMIDAZOLE, "imidazole Nδ1"), ("NE2", IMIDAZOLE, "imidazole Nε2")],
    "HID": [("ND1", IMIDAZOLE, "imidazole Nδ1"), ("NE2", IMIDAZOLE, "imidazole Nε2")],
    "HIE": [("ND1", IMIDAZOLE, "imidazole Nδ1"), ("NE2", IMIDAZOLE, "imidazole Nε2")],
    "HIP": [("ND1", IMIDAZOLE, "imidazole Nδ1"), ("NE2", IMIDAZOLE, "imidazole Nε2")],
    "HSD": [("ND1", IMIDAZOLE, "imidazole Nδ1"), ("NE2", IMIDAZOLE, "imidazole Nε2")],
    "HSE": [("ND1", IMIDAZOLE, "imidazole Nδ1"), ("NE2", IMIDAZOLE, "imidazole Nε2")],
    "ASP": [("OD1", CARBOXYLATE, "carboxylate"), ("OD2", CARBOXYLATE, "carboxylate")],
    "GLU": [("OE1", CARBOXYLATE, "carboxylate"), ("OE2", CARBOXYLATE, "carboxylate")],
}

# Basic/polar partners that lower a cysteine pKa when nearby.
ACTIVATING_ATOMS = {
    ("LYS", "NZ"), ("ARG", "NH1"), ("ARG", "NH2"), ("ARG", "NE"),
    ("HIS", "ND1"), ("HIS", "NE2"), ("HSD", "ND1"), ("HSE", "NE2"),
}

NUCLEOPHILE_LABELS = {
    THIOL: "thiol (S)",
    SELENOL: "selenol (Se)",
    AMINE: "amine (N)",
    HYDROXYL: "hydroxyl (O)",
    IMIDAZOLE: "imidazole (N)",
    CARBOXYLATE: "carboxylate (O)",
}

# The attacking element, by nucleophile type. Sets the bond length used when
# the ligand is placed at the covalent site (see docking/prepare_ligand.py);
# taking the first letter of the atom name instead would read Sec's SE as S.
NUCLEOPHILE_ELEMENT = {
    THIOL: "S",
    SELENOL: "SE",
    AMINE: "N",
    HYDROXYL: "O",
    IMIDAZOLE: "N",
    CARBOXYLATE: "O",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Atom:
    serial: int
    name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    occupancy: float
    bfactor: float
    element: str
    hetatm: bool

    @property
    def xyz(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class Residue:
    chain: str
    resseq: int
    icode: str
    resname: str
    atoms: List[Atom] = field(default_factory=list)
    kind: str = "protein"          # protein | nucleic | water | hetero

    @property
    def key(self) -> str:
        return f"{self.chain}_{self.resseq}{self.icode.strip()}"

    @property
    def one_letter(self) -> str:
        if self.kind == "nucleic":
            return NUCLEIC_TO_1.get(self.resname.strip(), "N")
        return AA3_TO_1.get(self.resname.strip().upper(), "X")

    @property
    def label(self) -> str:
        name = self.resname.strip().capitalize()
        return f"{name}{self.resseq}{self.icode.strip()}"

    def atom(self, name: str) -> Optional[Atom]:
        for a in self.atoms:
            if a.name.strip() == name:
                return a
        return None


@dataclass
class Chain:
    chain_id: str
    residues: List[Residue] = field(default_factory=list)

    @property
    def kind(self) -> str:
        kinds = {r.kind for r in self.residues}
        if "protein" in kinds:
            return "protein"
        if "nucleic" in kinds:
            return "nucleic"
        return "hetero"

    @property
    def polymer(self) -> List[Residue]:
        return [r for r in self.residues if r.kind in ("protein", "nucleic")]

    @property
    def sequence(self) -> str:
        return "".join(r.one_letter for r in self.polymer)


@dataclass
class Structure:
    atoms: List[Atom]
    chains: List[Chain]
    title: str = ""
    source: str = ""

    def chain(self, chain_id: str) -> Optional[Chain]:
        for c in self.chains:
            if c.chain_id == chain_id:
                return c
        return None

    @property
    def protein_chains(self) -> List[Chain]:
        return [c for c in self.chains if c.kind == "protein"]

    def residue_by_key(self, key: str) -> Optional[Residue]:
        for c in self.chains:
            for r in c.residues:
                if r.key == key:
                    return r
        return None

    def summary(self) -> Dict[str, object]:
        n_res = sum(len(c.polymer) for c in self.chains)
        het = {
            r.resname.strip()
            for c in self.chains for r in c.residues
            if r.kind == "hetero"
        }
        return {
            "atoms": len(self.atoms),
            "chains": [c.chain_id for c in self.chains],
            "protein_chains": [c.chain_id for c in self.protein_chains],
            "nucleic_chains": [c.chain_id for c in self.chains if c.kind == "nucleic"],
            "residues": n_res,
            "hetero": sorted(het),
        }


@dataclass
class Nucleophile:
    """One candidate covalent attachment point on the protein."""
    chain: str
    resseq: int
    icode: str
    resname: str
    atom_name: str
    element: str
    nucleophile_type: str
    description: str
    xyz: Tuple[float, float, float]
    bfactor: float
    residue_key: str
    neighbor_count: int = 0          # heavy atoms within 8 A (burial proxy)
    activating_contacts: int = 0     # basic N atoms within 6 A
    disulfide_partner: Optional[str] = None
    ligand_distance: Optional[float] = None   # to nearest non-water HETATM
    flags: List[str] = field(default_factory=list)

    @property
    def selector(self) -> str:
        """gnina --covalent_rec_atom argument."""
        chain = self.chain.strip() or "A"
        return f"{chain}:{self.resseq}:{self.atom_name}"

    @property
    def label(self) -> str:
        return f"{self.resname.strip().capitalize()}{self.resseq}{self.icode.strip()}-{self.atom_name}"

    @property
    def exposure(self) -> str:
        n = self.neighbor_count
        if n <= 24:
            return "exposed"
        if n <= 44:
            return "partly buried"
        return "buried"

    @property
    def accessible(self) -> bool:
        return self.disulfide_partner is None and self.neighbor_count <= 60

    def context_note(self) -> str:
        bits = [self.exposure]
        if self.disulfide_partner:
            bits.append(f"disulfide with {self.disulfide_partner}")
        if self.activating_contacts:
            bits.append(f"{self.activating_contacts} basic contact(s)")
        if self.ligand_distance is not None and self.ligand_distance < 8:
            bits.append(f"{self.ligand_distance:.1f} Å from bound ligand")
        return " · ".join(bits)


# ─────────────────────────────────────────────────────────────────────────────
#  Parsing
# ─────────────────────────────────────────────────────────────────────────────

def _f(text: str, default: float = 0.0) -> float:
    try:
        return float(text)
    except ValueError:
        return default


def _guess_element(name: str, resname: str) -> str:
    n = name.strip()
    if not n:
        return "C"
    if n[0].isdigit():
        n = n[1:]
    two = n[:2].capitalize()
    if two in ("Cl", "Br", "Se", "Zn", "Fe", "Mg", "Mn", "Ca", "Na", "Cu", "Ni", "Co"):
        return two.upper()
    return n[0].upper()


def parse_pdb(text: str, source: str = "") -> Structure:
    """Parse PDB text (ATOM/HETATM records). Keeps the first MODEL only."""
    atoms: List[Atom] = []
    chains: Dict[str, Chain] = {}
    order: List[str] = []
    residues: Dict[Tuple[str, int, str, str], Residue] = {}
    title_parts: List[str] = []
    in_model = False

    for raw in text.splitlines():
        rec = raw[:6]
        if rec.startswith("TITLE"):
            title_parts.append(raw[10:].strip())
            continue
        if rec.startswith("MODEL"):
            if in_model:
                break            # second model: stop
            in_model = True
            continue
        if rec.startswith("ENDMDL"):
            break
        if not (rec.startswith("ATOM") or rec.startswith("HETATM")):
            continue

        line = raw.ljust(80)
        altloc = line[16]
        if altloc not in (" ", "A", "1"):
            continue

        resname = line[17:20].strip().upper()
        chain_id = line[21].strip() or "A"
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue

        element = line[76:78].strip().upper()
        name = line[12:16]
        if not element:
            element = _guess_element(name, resname)
        if element == "H" or element == "D":
            continue            # hydrogens are not needed for site selection

        atom = Atom(
            serial=int(line[6:11]) if line[6:11].strip().isdigit() else len(atoms) + 1,
            name=name.strip(),
            altloc=altloc,
            resname=resname,
            chain=chain_id,
            resseq=resseq,
            icode=line[26],
            x=_f(line[30:38]), y=_f(line[38:46]), z=_f(line[46:54]),
            occupancy=_f(line[54:60], 1.0),
            bfactor=_f(line[60:66]),
            element=element,
            hetatm=rec.startswith("HETATM"),
        )
        atoms.append(atom)

        rkey = (chain_id, resseq, atom.icode, resname)
        res = residues.get(rkey)
        if res is None:
            if resname in WATERS:
                kind = "water"
            elif resname in AA3_TO_1 and resname not in ("UNK",):
                kind = "protein"
            elif resname in NUCLEIC_TO_1:
                kind = "nucleic"
            else:
                kind = "hetero"
            res = Residue(chain=chain_id, resseq=resseq, icode=atom.icode,
                          resname=resname, kind=kind)
            residues[rkey] = res
            if chain_id not in chains:
                chains[chain_id] = Chain(chain_id=chain_id)
                order.append(chain_id)
            chains[chain_id].residues.append(res)
        res.atoms.append(atom)

    return Structure(
        atoms=atoms,
        chains=[chains[c] for c in order],
        title=" ".join(title_parts).strip(),
        source=source,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  mmCIF
# ─────────────────────────────────────────────────────────────────────────────
#
# Entries deposited since the PDB ran out of 3-character ligand codes exist
# only as mmCIF: a 5-character het code or a 3-character chain ID cannot be
# written into the fixed columns of a .pdb file, so RCSB does not produce one.
# This reads the _atom_site loop into the same Structure the PDB parser makes;
# everything downstream is unchanged.

def _cif_tokens(line: str) -> List[str]:
    """Split one mmCIF data row, honouring single and double quoting."""
    out: List[str] = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if c in " \t":
            i += 1
            continue
        if c in "'\"":
            # A quote only closes when followed by whitespace, so an atom name
            # like O5' does not terminate the token early.
            quote, i = c, i + 1
            start = i
            while i < n and not (line[i] == quote
                                 and (i + 1 >= n or line[i + 1] in " \t")):
                i += 1
            out.append(line[start:i])
            i += 1
        else:
            start = i
            while i < n and line[i] not in " \t":
                i += 1
            out.append(line[start:i])
    return out


def parse_mmcif(text: str, source: str = "") -> Structure:
    """
    Parse the _atom_site loop of an mmCIF file. First model only.

    Author numbering is preferred over the label_* fields, because that is what
    the PDB entry, the literature and the CovPDB table all quote.
    """
    atoms: List[Atom] = []
    chains: Dict[str, Chain] = {}
    order: List[str] = []
    residues: Dict[Tuple[str, int, str, str], Residue] = {}
    title_parts: List[str] = []

    lines = text.splitlines()
    i, n = 0, len(lines)
    columns: Dict[str, int] = {}
    first_model: Optional[str] = None

    while i < n:
        raw = lines[i].strip()
        if raw.startswith("_struct.title"):
            rest = raw[len("_struct.title"):].strip()
            if rest:
                title_parts.append(rest.strip("'\""))
        if raw != "loop_":
            i += 1
            continue

        # Collect the tag block, then decide whether this is the atom loop.
        i += 1
        tags: List[str] = []
        while i < n and lines[i].strip().startswith("_"):
            tags.append(lines[i].strip().split()[0])
            i += 1
        if not tags or not tags[0].startswith("_atom_site."):
            continue
        columns = {t.split(".", 1)[1]: k for k, t in enumerate(tags)}

        def field(row: List[str], *names: str, default: str = "") -> str:
            for name in names:
                k = columns.get(name)
                if k is not None and k < len(row):
                    value = row[k]
                    if value not in (".", "?"):
                        return value
            return default

        while i < n:
            line = lines[i]
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "loop_", "_", "data_")):
                break
            i += 1
            row = _cif_tokens(line)
            if len(row) < len(tags) - 2:          # a malformed or wrapped row
                continue

            model = field(row, "pdbx_PDB_model_num", default="1")
            if first_model is None:
                first_model = model
            elif model != first_model:
                break

            altloc = field(row, "label_alt_id", default=" ") or " "
            if altloc not in (" ", "A", "1"):
                continue
            element = field(row, "type_symbol").upper()
            if element in ("H", "D"):
                continue

            name = field(row, "auth_atom_id", "label_atom_id")
            resname = field(row, "auth_comp_id", "label_comp_id").upper()
            chain_id = field(row, "auth_asym_id", "label_asym_id", default="A") or "A"
            seq = field(row, "auth_seq_id", "label_seq_id")
            try:
                resseq = int(seq)
            except ValueError:
                continue
            icode = field(row, "pdbx_PDB_ins_code", default=" ") or " "
            try:
                serial = int(field(row, "id", default="0"))
            except ValueError:
                serial = len(atoms) + 1

            atom = Atom(
                serial=serial, name=name, altloc=altloc, resname=resname,
                chain=chain_id, resseq=resseq, icode=icode,
                x=_f(field(row, "Cartn_x")), y=_f(field(row, "Cartn_y")),
                z=_f(field(row, "Cartn_z")),
                occupancy=_f(field(row, "occupancy"), 1.0),
                bfactor=_f(field(row, "B_iso_or_equiv")),
                element=element or _guess_element(name, resname),
                hetatm=field(row, "group_PDB", default="ATOM").upper() == "HETATM",
            )
            atoms.append(atom)

            rkey = (chain_id, resseq, icode, resname)
            res = residues.get(rkey)
            if res is None:
                if resname in WATERS:
                    kind = "water"
                elif resname in AA3_TO_1 and resname != "UNK":
                    kind = "protein"
                elif resname in NUCLEIC_TO_1:
                    kind = "nucleic"
                else:
                    kind = "hetero"
                res = Residue(chain=chain_id, resseq=resseq, icode=icode,
                              resname=resname, kind=kind)
                residues[rkey] = res
                if chain_id not in chains:
                    chains[chain_id] = Chain(chain_id=chain_id)
                    order.append(chain_id)
                chains[chain_id].residues.append(res)
            res.atoms.append(atom)
        break

    return Structure(atoms=atoms, chains=[chains[c] for c in order],
                     title=" ".join(title_parts).strip(), source=source)


_PDB_CHAIN_POOL = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                   "abcdefghijklmnopqrstuvwxyz0123456789")


def structure_to_pdb(structure: Structure) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    """
    Write a Structure as PDB text, renaming whatever will not fit.

    An mmCIF-only entry is mmCIF-only precisely because something in it is too
    wide for the fixed columns. Those names are arbitrary labels as far as
    docking is concerned, so they are remapped rather than refused: chain
    "AAA" becomes "A", het code "A1BJC" becomes a free 3-character code.

    Returns the text and both maps, because the caller has to ask for the
    ligand and the covalent residue by their new names.
    """
    chain_map: Dict[str, str] = {}
    used_chains = {c.chain_id for c in structure.chains if len(c.chain_id) == 1}
    for chain in structure.chains:
        cid = chain.chain_id
        if len(cid) == 1:
            chain_map[cid] = cid
            continue
        for candidate in _PDB_CHAIN_POOL:
            if candidate not in used_chains:
                chain_map[cid] = candidate
                used_chains.add(candidate)
                break
        else:
            chain_map[cid] = "Z"

    het_map: Dict[str, str] = {}
    used_codes = {a.resname for a in structure.atoms if len(a.resname) <= 3}
    counter = 0
    for atom in structure.atoms:
        if len(atom.resname) <= 3 or atom.resname in het_map:
            continue
        while counter < 36 ** 2:
            candidate = f"L{counter // 36:X}{counter % 36:X}"[:3]
            counter += 1
            if candidate not in used_codes:
                het_map[atom.resname] = candidate
                used_codes.add(candidate)
                break
        else:
            het_map[atom.resname] = "LIG"

    lines: List[str] = []
    for atom in structure.atoms:
        name = atom.name.strip()
        # Columns 13-16: a one-letter element is indented by one, which is what
        # every PDB reader keys off to split the name from the element.
        field = f"{name:<4}" if len(name) >= 4 or len(atom.element) == 2 \
            else f" {name:<3}"
        lines.append(
            f"{'HETATM' if atom.hetatm else 'ATOM  '}"
            f"{atom.serial % 100000:>5} {field}"
            f"{atom.altloc if atom.altloc.strip() else ' '}"
            f"{het_map.get(atom.resname, atom.resname):>3} "
            f"{chain_map.get(atom.chain, atom.chain)[:1]}"
            f"{atom.resseq % 10000:>4}{atom.icode if atom.icode.strip() else ' '}   "
            f"{atom.x:>8.3f}{atom.y:>8.3f}{atom.z:>8.3f}"
            f"{atom.occupancy:>6.2f}{atom.bfactor:>6.2f}"
            f"{'':>10}{atom.element:>2}")
    lines.append("END")
    return "\n".join(lines) + "\n", chain_map, het_map


def as_pdb_text(text: str, filename: str = ""
                ) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    """
    PDB text for content already in memory — an upload or a paste.

    The format is taken from the file name where there is one, and otherwise
    from the content: every mmCIF file opens with a `data_` block header, which
    no PDB file does.
    """
    if filename.lower().endswith((".cif", ".mmcif")) or \
            text.lstrip().startswith("data_"):
        return structure_to_pdb(parse_mmcif(text))
    return text, {}, {}


def load_structure_text(path: str) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    """
    PDB text for any supported structure file, plus the chain and het-code
    renames applied. A .pdb file is returned untouched with empty maps.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as fh:
        text = fh.read()
    stem = path[:-3] if path.endswith(".gz") else path
    if stem.lower().endswith((".cif", ".mmcif")):
        return structure_to_pdb(parse_mmcif(text, source=path))
    return text, {}, {}


def read_pdb_file(path: str) -> Structure:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as fh:
        return parse_pdb(fh.read(), source=path)


# ─────────────────────────────────────────────────────────────────────────────
#  Neighbour search (cell list — no numpy/scipy required)
# ─────────────────────────────────────────────────────────────────────────────

class _Grid:
    def __init__(self, atoms: Sequence[Atom], cell: float = 8.0):
        self.cell = cell
        self.bins: Dict[Tuple[int, int, int], List[Atom]] = {}
        for a in atoms:
            self.bins.setdefault(self._key(a.x, a.y, a.z), []).append(a)

    def _key(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        c = self.cell
        return (int(math.floor(x / c)), int(math.floor(y / c)), int(math.floor(z / c)))

    def near(self, x: float, y: float, z: float, radius: float) -> Iterable[Atom]:
        span = int(math.ceil(radius / self.cell))
        i0, j0, k0 = self._key(x, y, z)
        for i in range(i0 - span, i0 + span + 1):
            for j in range(j0 - span, j0 + span + 1):
                for k in range(k0 - span, k0 + span + 1):
                    for a in self.bins.get((i, j, k), ()):
                        yield a


def _dist(a: Atom, x: float, y: float, z: float) -> float:
    return math.sqrt((a.x - x) ** 2 + (a.y - y) ** 2 + (a.z - z) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
#  Nucleophile detection
# ─────────────────────────────────────────────────────────────────────────────

def find_nucleophiles(
    structure: Structure,
    chains: Optional[Sequence[str]] = None,
    types: Optional[Sequence[str]] = None,
    burial_radius: float = 8.0,
) -> List[Nucleophile]:
    """
    Return every chemically plausible nucleophilic atom, annotated with context.

    `chains` restricts to given chain IDs; `types` restricts to nucleophile
    types (thiol, amine, hydroxyl, imidazole, carboxylate, selenol).
    """
    heavy = [a for a in structure.atoms if a.resname not in WATERS]
    grid = _Grid(heavy, cell=burial_radius)
    het_atoms = [a for a in heavy if a.hetatm and a.resname not in AA3_TO_1]
    het_grid = _Grid(het_atoms, cell=10.0) if het_atoms else None

    # Pre-index all SG atoms for disulfide detection.
    sg_atoms = [a for a in heavy if a.name.strip() == "SG"]

    out: List[Nucleophile] = []
    for chain in structure.chains:
        if chains and chain.chain_id not in chains:
            continue
        for res in chain.residues:
            spec = NUCLEOPHILE_ATOMS.get(res.resname.strip().upper())
            if not spec:
                continue
            for atom_name, nu_type, desc in spec:
                if types and nu_type not in types:
                    continue
                atom = res.atom(atom_name)
                if atom is None:
                    continue

                x, y, z = atom.xyz
                n_neighbors = 0
                activating = 0
                for other in grid.near(x, y, z, burial_radius):
                    if other is atom:
                        continue
                    d = _dist(other, x, y, z)
                    if d <= burial_radius and (other.chain, other.resseq, other.icode) != (
                        atom.chain, atom.resseq, atom.icode
                    ):
                        n_neighbors += 1
                    if d <= 6.0 and (other.resname, other.name.strip()) in ACTIVATING_ATOMS:
                        activating += 1

                partner = None
                if atom_name == "SG":
                    for other in sg_atoms:
                        if other is atom:
                            continue
                        if _dist(other, x, y, z) <= 2.5:
                            partner = f"{other.resname.capitalize()}{other.resseq}"
                            break

                lig_d = None
                if het_grid is not None:
                    best = None
                    for other in het_grid.near(x, y, z, 12.0):
                        d = _dist(other, x, y, z)
                        if best is None or d < best:
                            best = d
                    lig_d = round(best, 2) if best is not None and best <= 12.0 else None

                nu = Nucleophile(
                    chain=atom.chain, resseq=atom.resseq, icode=atom.icode,
                    resname=res.resname, atom_name=atom_name, element=atom.element,
                    nucleophile_type=nu_type, description=desc, xyz=(x, y, z),
                    bfactor=atom.bfactor, residue_key=res.key,
                    neighbor_count=n_neighbors, activating_contacts=activating,
                    disulfide_partner=partner, ligand_distance=lig_d,
                )
                if partner:
                    nu.flags.append("disulfide-bonded")
                if nu.neighbor_count > 60:
                    nu.flags.append("deeply buried")
                if lig_d is not None and lig_d <= 6.0:
                    nu.flags.append("at a ligand site")
                if nu_type == CARBOXYLATE:
                    nu.flags.append("weak nucleophile")
                out.append(nu)

    return out


def rank_nucleophiles(nucleophiles: Sequence[Nucleophile]) -> List[Nucleophile]:
    """
    Heuristic ordering for the UI: accessible, activated, ligand-proximal
    cysteines first. This is a convenience ranking, not a reactivity prediction.
    """
    type_weight = {THIOL: 0, SELENOL: 0, IMIDAZOLE: 1, AMINE: 1,
                   HYDROXYL: 2, CARBOXYLATE: 3}

    def key(n: Nucleophile):
        return (
            1 if n.disulfide_partner else 0,
            type_weight.get(n.nucleophile_type, 4),
            0 if (n.ligand_distance is not None and n.ligand_distance <= 8) else 1,
            -n.activating_contacts,
            n.neighbor_count,
            n.chain,
            n.resseq,
        )

    return sorted(nucleophiles, key=key)


def sequence_rows(chain: Chain) -> List[Residue]:
    """Polymer residues in file order — what the sequence viewer renders."""
    return chain.polymer


def find_atom(structure: Structure, chain_id: str, resseq: int,
              atom_name: str, icode: str = " ") -> Optional[Atom]:
    ch = structure.chain(chain_id)
    if ch is None:
        return None
    for res in ch.residues:
        if res.resseq == resseq and (res.icode == icode or icode == " "):
            a = res.atom(atom_name)
            if a is not None:
                return a
    return None
