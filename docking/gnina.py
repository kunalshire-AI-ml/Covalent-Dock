"""
gnina.py — build and run the covalent docking command.

The interface (gnina 1.3):

  gnina -r rec.pdb -l lig.sdf --autobox_ligand site.pdb \
        --covalent_rec_atom A:12:SG \
        --covalent_lig_atom_pattern '[$([CH3][CH2][C](=O))]' \
        --covalent_bond_order 1 -o out.sdf \
        --out_flex gnina_complex.sdf --full_flex_output \
        --cnn_scoring none --no_gpu

The receptor atom is CHAIN:RESNUM:ATOMNAME (or x,y,z). The ligand atom is a
SMARTS expression and *every* matching atom is evaluated. The ligand must
already be in its bound form; gnina does not perform the reaction.
"""

from __future__ import annotations

import glob
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class GninaConfig:
    receptor: str
    ligand: str
    out: str

    # Flexible-residue output. Covalent docking makes the receptor residue
    # flexible, so this is where the reacted side chain lands; with
    # --full_flex_output the whole receptor is written alongside it.
    out_flex: Optional[str] = None
    full_flex_output: bool = True

    # covalent options
    covalent_rec_atom: Optional[str] = None          # A:12:SG or x,y,z
    covalent_lig_atom_pattern: Optional[str] = None  # SMARTS
    covalent_bond_order: int = 1
    covalent_optimize_lig: bool = True
    covalent_lig_atom_position: Optional[Tuple[float, float, float]] = None
    covalent_fix_lig_atom_position: bool = False

    # search space — either an explicit box or a reference file
    center: Optional[Tuple[float, float, float]] = None
    box_size: float = 20.0
    autobox_ligand: Optional[str] = None
    autobox_add: Optional[float] = None

    # sampling and scoring
    exhaustiveness: int = 16
    num_modes: int = 9
    seed: Optional[int] = 42
    # No GPU on the deployment host, so the CNN is off and gnina is told to
    # stay on the CPU. Both are deliberate defaults, not placeholders.
    cnn_scoring: str = "none"          # none | rescore | refinement | all
    scoring: Optional[str] = None      # vina | vinardo | ...
    cpu: Optional[int] = None
    no_gpu: bool = True

    log: Optional[str] = None
    extra_args: List[str] = field(default_factory=list)
    binary: str = "gnina"

    def validate(self) -> List[str]:
        problems = []
        if not os.path.isfile(self.receptor):
            problems.append(f"Receptor file not found: {self.receptor}")
        if not os.path.isfile(self.ligand):
            problems.append(f"Ligand file not found: {self.ligand}")
        if self.covalent_rec_atom and not self.covalent_lig_atom_pattern:
            problems.append("A receptor atom is set but no ligand SMARTS pattern.")
        if self.covalent_lig_atom_pattern and not self.covalent_rec_atom:
            problems.append("A ligand SMARTS is set but no receptor atom.")
        if self.center is None and not self.autobox_ligand:
            problems.append("No search box: give a centre or an autobox file.")
        if self.covalent_bond_order not in (1, 2, 3):
            problems.append("Bond order must be 1, 2 or 3.")
        return problems

    def command(self) -> List[str]:
        cmd: List[str] = [self.binary, "-r", self.receptor, "-l", self.ligand,
                          "-o", self.out]

        if self.out_flex:
            cmd += ["--out_flex", self.out_flex]
            if self.full_flex_output:
                cmd += ["--full_flex_output"]

        if self.autobox_ligand:
            cmd += ["--autobox_ligand", self.autobox_ligand]
            if self.autobox_add is not None:
                cmd += ["--autobox_add", str(self.autobox_add)]
        elif self.center is not None:
            cx, cy, cz = self.center
            cmd += ["--center_x", f"{cx:.3f}",
                    "--center_y", f"{cy:.3f}",
                    "--center_z", f"{cz:.3f}",
                    "--size_x", f"{self.box_size:g}",
                    "--size_y", f"{self.box_size:g}",
                    "--size_z", f"{self.box_size:g}"]

        if self.covalent_rec_atom:
            cmd += ["--covalent_rec_atom", self.covalent_rec_atom]
        if self.covalent_lig_atom_pattern:
            cmd += ["--covalent_lig_atom_pattern", self.covalent_lig_atom_pattern]
        if self.covalent_rec_atom:
            cmd += ["--covalent_bond_order", str(self.covalent_bond_order)]
            if self.covalent_optimize_lig:
                cmd += ["--covalent_optimize_lig"]
            if self.covalent_lig_atom_position:
                x, y, z = self.covalent_lig_atom_position
                cmd += ["--covalent_lig_atom_position", f"{x:.3f},{y:.3f},{z:.3f}"]
            if self.covalent_fix_lig_atom_position:
                cmd += ["--covalent_fix_lig_atom_position"]

        cmd += ["--exhaustiveness", str(self.exhaustiveness),
                "--num_modes", str(self.num_modes),
                "--cnn_scoring", self.cnn_scoring]
        if self.scoring:
            cmd += ["--scoring", self.scoring]
        if self.seed is not None:
            cmd += ["--seed", str(self.seed)]
        if self.cpu:
            cmd += ["--cpu", str(self.cpu)]
        if self.no_gpu:
            cmd += ["--no_gpu"]
        if self.log:
            cmd += ["--log", self.log]
        cmd += list(self.extra_args)
        return cmd

    def command_string(self, wrap: bool = True) -> str:
        """Copy-pasteable shell command."""
        parts = [shlex.quote(c) for c in self.command()]
        if not wrap:
            return " ".join(parts)
        def is_flag(part: str) -> bool:
            # A negative coordinate also starts with '-', and breaking the line
            # there splits '--center_x' from its own value.
            return part.startswith("-") and not part[1:2].isdigit() \
                and not part.startswith("-.")

        lines, current = [], parts[0]
        for part in parts[1:]:
            if is_flag(part) and len(current) > 0:
                lines.append(current)
                current = "    " + part
            else:
                current += " " + part
        lines.append(current)
        return " \\\n".join(lines)


@dataclass
class GninaResult:
    ok: bool
    returncode: int
    command: str
    stdout: str
    stderr: str
    out_path: str
    seconds: float
    n_poses: int = 0
    error: Optional[str] = None


def find_binary(candidates: Sequence[str] = ("gnina",)) -> Optional[str]:
    for name in candidates:
        path = shutil.which(name)
        if path:
            return path
    for path in ("/usr/local/bin/gnina", "/opt/gnina/bin/gnina",
                 os.path.expanduser("~/bin/gnina")):
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


#: The force-field parameter file OpenBabel loads for --covalent_optimize_lig.
#: Its presence is what makes a directory the Open Babel data directory.
BABEL_MARKER = "UFF.prm"


def _datadir_candidates(binary: Optional[str]) -> List[str]:
    """Where an Open Babel data directory plausibly lives, best guess first."""
    out: List[str] = []
    if binary:
        # A relocatable gnina ships its data beside or just above the binary.
        base = os.path.dirname(os.path.abspath(binary))
        out += [os.path.join(base, "data"),
                os.path.join(os.path.dirname(base), "share", "openbabel"),
                os.path.join(os.path.dirname(base), "share")]
    # The environment the app itself runs in, conda or venv.
    for prefix in (os.environ.get("CONDA_PREFIX"), sys.prefix):
        if not prefix:
            continue
        out += [os.path.join(prefix, "share", "openbabel"),
                os.path.join(prefix, "share")]
        # `conda install openbabel` lands in <prefix>/share/openbabel, but
        # `pip install openbabel` keeps its data inside the package instead and
        # leaves <prefix>/share empty — so that path has to be searched too, or
        # a pip-only environment finds nothing and falls through to whatever
        # version the system happens to carry.
        out += sorted(glob.glob(os.path.join(
            prefix, "lib", "python*", "site-packages", "openbabel",
            "share", "openbabel")), reverse=True)
    out += ["/usr/local/share/openbabel", "/usr/share/openbabel",
            "/usr/local/share", "/usr/share"]
    return out


def openbabel_datadir(binary: Optional[str] = None) -> Optional[str]:
    """
    A directory containing UFF.prm, or None.

    gnina reads structures through Open Babel, and `--covalent_optimize_lig`
    asks it for a UFF force field. Open Babel finds its parameter files through
    BABEL_DATADIR, falling back to the path compiled into it — which is the
    build machine's, and usually does not exist here. When it cannot find them
    it prints "Cannot open UFF.prm" and carries on *without* optimising, so the
    poses come back unrelaxed and nothing in the exit status says so.

    Open Babel installs its data under a version directory, so the marker is
    searched for one level down as well as directly.
    """
    env = os.environ.get("BABEL_DATADIR")
    if env and os.path.isfile(os.path.join(env, BABEL_MARKER)):
        return env                                   # already set correctly

    for base in _datadir_candidates(binary):
        if not os.path.isdir(base):
            continue
        if os.path.isfile(os.path.join(base, BABEL_MARKER)):
            return base
        try:
            for entry in sorted(os.listdir(base), reverse=True):
                nested = os.path.join(base, entry)
                if os.path.isdir(nested) and \
                        os.path.isfile(os.path.join(nested, BABEL_MARKER)):
                    return nested
        except OSError:
            continue
    return None


def _child_env(binary: Optional[str]) -> dict:
    """The environment gnina is given, with BABEL_DATADIR filled in if needed."""
    env = dict(os.environ)
    if env.get("BABEL_DATADIR") and os.path.isfile(
            os.path.join(env["BABEL_DATADIR"], BABEL_MARKER)):
        return env                                   # the user set it; leave it
    found = openbabel_datadir(binary)
    if found:
        env["BABEL_DATADIR"] = found
    return env


def version(binary: str = "gnina") -> Optional[str]:
    path = shutil.which(binary) or binary
    try:
        proc = subprocess.run([path, "--version"], capture_output=True,
                              text=True, timeout=30)
    except Exception:                                         # noqa: BLE001
        return None
    text = (proc.stdout or proc.stderr).strip().splitlines()
    return text[0] if text else None


def run(config: GninaConfig,
        on_line: Optional[Callable[[str], None]] = None,
        timeout: Optional[int] = None) -> GninaResult:
    """Run gnina, streaming output to `on_line` as it arrives."""
    problems = config.validate()
    if problems:
        return GninaResult(False, -1, config.command_string(), "", "", config.out,
                           0.0, error="; ".join(problems))

    if find_binary((config.binary,)) is None:
        return GninaResult(
            False, -1, config.command_string(), "", "", config.out, 0.0,
            error=(f"'{config.binary}' was not found on PATH. Install gnina, or "
                   "copy the command below and run it where gnina is available."),
        )

    os.makedirs(os.path.dirname(os.path.abspath(config.out)), exist_ok=True)
    started = time.time()
    lines: List[str] = []
    try:
        proc = subprocess.Popen(config.command(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                env=_child_env(find_binary((config.binary,))))
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line.rstrip("\n"))
            if on_line:
                on_line(line.rstrip("\n"))
        proc.wait(timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        return GninaResult(False, -1, config.command_string(), "\n".join(lines), "",
                           config.out, time.time() - started,
                           error=f"gnina exceeded the {timeout}s time limit.")
    except Exception as exc:                                  # noqa: BLE001
        return GninaResult(False, -1, config.command_string(), "\n".join(lines), "",
                           config.out, time.time() - started, error=str(exc))

    elapsed = time.time() - started
    n_poses = 0
    if os.path.isfile(config.out):
        with open(config.out, errors="replace") as fh:
            n_poses = sum(1 for line in fh if line.strip() == "$$$$")

    ok = rc == 0 and n_poses > 0
    error = None
    if rc != 0:
        error = f"gnina exited with code {rc}."
    elif n_poses == 0:
        error = ("gnina finished but wrote no poses. The usual cause is a SMARTS "
                 "that matches nothing in the ligand file, or a receptor atom "
                 "selector that does not resolve.")
    return GninaResult(ok, rc, config.command_string(), "\n".join(lines), "",
                       config.out, elapsed, n_poses=n_poses, error=error)


def explain_failure(result: GninaResult) -> List[str]:
    """Turn common gnina errors into something actionable."""
    text = (result.stdout or "") + (result.error or "")
    low = text.lower()
    tips: List[str] = []
    if "could not find" in low and "atom" in low:
        tips.append("The receptor atom selector did not resolve. Check that the "
                    "chain ID, residue number and atom name match the prepared "
                    "receptor file exactly.")
    if "smarts" in low or "pattern" in low:
        tips.append("Check the ligand pattern against the prepared SDF — the "
                    "pattern must match the bound form being docked, not the "
                    "unreacted ligand.")
    if "cuda" in low or "gpu" in low:
        tips.append("GPU initialisation failed. Re-run with --no_gpu.")
    if BABEL_MARKER.lower() in low or "parseparamfile" in low:
        # Not fatal — gnina keeps going — but --covalent_optimize_lig silently
        # does nothing without the force field, so the poses are never relaxed
        # after the bond is made. Worth saying out loud rather than leaving it
        # in the log as noise.
        found = openbabel_datadir(find_binary((result.command.split()[0],)))
        if found:
            tips.append(
                f"Open Babel could not load {BABEL_MARKER} even though a data "
                f"directory exists at {found}. Set BABEL_DATADIR to it before "
                "launching, and check the file is readable.")
        else:
            tips.append(
                f"Open Babel cannot find its data files ({BABEL_MARKER}), so "
                "--covalent_optimize_lig did nothing and the bonded ligand was "
                "never relaxed. Locate the file with "
                "`find / -name UFF.prm 2>/dev/null` and export BABEL_DATADIR to "
                "the directory holding it, or install openbabel into the "
                "environment the app runs in.")
    if "box" in low:
        tips.append("The search box may not cover the covalent site.")
    if not tips and result.error:
        tips.append(result.error)
    return tips
