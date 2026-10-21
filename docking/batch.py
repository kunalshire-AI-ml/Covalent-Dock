"""
batch.py — multi-ligand covalent docking pipeline.

Takes a list of ligands (SMILES or SDF files), a protein, and a nucleophile,
then runs the full pipeline for each ligand:
  detect warheads → check compatibility → build bound form → generate SMARTS
  → prepare receptor & ligand → run gnina → parse scores → build complex.

Results are collected into a summary table for comparison.
"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from chemistry import bound_form as bf_mod
from chemistry import compatibility as compat
from chemistry import ligand_reactivity as lig_chem
from chemistry import protein_reactivity as prot_chem
from chemistry import smarts_generator as smarts_mod
from chemistry.protein_reactivity import WATERS
from docking import complex_builder, gnina, prepare_ligand, prepare_receptor
from analysis import scores as scores_mod


@dataclass
class LigandJob:
    """One ligand's docking job specification."""
    name: str
    smiles: str = ""
    sdf_path: str = ""
    mol: object = None  # RDKit Mol


@dataclass
class LigandResult:
    """Result of docking one ligand."""
    name: str
    status: str = "pending"   # pending, setup, docking, done, failed
    error: str = ""
    warhead: str = ""
    mechanism: str = ""
    compatibility: str = ""
    smarts: str = ""
    n_poses: int = 0
    best_affinity: Optional[float] = None
    best_cnn: Optional[float] = None
    bond_distance: Optional[float] = None
    poses_path: str = ""
    complex_path: str = ""
    out_dir: str = ""
    seconds: float = 0.0

    def summary_row(self) -> Dict[str, object]:
        """Format as a row for display in a table."""
        return {
            "Ligand": self.name,
            "Status": self.status,
            "Warhead": self.warhead or "—",
            "Mechanism": self.mechanism or "—",
            "Match": self.compatibility or "—",
            "Poses": self.n_poses if self.n_poses else "—",
            "Best affinity (kcal/mol)": (f"{self.best_affinity:.2f}"
                                         if self.best_affinity is not None
                                         else "—"),
            "Best CNN": (f"{self.best_cnn:.3f}"
                         if self.best_cnn is not None else "—"),
            "Bond (Å)": (f"{self.bond_distance:.2f}"
                          if self.bond_distance is not None else "—"),
            "Time (s)": (f"{self.seconds:.1f}"
                          if self.seconds else "—"),
            "Error": self.error or "",
        }


def load_ligands_from_smiles(smiles_list: Sequence[str]) -> List[LigandJob]:
    """Create LigandJob entries from SMILES strings."""
    jobs = []
    for i, smi in enumerate(smiles_list):
        smi = smi.strip()
        if not smi or smi.startswith("#"):
            continue
        # Name can be specified as "SMILES name" (space-separated)
        parts = smi.split(None, 1)
        smile = parts[0]
        name = parts[1] if len(parts) > 1 else f"ligand_{i + 1}"
        jobs.append(LigandJob(name=name, smiles=smile))
    return jobs


def load_ligands_from_sdf(sdf_path: str) -> List[LigandJob]:
    """Load multiple molecules from a single SDF file."""
    jobs = []
    try:
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        for i, mol in enumerate(suppl):
            if mol is None:
                continue
            name = mol.GetProp("_Name") if mol.HasProp("_Name") and mol.GetProp("_Name").strip() else f"mol_{i + 1}"
            smi = Chem.MolToSmiles(mol)
            jobs.append(LigandJob(name=name, smiles=smi, mol=mol))
    except Exception:
        # Fallback: treat as single molecule
        jobs.append(LigandJob(name=os.path.basename(sdf_path), sdf_path=sdf_path))
    return jobs


def run_batch(
    ligand_jobs: List[LigandJob],
    pdb_text: str,
    structure: prot_chem.Structure,
    nucleophile: prot_chem.Nucleophile,
    base_dir: str,
    box_size: float = 25.0,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    seed: int = 42,
    cnn_scoring: str = "none",
    no_gpu: bool = True,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    on_result: Optional[Callable[[LigandResult], None]] = None,
) -> List[LigandResult]:
    """
    Run the full covalent docking pipeline for each ligand.

    Parameters
    ----------
    ligand_jobs   : list of LigandJob specs.
    pdb_text      : raw PDB text of the protein.
    structure     : parsed Structure object.
    nucleophile   : the selected protein nucleophile.
    base_dir      : root directory for all outputs.
    on_progress   : callback(current, total, message) for progress updates.
    on_result     : callback(result) called after each ligand finishes.

    Returns list of LigandResult for every ligand.
    """
    results: List[LigandResult] = []
    total = len(ligand_jobs)
    os.makedirs(base_dir, exist_ok=True)

    for idx, job in enumerate(ligand_jobs):
        res = LigandResult(name=job.name)
        out_dir = os.path.join(base_dir, f"ligand_{idx + 1}_{job.name}")
        # Clean name for directory
        out_dir = out_dir.replace(" ", "_").replace("/", "_").replace("\\", "_")
        os.makedirs(out_dir, exist_ok=True)
        res.out_dir = out_dir

        if on_progress:
            on_progress(idx + 1, total, f"Processing {job.name}…")

        try:
            # ── 1. Load ligand ──
            res.status = "setup"
            if job.mol is not None:
                loaded = lig_chem.LoadedLigand(
                    mol=job.mol, name=job.name, source_format="sdf",
                    has_3d=True, n_atoms=job.mol.GetNumAtoms(),
                    n_heavy=job.mol.GetNumHeavyAtoms())
            elif job.sdf_path:
                loaded = lig_chem.load_ligand_file(job.sdf_path)
            elif job.smiles:
                loaded = lig_chem.load_ligand_text(job.smiles, "smi", name=job.name)
            else:
                res.status = "failed"
                res.error = "No ligand source provided."
                results.append(res)
                continue

            if loaded.mol is None:
                res.status = "failed"
                res.error = "Could not parse ligand."
                results.append(res)
                continue

            # ── 2. Detect electrophiles and check compatibility ──
            els = lig_chem.find_electrophiles(loaded.mol)
            rows = compat.filter_for(nucleophile, els)
            if not rows:
                res.status = "failed"
                res.error = "No compatible electrophile found."
                results.append(res)
                if on_result:
                    on_result(res)
                continue

            el, verdict = rows[0]  # Best match
            res.warhead = el.warhead_label
            res.mechanism = el.mechanism_label
            res.compatibility = verdict.badge

            # ── 3. Build bound form ──
            bound = bf_mod.build_bound_form(
                loaded.mol, el.atom_idx, el.warhead_key,
                el.transform, el.leaving_atom, embed=True, random_seed=seed)

            # ── 4. Generate SMARTS ──
            sres = smarts_mod.generate_smarts(
                bound.mol, bound.reactive_idx, el.warhead_key, min_radius=3)
            res.smarts = sres.smarts

            # ── 5. Prepare receptor ──
            rec_path = os.path.join(out_dir, "receptor.pdb")
            rec = prepare_receptor.prepare_receptor(
                pdb_text, rec_path,
                remove_waters=True, remove_hydrogens=True,
                selector=nucleophile.selector)
            if rec.selector_ok is False:
                res.status = "failed"
                res.error = f"{nucleophile.selector} not in written receptor."
                results.append(res)
                if on_result:
                    on_result(res)
                continue

            # ── 6. Prepare ligand (embed + place near site) ──
            lig_path = os.path.join(out_dir, "ligand.sdf")
            protein_xyz = [a.xyz for a in structure.atoms
                           if a.resname not in WATERS]
            nuc_element = nucleophile.element or "S"
            prep = prepare_ligand.prepare_ligand(
                bound.mol, lig_path, name=job.name,
                reactive_idx=bound.reactive_idx, random_seed=seed,
                anchor_xyz=nucleophile.xyz,
                anchor_element=nuc_element,
                protein_xyz=protein_xyz)

            # Verify SMARTS against the written SDF
            check = prepare_ligand.verify_pattern_against_file(
                lig_path, sres.smarts)
            if not check["ok"]:
                res.status = "failed"
                res.error = f"SMARTS check failed: {check.get('error')}"
                results.append(res)
                if on_result:
                    on_result(res)
                continue

            # Adjust box size
            actual_box = box_size
            if prep.placement and prep.placement.required_box:
                if prep.placement.required_box > actual_box:
                    actual_box = prep.placement.required_box

            # ── 7. Run gnina ──
            res.status = "docking"
            if on_progress:
                on_progress(idx + 1, total, f"Docking {job.name}…")

            poses_path = os.path.join(out_dir, "poses.sdf")
            flex_path = os.path.join(out_dir, "gnina_complex.sdf")
            cfg = gnina.GninaConfig(
                receptor=rec_path, ligand=lig_path,
                out=poses_path,
                out_flex=flex_path,
                full_flex_output=True,
                log=os.path.join(out_dir, "gnina.log"),
                covalent_rec_atom=nucleophile.selector,
                covalent_lig_atom_pattern=sres.smarts,
                covalent_bond_order=el.bond_order,
                center=nucleophile.xyz, box_size=actual_box,
                exhaustiveness=exhaustiveness,
                num_modes=num_modes, seed=seed,
                cnn_scoring=cnn_scoring, no_gpu=no_gpu,
            )
            problems = cfg.validate()
            if problems:
                res.status = "failed"
                res.error = "; ".join(problems)
                results.append(res)
                if on_result:
                    on_result(res)
                continue

            dock_result = gnina.run(cfg)
            res.seconds = dock_result.seconds

            if not dock_result.ok:
                res.status = "failed"
                res.error = dock_result.error or "gnina failed"
                results.append(res)
                if on_result:
                    on_result(res)
                continue

            # ── 8. Parse scores ──
            poses = scores_mod.read_poses(poses_path)
            res.n_poses = len(poses)
            res.poses_path = poses_path
            if poses:
                summary = scores_mod.summarise(poses)
                res.best_affinity = summary.get("best_affinity")
                res.best_cnn = summary.get("best_cnn")

                # Write scores CSV
                rows = scores_mod.pose_table(poses)
                scores_mod.write_csv(rows, os.path.join(out_dir, "scores.csv"))

            # ── 9. Build complex ──
            # Ligand poses come from poses.sdf; the --out_flex file holds receptor
            # residues, not the ligand.
            complex_path = os.path.join(out_dir, "complex.pdb")
            try:
                merged = complex_builder.merge_complex(
                    rec_path, poses_path, complex_path, pose_index=0,
                    covalent_rec_atom=nucleophile.selector,
                    lig_smarts=sres.smarts)
                res.complex_path = merged.path
                if merged.bond:
                    res.bond_distance = merged.bond.distance
            except Exception as exc:                              # noqa: BLE001
                res.error = f"Complex not built: {exc}"

            res.status = "done"

        except Exception as exc:
            res.status = "failed"
            res.error = str(exc)
            # Save traceback for debugging
            tb_path = os.path.join(out_dir, "error.txt")
            try:
                with open(tb_path, "w") as f:
                    f.write(traceback.format_exc())
            except Exception:
                pass

        results.append(res)
        if on_result:
            on_result(res)

    return results


def results_table(results: List[LigandResult]) -> List[Dict[str, object]]:
    """Convert results to a list of row dicts for st.dataframe."""
    return [r.summary_row() for r in results]


def rank_results(results: List[LigandResult]) -> List[LigandResult]:
    """Sort results: successful first (by best affinity), then failed."""
    done = [r for r in results if r.status == "done" and r.best_affinity is not None]
    rest = [r for r in results if r not in done]
    done.sort(key=lambda r: r.best_affinity)  # More negative = better
    return done + rest
