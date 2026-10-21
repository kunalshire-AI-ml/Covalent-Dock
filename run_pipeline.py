#!/usr/bin/env python3
"""
run_pipeline.py — the app's workflow without the UI.

Same chemistry engine as app.py: it finds the nucleophiles, finds the
electrophiles, checks they are compatible, builds the bound form, generates the
SMARTS and assembles the gnina command. With --run it calls gnina; without it,
it prints the command for you to run wherever gnina lives.

Examples
--------
List the candidate protein atoms and stop:

    python3 run_pipeline.py --protein 1abc.pdb --list-sites

List the warheads RDKit finds on a ligand and stop:

    python3 run_pipeline.py --ligand "C=CC(N1CCCCC1)=O" --list-warheads

Set up one docking run:

    python3 run_pipeline.py --protein 1abc.pdb --site A:12:SG \\
        --ligand "C=CC(N1CCCCC1)=O" --out runs/cys12_f1

Same, but actually dock and build the complex:

    python3 run_pipeline.py --protein 1abc.pdb --site A:12:SG \\
        --ligand fragment.sdf --out runs/cys12_f1 --run

Sweep one ligand across every cysteine in the structure:

    python3 run_pipeline.py --protein 1abc.pdb --all-cys \\
        --ligand "C=CC(N1CCCCC1)=O" --out runs/sweep --run
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import scores as scores_mod
from chemistry import bound_form as bf_mod
from chemistry import compatibility as compat
from chemistry import ligand_reactivity as lig_chem
from chemistry import protein_reactivity as prot_chem
from chemistry import smarts_generator as smarts_mod
from chemistry.protein_reactivity import NUCLEOPHILE_LABELS
from docking import complex_builder, gnina, prepare_ligand, prepare_receptor

GREEN, YEL, RED, DIM, RESET = (
    "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[0m")


def ok(msg: str) -> None:
    print(f"{GREEN}  ✓  {msg}{RESET}")


def warn(msg: str) -> None:
    print(f"{YEL}  ⚠  {msg}{RESET}")


def err(msg: str) -> None:
    print(f"{RED}  ✗  {msg}{RESET}")


def head(msg: str) -> None:
    print(f"\n{msg}\n{'─' * max(len(msg), 40)}")


# ═════════════════════════════════════════════════════════════════════════════
#  Loading
# ═════════════════════════════════════════════════════════════════════════════

def load_protein(path: str) -> prot_chem.Structure:
    with open(path) as fh:
        return prot_chem.parse_pdb(fh.read())


def load_ligand(spec: str) -> lig_chem.LoadedLigand:
    """A file path, or a bare SMILES string."""
    if os.path.isfile(spec):
        return lig_chem.load_ligand_file(spec)
    return lig_chem.load_ligand_text(spec, "smiles", name="ligand")


# ═════════════════════════════════════════════════════════════════════════════
#  Listing modes
# ═════════════════════════════════════════════════════════════════════════════

def list_sites(structure: prot_chem.Structure, limit: int = 60) -> None:
    nucs = prot_chem.rank_nucleophiles(prot_chem.find_nucleophiles(structure))
    head(f"Candidate covalent sites  ({len(nucs)} found)")
    print(f"  {'selector':16s} {'residue':12s} {'type':10s} notes")
    for n in nucs[:limit]:
        notes = []
        if n.disulfide_partner:
            notes.append(f"disulfide with {n.disulfide_partner}")
        if n.neighbor_count:
            notes.append(f"{n.neighbor_count} neighbours")
        if n.activating_contacts:
            notes.append(f"{n.activating_contacts} basic contacts")
        print(f"  {n.selector:16s} {n.label:12s} "
              f"{NUCLEOPHILE_LABELS[n.nucleophile_type]:10s} "
              f"{DIM}{'; '.join(notes)}{RESET}")
    if len(nucs) > limit:
        print(f"  … and {len(nucs) - limit} more")
    print(f"\n{DIM}  These are chemically plausible, not confirmed reactive."
          f"{RESET}")


def list_warheads(loaded: lig_chem.LoadedLigand) -> None:
    els = lig_chem.find_electrophiles(loaded.mol)
    head(f"Detected warheads  ({len(els)} reactive atoms)")
    if not els:
        warn("No warhead matched. The ligand may be non-covalent, or already "
             "in its bound form.")
        return
    print(f"  {'atom':6s} {'warhead':30s} {'role':16s} {'conf':7s} nucleophiles")
    for e in els:
        print(f"  {e.label:6s} {e.warhead_label:30s} {e.role_label:16s} "
              f"{e.confidence:7s} {', '.join(e.nucleophiles)}")


# ═════════════════════════════════════════════════════════════════════════════
#  One docking set-up
# ═════════════════════════════════════════════════════════════════════════════

def setup_one(args, structure, pdb_text, loaded, nuc, out_dir) -> Optional[dict]:
    os.makedirs(out_dir, exist_ok=True)
    mol = loaded.mol

    # ── electrophile, filtered by what this nucleophile can attack ───────────
    els = lig_chem.find_electrophiles(mol)
    rows = compat.filter_for(nuc, els)
    if not rows:
        err(f"{nuc.selector}: no ligand atom is compatible with a "
            f"{NUCLEOPHILE_LABELS[nuc.nucleophile_type]}.")
        return None

    if args.ligand_atom:
        picked = [(e, v) for e, v in rows if e.label == args.ligand_atom]
        if not picked:
            err(f"Atom {args.ligand_atom} is not among the compatible atoms: "
                f"{', '.join(e.label for e, _ in rows)}")
            return None
        el, verdict = picked[0]
    else:
        el, verdict = rows[0]

    ok(f"{nuc.label}  +  {el.label} ({el.warhead_label}) "
       f"→ {verdict.badge}, bond {verdict.bond}")
    for caveat in verdict.caveats:
        warn(caveat)

    # ── bound form: gnina does not model the reaction ────────────────────────
    bound = bf_mod.build_bound_form(mol, el.atom_idx, el.warhead_key,
                                    leaving_atom=el.leaving_atom)
    if bound.changed:
        ok(f"bound form: {bound.smiles}   ({bound.description})")
    else:
        print(f"  {DIM}ligand used as supplied — {bound.description}{RESET}")
    for w in bound.warnings:
        warn(w)

    # ── SMARTS from the chemical environment, not the atom index ─────────────
    sres = smarts_mod.generate_smarts(bound.mol, bound.reactive_idx,
                                      el.warhead_key,
                                      min_radius=args.min_radius)
    pattern = args.smarts or sres.smarts
    if args.smarts:
        warn(f"using your override: {pattern}")
    else:
        ok(f"SMARTS: {pattern}   ({sres.n_matches} match"
           f"{'' if sres.n_matches == 1 else 'es'}, radius {sres.radius})")
        if not sres.unique:
            warn("The pattern matches more than one atom. gnina will try each "
                 "match, so you will get poses bonded through every one.")

    # ── files ────────────────────────────────────────────────────────────────
    rec_path = os.path.join(out_dir, "receptor.pdb")
    rec = prepare_receptor.prepare_receptor(
        pdb_text, rec_path,
        keep_chains=[nuc.chain] if not args.keep_all_chains else None,
        remove_waters=not args.keep_waters,
        remove_hetero=args.remove_hetero,
        selector=nuc.selector)
    if rec.selector_ok is False:
        err(f"{nuc.selector} is not in the written receptor — gnina will fail.")
        return None
    ok(f"receptor: {rec.report()}")
    for w in rec.warnings:
        warn(w)

    lig_path = os.path.join(out_dir, "ligand.sdf")
    # Bring the ligand to the covalent site: it is embedded in its own frame,
    # so without this the search box centred on `nuc` may not contain it.
    protein_xyz = [a.xyz for a in structure.atoms
                   if a.resname not in prot_chem.WATERS]
    prep = prepare_ligand.prepare_ligand(bound.mol, lig_path,
                                         name=loaded.name,
                                         reactive_idx=bound.reactive_idx,
                                         anchor_xyz=nuc.xyz,
                                         anchor_element=nuc.element,
                                         protein_xyz=protein_xyz)
    ok(f"ligand: {prep.n_atoms} atoms, {prep.method}")
    if prep.placement and prep.placement.moved:
        ok(f"placed at the site: {prep.placement.start_distance:.1f} Å → "
           f"{prep.placement.final_distance:.2f} Å from {nuc.selector}")
    for w in prep.warnings:
        warn(w)

    # A tethered ligand pivots about the receptor atom, so the box has to cover
    # its whole reach; too small and gnina scores a pose it never really sampled.
    box_size = args.box_size
    required = prep.placement.required_box if prep.placement else None
    if required and required > box_size:
        if args.no_autobox:
            warn(f"the placed ligand needs ~{required:.0f} Å of box but the box "
                 f"is {box_size:.0f} Å — expect 'ligand outside box'")
        else:
            box_size = required
            ok(f"box grown {args.box_size:.0f} Å → {box_size:.0f} Å to fit the "
               "placed ligand")

    # the check that matters — does the pattern match the file gnina reads?
    check = prepare_ligand.verify_pattern_against_file(lig_path, pattern)
    if not check["ok"]:
        err(f"pattern does not match the written SDF: {check['error']}")
        return None
    ok(f"pattern verified against {os.path.basename(lig_path)} "
       f"({check['n_matches']} match)")

    # ── command ──────────────────────────────────────────────────────────────
    cfg = gnina.GninaConfig(
        receptor=rec_path, ligand=lig_path,
        out=os.path.join(out_dir, "poses.sdf"),
        out_flex=os.path.join(out_dir, "gnina_complex.sdf"),
        full_flex_output=True,
        log=os.path.join(out_dir, "gnina.log"),
        covalent_rec_atom=nuc.selector,
        covalent_lig_atom_pattern=pattern,
        covalent_bond_order=el.bond_order,
        covalent_optimize_lig=not args.no_optimize_lig,
        center=nuc.xyz, box_size=box_size,
        exhaustiveness=args.exhaustiveness, num_modes=args.num_modes,
        cnn_scoring=args.cnn_scoring, seed=args.seed, no_gpu=True,
    )
    problems = cfg.validate()
    if problems:
        for p in problems:
            err(p)
        return None

    cmd_path = os.path.join(out_dir, "gnina_command.sh")
    with open(cmd_path, "w") as fh:
        fh.write("#!/bin/sh\n" + cfg.command_string() + "\n")
    os.chmod(cmd_path, 0o755)

    head("gnina command")
    print(cfg.command_string())
    print(f"\n{DIM}  saved to {cmd_path}{RESET}")

    return {"config": cfg, "nucleophile": nuc, "electrophile": el,
            "reactive_idx": prep.reactive_idx,
            "out_dir": out_dir, "receptor": rec_path}


def dock_one(job: dict, args) -> None:
    cfg = job["config"]
    head(f"Running gnina → {os.path.basename(job['out_dir'])}")
    result = gnina.run(cfg, on_line=lambda line: print("   " + line))
    if not result.ok:
        err(result.error or "gnina failed.")
        for hint in gnina.explain_failure(result):
            warn(hint)
        return
    ok(f"finished in {result.seconds:.1f} s")

    poses = scores_mod.read_poses(cfg.out)
    if not poses:
        warn("gnina wrote no poses.")
        return
    summary = scores_mod.summarise(poses)
    ok(f"{summary['n_poses']} poses · best affinity "
       f"{summary['best_affinity']} kcal/mol · best CNN {summary['best_cnn']}")

    rows = scores_mod.pose_table(poses)
    csv_path = os.path.join(job["out_dir"], "scores.csv")
    scores_mod.write_csv(rows, csv_path)
    ok(f"scores → {csv_path}")
    try:
        xlsx_path = os.path.join(job["out_dir"], "scores.xlsx")
        scores_mod.write_xlsx(rows, xlsx_path)
        ok(f"scores → {xlsx_path}")
    except Exception as exc:                                  # noqa: BLE001
        warn(f"xlsx export skipped ({exc})")

    # The bonded ligand atom is read from the pose (the atom bonded to gnina's
    # copy of the receptor atom), not from an input-file index: gnina reorders
    # atoms, and its poses carry the reacting residue, whose copy of the
    # receptor atom would otherwise be taken as the ligand atom at 0.00 Å.
    nuc = job["nucleophile"]
    complex_path = os.path.join(job["out_dir"], "complex.pdb")
    merged = complex_builder.merge_complex(
        job["receptor"], cfg.out, complex_path, pose_index=0,
        covalent_rec_atom=nuc.selector,
        lig_smarts=cfg.covalent_lig_atom_pattern,
        remarks=[f"Covalent docking: {nuc.selector} + "
                 f"{job['electrophile'].warhead_label}",
                 "Candidate covalent site — not an experimentally "
                 "confirmed adduct."])
    bond = merged.bond
    ok(f"complex → {complex_path}  (bond {bond.rec_atom}–{bond.lig_atom} "
       f"{bond.distance:.2f} Å)" if bond else f"complex → {complex_path}")
    for w in merged.warnings:
        warn(w)

    if args.all_poses:
        out_dir = os.path.join(job["out_dir"], "complexes")
        n_poses = complex_builder.count_poses(cfg.out)
        for i in range(n_poses):
            built = complex_builder.merge_complex(
                job["receptor"], cfg.out,
                os.path.join(out_dir, f"complex_pose{i + 1}.pdb"), pose_index=i,
                covalent_rec_atom=nuc.selector,
                lig_smarts=cfg.covalent_lig_atom_pattern)
            for w in built.warnings:
                warn(f"pose {i + 1}: {w}")
        ok(f"{n_poses} pose complexes → {out_dir}/")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Covalent docking set-up, headless.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples")[-1])
    p.add_argument("--protein", help="receptor PDB")
    p.add_argument("--ligand", help="SDF/MOL/MOL2/PDB file, or a SMILES string")

    p.add_argument("--site", help="protein atom, e.g. A:12:SG")
    p.add_argument("--all-cys", action="store_true",
                   help="one run per free cysteine")
    p.add_argument("--all-sites", action="store_true",
                   help="one run per candidate nucleophile (can be many)")
    p.add_argument("--ligand-atom",
                   help="ligand atom label, e.g. C25 (default: best match)")

    p.add_argument("--list-sites", action="store_true")
    p.add_argument("--list-warheads", action="store_true")

    p.add_argument("--out", default="runs/cli", help="output directory")
    p.add_argument("--run", action="store_true", help="actually call gnina")
    p.add_argument("--all-poses", action="store_true",
                   help="write a complex PDB for every pose, not just the best")

    p.add_argument("--smarts", help="override the generated pattern")
    p.add_argument("--min-radius", type=int, default=3,
                   help="bonds of context in the generated pattern")

    p.add_argument("--keep-waters", action="store_true")
    p.add_argument("--keep-all-chains", action="store_true")
    p.add_argument("--remove-hetero", action="store_true")

    p.add_argument("--exhaustiveness", type=int, default=16)
    p.add_argument("--num-modes", type=int, default=9)
    p.add_argument("--box-size", type=float, default=20.0,
                   help="minimum box edge; grown to fit the placed ligand "
                        "unless --no-autobox")
    p.add_argument("--no-autobox", action="store_true",
                   help="keep --box-size even if the ligand does not fit in it")
    p.add_argument("--cnn-scoring", default="none",
                   choices=["none", "rescore", "refinement", "all"],
                   help="off by default: the deployment host has no GPU")
    p.add_argument("--no-optimize-lig", action="store_true")
    p.add_argument("--no-gpu", action="store_true",
                   help="always on — gnina is run with --no_gpu regardless")
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> int:
    args = build_parser().parse_args()

    structure = pdb_text = loaded = None
    if args.protein:
        if not os.path.isfile(args.protein):
            err(f"No such file: {args.protein}")
            return 1
        pdb_text = open(args.protein).read()
        structure = prot_chem.parse_pdb(pdb_text)
        ok(f"{args.protein}: {len(structure.chains)} chain(s), "
           f"{sum(len(c.residues) for c in structure.chains)} residues")

    if args.ligand:
        loaded = load_ligand(args.ligand)
        info = lig_chem.describe_ligand(loaded.mol)
        ok(f"ligand: {info['smiles']}  ({info['heavy_atoms']} heavy atoms, "
           f"MW {info['mw']})")
        for w in loaded.warnings:
            warn(w)

    if args.list_sites:
        if structure is None:
            err("--list-sites needs --protein")
            return 1
        list_sites(structure)
        return 0

    if args.list_warheads:
        if loaded is None:
            err("--list-warheads needs --ligand")
            return 1
        list_warheads(loaded)
        return 0

    if structure is None or loaded is None:
        err("Give both --protein and --ligand (or use a --list-* mode).")
        return 1

    # ── which sites ──────────────────────────────────────────────────────────
    nucs = prot_chem.rank_nucleophiles(prot_chem.find_nucleophiles(structure))
    targets: List[prot_chem.Nucleophile] = []
    if args.site:
        targets = [n for n in nucs if n.selector == args.site]
        if not targets:
            err(f"{args.site} is not a candidate nucleophile. "
                "Run with --list-sites to see the options.")
            return 1
    elif args.all_cys:
        targets = [n for n in nucs
                   if n.nucleophile_type == "thiol" and not n.disulfide_partner]
        ok(f"{len(targets)} free cysteine(s)")
    elif args.all_sites:
        targets = nucs
    else:
        err("Choose --site A:12:SG, or --all-cys, or --all-sites.")
        return 1

    jobs = []
    for nuc in targets:
        out_dir = (args.out if len(targets) == 1
                   else os.path.join(args.out, nuc.selector.replace(":", "_")))
        head(f"{nuc.label}   →   {out_dir}")
        job = setup_one(args, structure, pdb_text, loaded, nuc, out_dir)
        if job:
            jobs.append(job)

    if not jobs:
        err("Nothing was set up.")
        return 1

    if not args.run:
        print(f"\n{DIM}  Set-up only. Add --run to call gnina, or run the "
              f"saved gnina_command.sh yourself.{RESET}")
        return 0

    if gnina.find_binary() is None:
        err("gnina is not on PATH. The commands are saved — run them where "
            "gnina is installed.")
        return 1

    for job in jobs:
        dock_one(job, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
