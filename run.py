#!/usr/bin/env python3
"""EModelS end-to-end CLI: density map → model + HMM profiles → sequence hits.

Example
-------
python run.py \\
  --map examples/maps/emd_32336.map.gz \\
  --fasta-db /path/to/uniprot_trembl.fasta \\
  --output-dir outputs/emd_32336 \\
  --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import torch

# NumPy 1.24+ removed aliases used by older deps (open3d / sklearn).
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "complex"):
    np.complex = complex

warnings.filterwarnings("ignore")

from emodels.hmm.hmm_search import run_hmm_search
from emodels.solver import Solver
from emodels.unet3d.emmodel import ResUNet3D4EM, ResUNet3D4EM_new

ROOT = os.path.dirname(os.path.abspath(__file__))


def _default_model(name: str) -> str:
    return os.path.join(ROOT, "models", name)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EModelS: sequence identification from cryo-EM maps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = p.add_argument_group("Input / output")
    io.add_argument("--map", "--EM_map", dest="EM_map", required=True, help="Cryo-EM density map (.mrc/.map[.gz])")
    io.add_argument(
        "--fasta-db",
        dest="fasta_db",
        default="",
        help="Protein sequence database FASTA for hmmsearch (e.g. UniProtKB/TrEMBL)",
    )
    io.add_argument("--output-dir", "-o", default="./outputs", help="Output directory")
    io.add_argument(
        "--job-name",
        default="",
        help="Name used in output PDB / log filenames (default: inferred from map)",
    )
    io.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Map resolution (Å); optional metadata for logs",
    )

    stages = p.add_argument_group("Pipeline stages")
    stages.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip model/HMM building; only run search on existing hmm_profiles/",
    )
    stages.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip database search; only build model + HMM profiles",
    )
    stages.add_argument(
        "--no-write-reverse-hmm",
        action="store_false",
        dest="write_reverse_hmm",
        default=True,
        help="Do not write reverse-chain HMM profiles (default: write both forward and reverse)",
    )

    models = p.add_argument_group("Network weights")
    models.add_argument("--bb-model", default=_default_model("best_BB_model.ckpt"))
    models.add_argument("--ca-model", default=_default_model("best_CA_model.ckpt"))
    models.add_argument(
        "--aa-model",
        default=_default_model("best_AA_model.ckpt"),
        help="EModelS amino-acid model (contrastive-learning checkpoint)",
    )
    models.add_argument("--device", default="cuda", help="Torch device, e.g. cuda / cuda:0 / cpu")
    models.add_argument("--seed", type=int, default=2022)

    build = p.add_argument_group("Model-building parameters")
    build.add_argument("--cluster-eps", type=int, default=10)
    build.add_argument("--cluster-min-points", type=int, default=10)
    build.add_argument("--nms-radius", type=int, default=9)
    build.add_argument("--ca-score-thrh", type=float, default=0.35)
    build.add_argument("--frags-len", type=int, default=150)
    build.add_argument("--n-hop", type=int, default=6)
    build.add_argument("--neigh-mat-thrh", type=float, default=0.7)
    build.add_argument("--mul-proc-num", type=int, default=30)
    build.add_argument("--score-thrh", type=float, default=2.0)
    build.add_argument("--gap-len", type=int, default=3)
    build.add_argument("--struct-len", type=int, default=5)
    build.add_argument(
        "--hmm-delta",
        type=float,
        default=0.05,
        help="HMM transition delta (paper: δ=0.05)",
    )
    build.add_argument(
        "--hmm-gamma",
        type=float,
        default=0.5,
        help="HMM match-state floor gamma (paper: γ=0.5)",
    )

    search = p.add_argument_group("HMM search parameters")
    search.add_argument("--search-outdir", default="", help="Search output dir (default: <output-dir>/hmm_hits)")
    search.add_argument("--alphabet", default="amino", choices=["amino", "RNA", "DNA"])
    search.add_argument("--F1", type=float, default=0.02)
    search.add_argument("--F2", type=float, default=0.001)
    search.add_argument("--F3", type=float, default=1e-5)
    search.add_argument(
        "--E",
        type=float,
        default=10.0,
        help="HMMER reporting E-value cutoff used while collecting hits",
    )
    search.add_argument("--T", type=float, default=None, help="Optional HMMER bit-score cutoff")
    search.add_argument(
        "--min-profile-len",
        type=int,
        default=30,
        help="Skip HMM profiles shorter than this (residues)",
    )
    search.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Keep global top-K hits plus best hit per chain in best_hits.csv",
    )
    search.add_argument(
        "--hit-evalue",
        type=float,
        default=1e-10,
        help="Optional post-filter: write best_hits_strict.csv with E < this value",
    )
    return p


def _load_aa_checkpoint(model: torch.nn.Module, path: str, device: str) -> None:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = {
            k.replace("emmodel.", ""): v for k, v in ckpt["state_dict"].items()
        }
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(ckpt)


def _build_models(args: argparse.Namespace):
    for path, label in (
        (args.bb_model, "BB"),
        (args.ca_model, "CA"),
        (args.aa_model, "AA"),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing {label} model weights: {path}\n"
                "Download weights into ./models (see README / scripts/download_weights.sh)."
            )

    device = torch.device(args.device)
    bb = ResUNet3D4EM().to(device)
    ca = ResUNet3D4EM().to(device)
    aa = ResUNet3D4EM_new().to(device)
    bb.load_state_dict(torch.load(args.bb_model, map_location=device))
    ca.load_state_dict(torch.load(args.ca_model, map_location=device))
    _load_aa_checkpoint(aa, args.aa_model, str(device))
    bb.eval()
    ca.eval()
    aa.eval()
    return bb, ca, aa


def _solver_config(args: argparse.Namespace) -> SimpleNamespace:
    """Map CLI args onto the attribute names expected by Solver."""
    return SimpleNamespace(
        protocol="seq_free",
        EM_map=args.EM_map,
        fasta=None,
        job_name=args.job_name or None,
        template_dir="",
        download_afdb=False,
        run_pulchra=False,
        pulchra_path=None,
        run_phenix=False,
        resolution=args.resolution,
        phenix_act=None,
        phenix_param=None,
        output_dir=args.output_dir,
        seed=args.seed,
        cluster_eps=args.cluster_eps,
        cluster_min_points=args.cluster_min_points,
        nms_radius=args.nms_radius,
        CA_score_thrh=args.ca_score_thrh,
        frags_len=args.frags_len,
        n_hop=args.n_hop,
        neigh_mat_thrh=args.neigh_mat_thrh,
        mul_proc_num=args.mul_proc_num,
        score_thrh=args.score_thrh,
        gap_len=args.gap_len,
        struct_len=args.struct_len,
        afdb_allow_seq_id=0.6,
        write_reverse_hmm=args.write_reverse_hmm,
        hmm_delta=args.hmm_delta,
        hmm_gamma=args.hmm_gamma,
    )


def run_build(args: argparse.Namespace) -> str:
    os.makedirs(args.output_dir, exist_ok=True)
    bb, ca, aa = _build_models(args)
    torch.manual_seed(args.seed)
    cfg = _solver_config(args)
    result = Solver(cfg).run(bb, ca, aa)
    if result != "success":
        raise RuntimeError(f"EModelS build failed: {result}")
    return os.path.join(args.output_dir, "hmm_profiles")


def run_search(args: argparse.Namespace):
    if not args.fasta_db:
        raise ValueError("--fasta-db is required unless --skip-search is set")
    if not os.path.isfile(args.fasta_db):
        raise FileNotFoundError(f"--fasta-db not found: {args.fasta_db}")

    search_outdir = args.search_outdir or os.path.join(args.output_dir, "hmm_hits")
    hits = run_hmm_search(
        input_dir=args.output_dir,
        fasta_path=args.fasta_db,
        output_dir=search_outdir,
        alphabet=args.alphabet,
        F1=args.F1,
        F2=args.F2,
        F3=args.F3,
        E=args.E,
        T=args.T,
        min_profile_len=args.min_profile_len,
        top_k=args.top_k,
    )

    if args.hit_evalue is not None and hits is not None and not hits.empty:
        strict = hits[hits["E-value"] < args.hit_evalue].copy()
        strict_path = os.path.join(search_outdir, "best_hits_strict.csv")
        strict.to_csv(strict_path, index=False)
        print(
            f"[EModelS] Wrote {strict_path} "
            f"({len(strict)} hits with E < {args.hit_evalue:g})"
        )
    return hits


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.isfile(args.EM_map) and not args.skip_build:
        raise FileNotFoundError(f"--map not found: {args.EM_map}")
    if args.skip_build and args.skip_search:
        raise ValueError("Nothing to do: both --skip-build and --skip-search were set")

    if not args.skip_build:
        print("[EModelS] Building sequence-free model and HMM profiles...")
        run_build(args)
    else:
        print("[EModelS] Skipping build (--skip-build)")

    if not args.skip_search:
        print("[EModelS] Searching sequence database...")
        run_search(args)
    else:
        print("[EModelS] Skipping search (--skip-search)")
        print(f"[EModelS] HMM profiles are under {os.path.join(args.output_dir, 'hmm_profiles')}")

    print(f"[EModelS] Done. Outputs in {os.path.abspath(args.output_dir)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[EModelS] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
