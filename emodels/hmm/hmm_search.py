"""HMM profile search against a protein sequence database.

Hit reporting:

* skip profiles shorter than ``min_profile_len`` (default 30)
* ``best_hits.csv`` keeps the global top-``top_k`` hits **and** the
  best hit of every query chain (union)
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Optional

import numpy as np
import pandas as pd
import pyhmmer
import tqdm

from .fasta_utils import write_fasta_no_gaps
from .misc_utils import (
    abort_if_relion_abort,
    upper_and_lower_case_annotation,
    write_relion_job_exit_status,
)


def run_hmm_search(
    input_dir: str,
    fasta_path: str,
    output_dir: str = "hmm_hits",
    alphabet: str = "amino",
    F1: float = 0.02,
    F2: float = 0.001,
    F3: float = 1e-5,
    E: float = 10.0,
    T: Optional[float] = None,
    min_profile_len: int = 30,
    top_k: int = 10,
    pipeline_control: str = "",
) -> pd.DataFrame:
    """Search HMM profiles under ``input_dir`` against ``fasta_path``.

    Profiles are read from ``input_dir/hmm_profiles/*.hmm`` when present,
    otherwise from ``input_dir/*.hmm``.

    Returns
    -------
    pandas.DataFrame
        The filtered best-hit table written to ``best_hits.csv``.
    """
    hmm_profile_dir = os.path.join(input_dir, "hmm_profiles")
    if len(glob.glob(f"{hmm_profile_dir}/*.hmm")) == 0:
        if len(glob.glob(f"{input_dir}/*.hmm")) > 0:
            hmm_profile_dir = input_dir
    if not os.path.isdir(hmm_profile_dir):
        raise FileNotFoundError(
            f"No HMM profiles found under {input_dir}. "
            "Run EModelS model building first (writes hmm_profiles/)."
        )

    os.makedirs(output_dir, exist_ok=True)

    hmms = [
        (os.path.split(f)[-1].replace(".hmm", ""), pyhmmer.plan7.HMMFile(f).read())
        for f in sorted(glob.glob(f"{hmm_profile_dir}/*.hmm"))
    ]
    # Keep full profile stem (e.g. A, A_rev, Ar) so reverse chains stay distinct.

    alphabet_obj = {
        "amino": pyhmmer.easel.Alphabet.amino(),
        "RNA": pyhmmer.easel.Alphabet.rna(),
        "DNA": pyhmmer.easel.Alphabet.dna(),
    }[alphabet]

    pruned_hmms = [k for k in hmms if k[1].alphabet == alphabet_obj]
    run_set = [(n, h) for (n, h) in pruned_hmms if h.M >= min_profile_len]
    if not run_set:
        raise RuntimeError(
            f"No HMM profiles with length >= {min_profile_len} in {hmm_profile_dir}"
        )

    try:
        with pyhmmer.easel.SequenceFile(
            fasta_path, alphabet=alphabet_obj, digital=True
        ) as sf:
            digital_sequences = sf.read_block()
    except ValueError:
        new_fasta_path = write_fasta_no_gaps(fasta_path)
        with pyhmmer.easel.SequenceFile(
            new_fasta_path, alphabet=alphabet_obj, digital=True
        ) as sf:
            digital_sequences = sf.read_block()

    all_hits = pyhmmer.hmmer.hmmsearch(
        [hmm for (_, hmm) in run_set],
        digital_sequences,
        F1=F1,
        F2=F2,
        F3=F3,
        E=E,
        T=T,
    )

    hits_csv = {
        "target_name": [],
        "query_name": [],
        "accession": [],
        "E-value": [],
        "score": [],
        "bias": [],
        "description": [],
    }

    for hits, (name, _) in tqdm.tqdm(zip(all_hits, run_set), total=len(run_set)):
        annotation = upper_and_lower_case_annotation(name)
        if pipeline_control:
            abort_if_relion_abort(output_dir)
        with open(os.path.join(output_dir, f"{annotation}.hhr"), "wb") as f:
            hits.write(f)
        for hit in hits:
            if hit.evalue < E:
                try:
                    hits_csv["target_name"].append(hit.name.decode("utf-8"))
                    hits_csv["query_name"].append(name)
                    hits_csv["accession"].append(
                        hit.accession.decode("utf-8") if hit.accession else ""
                    )
                    hits_csv["E-value"].append(hit.evalue)
                    hits_csv["score"].append(hit.score)
                    hits_csv["bias"].append(hit.bias)
                    hits_csv["description"].append(hit.description.decode("utf-8"))
                except Exception:
                    pass
        try:
            msa = hits.to_msa(alphabet_obj)
            with open(os.path.join(output_dir, f"{annotation}.a2m"), "wb") as f:
                msa.write(f, "a2m")
        except Exception:
            pass

    hits_df = pd.DataFrame(hits_csv)
    hits_df.sort_values(by=["E-value"], inplace=True)
    hits_df.to_csv(os.path.join(output_dir, "all_hits.csv"), index=False)

    if hits_df.empty:
        filtered_df = hits_df
    else:
        hits = hits_df.copy()
        hits["E-value"] = pd.to_numeric(hits["E-value"], errors="coerce").fillna(
            np.inf
        )
        idx_top = hits["E-value"].nsmallest(top_k).index
        idx_best_per_query = pd.Index(hits.groupby("query_name")["E-value"].idxmin())
        keep_idx = idx_top.union(idx_best_per_query)
        filtered_df = (
            hits.loc[keep_idx]
            .sort_values("E-value", kind="mergesort")
            .reset_index(drop=True)
        )

    filtered_df.to_csv(os.path.join(output_dir, "best_hits.csv"), index=False)

    print("-" * 70)
    print("EModelS hmm_search completed successfully!")
    print(f"Profiles: {hmm_profile_dir}")
    print(f"Hits:     {output_dir}")
    print(f"  all_hits.csv  ({len(hits_df)} rows)")
    print(f"  best_hits.csv ({len(filtered_df)} rows; top-{top_k} ∪ best-per-chain)")
    print("-" * 70)

    if pipeline_control:
        write_relion_job_exit_status(
            output_dir, "SUCCESS", pipeline_control=pipeline_control
        )
    return filtered_df


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    main_args = parser.add_argument_group("Main arguments")
    main_args.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="Directory with hmm_profiles/ (or *.hmm files)",
    )
    main_args.add_argument(
        "--fasta-path",
        "-f",
        required=True,
        help="FASTA sequence database to search against",
    )
    main_args.add_argument(
        "--output-dir",
        "-o",
        default="hmm_hits",
        help="Directory for search results",
    )
    main_args.add_argument(
        "--alphabet",
        "-a",
        default="amino",
        choices=["amino", "RNA", "DNA"],
    )

    hmm_args = parser.add_argument_group("HMMSearch arguments")
    hmm_args.add_argument("--F1", type=float, default=0.02)
    hmm_args.add_argument("--F2", type=float, default=0.001)
    hmm_args.add_argument("--F3", type=float, default=1e-5)
    hmm_args.add_argument("--E", type=float, default=10)
    hmm_args.add_argument("--T", type=float, default=None)
    hmm_args.add_argument(
        "--min-profile-len",
        type=int,
        default=30,
        help="Skip HMM profiles shorter than this length",
    )
    hmm_args.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Keep global top-K hits in best_hits.csv (union with best-per-chain)",
    )
    parser.add_argument(
        "--pipeline-control",
        "--pipeline_control",
        type=str,
        default="",
        help=argparse.SUPPRESS,
    )
    return parser


def main(parsed_args: argparse.Namespace) -> None:
    run_hmm_search(
        input_dir=parsed_args.input_dir,
        fasta_path=parsed_args.fasta_path,
        output_dir=parsed_args.output_dir,
        alphabet=parsed_args.alphabet,
        F1=parsed_args.F1,
        F2=parsed_args.F2,
        F3=parsed_args.F3,
        E=parsed_args.E,
        T=parsed_args.T,
        min_profile_len=parsed_args.min_profile_len,
        top_k=parsed_args.top_k,
        pipeline_control=parsed_args.pipeline_control,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    main(add_args(parser).parse_args())
