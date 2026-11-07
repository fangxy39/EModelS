"""FASTA helpers for HMM search."""

from __future__ import annotations

import os

from Bio import SeqIO


def remove_gaps(seq: str) -> str:
    return seq.replace("-", "").replace(".", "")


def write_fasta_no_gaps(fasta_input_path: str) -> str:
    new_file_name = os.path.splitext(fasta_input_path)[0] + "_no_gaps.fasta"
    with open(new_file_name, "w") as f:
        for record in SeqIO.parse(fasta_input_path, "fasta"):
            f.write(f">{record.description}\n")
            f.write(remove_gaps(str(record.seq)) + "\n")
    return new_file_name
