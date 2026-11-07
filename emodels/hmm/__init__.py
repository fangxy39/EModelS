"""HMM profile writing and sequence-database search helpers."""

from .hmm_search import run_hmm_search

__all__ = ["aa_log_probs_to_hmm_file", "run_hmm_search"]


def __getattr__(name: str):
    if name == "aa_log_probs_to_hmm_file":
        from .aa_probs_to_hmm import aa_log_probs_to_hmm_file

        return aa_log_probs_to_hmm_file
    raise AttributeError(name)
