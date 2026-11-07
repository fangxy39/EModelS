"""Small helpers used by HMM search / profile writing."""

from __future__ import annotations

import os
import sys


def assertion_check(if_statement: bool, failure_message: str = "") -> None:
    assert if_statement, failure_message


def is_relion_abort(directory: str) -> bool:
    return os.path.isfile(os.path.join(directory, "RELION_JOB_ABORT_NOW"))


def write_relion_job_exit_status(
    directory: str,
    status: str,
    pipeline_control: str = "",
) -> None:
    if pipeline_control != "":
        open(os.path.join(directory, f"RELION_JOB_EXIT_{status}"), "a").close()
    elif status == "FAILURE":
        sys.exit(1)


def abort_if_relion_abort(directory: str) -> None:
    if is_relion_abort(directory):
        write_relion_job_exit_status(directory, "ABORTED")
        print("Aborting now...")
        sys.exit(1)


def upper_and_lower_case_annotation(string: str) -> str:
    output = []
    for char in string:
        if char.isalpha():
            output.append("u" if char.isupper() else "l")
    return string + "_" + "".join(output)
