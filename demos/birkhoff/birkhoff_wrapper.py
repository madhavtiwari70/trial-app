"""
Minimum Birkhoff Decomposition — config-driven wrapper.

The original demo (birkhoff_main_original.py, copied unmodified from
divi-demos/minimum_birkhoff_decomposition/main.py) is already
argument-driven via argparse. This wrapper just builds an args-like
object from the data file and calls its main() directly — no logic is
duplicated. Terminal output (including the matrix breakdown) is
captured and returned as text for display.
"""

import sys
import os
import io
import contextlib
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))

import birkhoff_main_original as bmo


def run_from_config(cfg: dict, progress_callback=None) -> dict:
    args = SimpleNamespace(
        dim=cfg["dim"],
        comb=cfg["comb"],
        matrix_type=cfg["matrix_type"],
        instance=cfg["instance"],
        iterations=cfg["iterations"],
        optimizer=cfg["optimizer"],
    )

    if progress_callback:
        progress_callback(f"Running Birkhoff decomposition (n={args.dim}, k={args.comb})...")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bmo.main(args)

    output_text = buf.getvalue()

    return {"output_text": output_text}
