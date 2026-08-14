"""
Quantum-Guided Cluster Algorithm — config-driven wrapper.

Reuses run_benchmark() from divi-demos/quantum_guided_cluster/main.py
unmodified — it already takes every problem-specific setting as a keyword
argument, so this wrapper just unpacks the data file into that call.
"""

import sys
import os
import glob

sys.path.insert(0, os.path.dirname(__file__))

from qgc_original import run_benchmark


def run_from_config(cfg: dict, progress_callback=None) -> dict:
    graph = cfg["graph"]
    algo = cfg["algorithm"]
    backend = cfg["backend"]

    if progress_callback:
        progress_callback(f"Running benchmark ({graph['n_nodes']} nodes, degree {graph['degree']})...")

    output_dir = "/tmp/qgc_plots"

    results = run_benchmark(
        n_nodes=graph["n_nodes"],
        degree=graph["degree"],
        qaoa_depths=cfg.get("qaoa_depths", [1, 2, 3]),
        pce_encodings=cfg.get("pce_encodings", []),
        n_iterations_factor=algo["n_iterations_factor"],
        n_repetitions=algo["n_repetitions"],
        lambda_scale=algo["lambda_scale"],
        seed=graph["seed"],
        use_cloud=backend["use_cloud"],
        shots=backend["shots"],
        output_dir=output_dir,
    )

    plot_paths = sorted(glob.glob(os.path.join(output_dir, "*.png")))

    return {
        "E_ground": results["E_ground"],
        "quantum_results": results["quantum_results"],
        "plot_paths": plot_paths,
    }
