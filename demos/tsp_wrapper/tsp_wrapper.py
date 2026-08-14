"""
Travelling Salesman Problem — config-driven wrapper.

Covers "Part A: Direct QAOA" from divi-demos/travelling_salesman/
travelling_salesman.py. Reuses the actual functions unmodified:
generate_cities, compute_distance_matrix, build_tsp_qubo,
classical_brute_force, solve_with_qaoa, extract_best_tour, plot_tour.
"""

import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from tsp_original import (
    generate_cities,
    compute_distance_matrix,
    build_tsp_qubo,
    classical_brute_force,
    solve_with_qaoa,
    extract_best_tour,
    plot_tour,
)
from divi.backends import MaestroSimulator, QoroService, JobConfig


def _resolve_backend(cfg: dict):
    b = cfg["backend"]
    if b["use_cloud"]:
        return QoroService(job_config=JobConfig(shots=b["shots"]))
    return MaestroSimulator(shots=b["shots"])


def run_from_config(cfg: dict, progress_callback=None) -> dict:
    import time

    n_cities = cfg["n_cities"]
    seed = cfg["seed"]
    qaoa_cfg = cfg["qaoa"]
    backend = _resolve_backend(cfg)

    if progress_callback:
        progress_callback("Generating cities...")
    cities = generate_cities(n_cities, seed=seed)
    dist_matrix = compute_distance_matrix(cities)
    bqm, _ = build_tsp_qubo(dist_matrix)

    if progress_callback:
        progress_callback("Computing classical optimum...")
    classical_tour, classical_dist = classical_brute_force(dist_matrix)

    if progress_callback:
        backend_name = "QoroService" if cfg["backend"]["use_cloud"] else "local MaestroSimulator"
        progress_callback(f"Running QAOA on {backend_name}...")

    t0 = time.time()
    qaoa = solve_with_qaoa(
        bqm,
        n_layers=qaoa_cfg["n_layers"],
        max_iterations=qaoa_cfg["max_iterations"],
        shots=cfg["backend"]["shots"],
        backend=backend,
    )
    runtime = time.time() - t0

    quantum_tour, quantum_dist = extract_best_tour(qaoa, bqm, dist_matrix)

    plot_tour(cities, quantum_tour, quantum_dist, title="Quantum QAOA Tour", color="#4FC3F7")
    fig = plt.gcf()

    return {
        "classical_tour": classical_tour,
        "classical_distance": classical_dist,
        "quantum_tour": quantum_tour,
        "quantum_distance": quantum_dist,
        "ratio": quantum_dist / classical_dist if classical_dist else None,
        "runtime_s": runtime,
        "figure": fig,
    }
