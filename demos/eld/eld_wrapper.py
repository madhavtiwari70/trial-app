"""
Economic Load Dispatch — config-driven wrapper.

Reuses the ACTUAL functions from divi-demos/economic_load_dispatch/
economic_load_dispatch.py unmodified: build_qubo, classical_brute_force,
solve_with_pce, find_best_repaired_solution. Only the generator specs,
demand, PCE settings, and backend come from a data file. Covers the
3-generator scenario (the original demo's 6-generator variant is not
included here, to keep the config schema focused).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from eld_original import (
    build_qubo,
    classical_brute_force,
    solve_with_pce,
    find_best_repaired_solution,
)
from divi.backends import MaestroSimulator, QoroService, JobConfig


def _resolve_backend(cfg: dict):
    b = cfg["backend"]
    if b["use_cloud"]:
        return QoroService(job_config=JobConfig(shots=b["shots"]))
    return MaestroSimulator(shots=b["shots"])


def run_from_config(cfg: dict, progress_callback=None) -> dict:
    generators = cfg["generators"]
    demand = cfg["demand"]
    pce_cfg = cfg["pce"]
    backend = _resolve_backend(cfg)

    if progress_callback:
        progress_callback("Building QUBO...")
    bqm, var_names = build_qubo(generators, demand=demand)

    if progress_callback:
        progress_callback("Computing classical optimum...")
    classical_best = classical_brute_force(generators, demand, bqm)

    if progress_callback:
        backend_name = "QoroService" if cfg["backend"]["use_cloud"] else "local MaestroSimulator"
        progress_callback(f"Running PCE-VQE on {backend_name}...")

    import time
    t0 = time.time()
    pce_solver = solve_with_pce(
        bqm,
        n_layers=pce_cfg["n_layers"],
        max_iterations=pce_cfg["max_iterations"],
        alpha=pce_cfg["alpha"],
        backend=backend,
    )
    runtime = time.time() - t0

    result = find_best_repaired_solution(pce_solver, bqm, generators, demand)

    output = {
        "n_variables": len(var_names),
        "n_qubits": pce_solver.n_qubits,
        "classical_cost": classical_best[3],
        "classical_powers": classical_best[:3],
        "runtime_s": runtime,
        "quantum_result": None,
    }

    if result is not None:
        powers, cost, prob = result
        output["quantum_result"] = {
            "powers": powers,
            "cost": cost,
            "seed_probability": prob,
        }

    return output
