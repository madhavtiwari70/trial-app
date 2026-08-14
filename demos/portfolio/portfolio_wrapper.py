"""
Portfolio Optimization — config-driven wrapper.

Covers the small synthetic-portfolio part of divi-demos/portfolio_optimization/
portfolio_optimization.ipynb. Reuses utils.py's build_full_portfolio_qubo
and evaluate_solution unmodified — only n_assets, seed, lambda, and QAOA
settings come from a data file.
"""

import sys
import os
import io
import contextlib

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import dimod

from utils import build_full_portfolio_qubo, evaluate_solution
from divi.backends import MaestroSimulator
from divi.qprog import QAOA
from divi.qprog.problems import BinaryOptimizationProblem
from divi.qprog.optimizers import MonteCarloOptimizer


def run_from_config(cfg: dict, progress_callback=None) -> dict:
    import time

    np.random.seed(cfg["seed"])
    n_assets = cfg["n_assets"]

    if progress_callback:
        progress_callback(f"Generating synthetic portfolio ({n_assets} assets)...")

    returns = np.random.uniform(0.01, 0.1, n_assets)
    A = np.random.randn(n_assets, n_assets) * 0.1
    covariance = A @ A.T + np.eye(n_assets) * 0.01

    qubo_matrix = build_full_portfolio_qubo(returns, covariance, cfg["lambda_param"])
    bqm = dimod.BinaryQuadraticModel(qubo_matrix, "BINARY")

    qaoa_cfg = cfg["qaoa"]
    backend = MaestroSimulator(shots=cfg["backend"]["shots"])
    optim = MonteCarloOptimizer(
        population_size=qaoa_cfg["population_size"],
        n_best_sets=qaoa_cfg["n_best_sets"],
    )

    if progress_callback:
        progress_callback("Running QAOA...")

    t0 = time.time()
    qaoa = QAOA(
        problem=BinaryOptimizationProblem(bqm),
        n_layers=qaoa_cfg["n_layers"],
        optimizer=optim,
        max_iterations=qaoa_cfg["max_iterations"],
        backend=backend,
    )
    qaoa.run()
    runtime = time.time() - t0

    qaoa_solution = np.array(qaoa.solution, dtype=int)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        evaluate_solution(qubo_matrix, qaoa_solution, returns, covariance)
    eval_text = buf.getvalue()

    return {
        "n_assets": n_assets,
        "assets_selected": np.where(qaoa_solution == 1)[0].tolist(),
        "runtime_s": runtime,
        "evaluation_text": eval_text,
    }
