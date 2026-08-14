"""
Spin Dynamics (TFIM) — config-driven wrapper.

Reuses the ACTUAL functions from divi-demos/spin_dynamics/spin_dynamics.py
and plotting.py unmodified — build_tfim_hamiltonian, run_trajectory,
plot_dynamics, plot_trajectory_error. Only the __main__ block's hardcoded
constants are replaced with values read from a data file.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from spin_dynamics_original import build_tfim_hamiltonian, run_trajectory
from plotting import plot_dynamics, plot_trajectory_error, absolute_trajectory_error

from divi.backends import MaestroSimulator, QoroService, JobConfig
from divi.hamiltonians import ExactTrotterization, QDrift
from divi.qprog import CustomPerQubitState, ZerosState


def _resolve_initial_state(spec: str):
    return ZerosState() if spec == "zeros" else CustomPerQubitState(spec)


def _resolve_backend(cfg: dict):
    sim = cfg["simulation"]
    if cfg["backend"]["use_cloud"]:
        return QoroService(job_config=JobConfig(shots=sim["shots"]))
    return MaestroSimulator(shots=sim["shots"])


def run_from_config(cfg: dict, progress_callback=None) -> list[dict]:
    import time

    sim = cfg["simulation"]
    backend = _resolve_backend(cfg)
    time_points = np.linspace(0.01, sim["t_max"], sim["n_points"]).tolist()

    results = []
    experiments = cfg["experiments"]
    for i, exp in enumerate(experiments):
        if progress_callback:
            progress_callback(i, len(experiments), exp["name"])

        H = build_tfim_hamiltonian(sim["n_qubits"], exp["J"], exp["h"])
        init_state = _resolve_initial_state(exp["initial_state"])

        t0 = time.time()
        t_exact, m_exact = run_trajectory(
            H, time_points, ExactTrotterization(), backend,
            n_steps=sim["n_steps"], initial_state=init_state,
        )
        runtime_exact = time.time() - t0

        t0 = time.time()
        t_qdrift, m_qdrift = run_trajectory(
            H, time_points, QDrift(sampling_budget=10), backend,
            n_steps=sim["n_steps"], initial_state=init_state,
        )
        runtime_qdrift = time.time() - t0

        # plot_dynamics/plot_trajectory_error save PNGs via the real plotting.py
        # functions; grab the just-created figure right after each call so the
        # Streamlit app can also render it inline.
        plot_dynamics(
            t_exact, m_exact, t_qdrift, m_qdrift,
            title=f"{exp['name']} (J={exp['J']}, h={exp['h']})",
            filename=f"/tmp/{exp['name'].replace(' ', '_')}_dynamics.png",
        )
        fig_dynamics = plt.gcf()

        plot_trajectory_error(
            t_exact, absolute_trajectory_error(m_exact, m_qdrift), exp["name"],
            filename=f"/tmp/{exp['name'].replace(' ', '_')}_error.png",
        )
        fig_error = plt.gcf()

        results.append({
            "name": exp["name"],
            "fig_dynamics": fig_dynamics,
            "fig_error": fig_error,
            "runtime_exact_s": runtime_exact,
            "runtime_qdrift_s": runtime_qdrift,
        })

    return results
