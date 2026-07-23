"""
Spin Dynamics (TFIM) — refactored as a config-driven library.

This is the *same* Divi logic as the original spin_dynamics.py demo script.
The only change: instead of constants hardcoded in `if __name__ == "__main__"`,
everything problem-specific is read from a data file (data/spin_dynamics.yaml).

To make a new demo variant: edit the YAML file. Never edit this file.
"""

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import yaml

from divi.backends import QiskitSimulator, QoroService, JobConfig
from divi.hamiltonians import ExactTrotterization, QDrift
from divi.qprog import (
    CustomPerQubitState,
    InitialState,
    TimeEvolutionTrajectory,
    ZerosState,
)


# ─────────────────────────────────────────────────────────────────────
#  Config loading
# ─────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_initial_state(spec: str) -> InitialState:
    """'zeros' -> ZerosState(); a bitstring like '101010' -> CustomPerQubitState."""
    if spec == "zeros":
        return ZerosState()
    return CustomPerQubitState(spec)


def _resolve_backend(cfg: dict):
    sim = cfg["simulation"]
    if cfg["backend"]["use_cloud"]:
        return QoroService(job_config=JobConfig(shots=sim["shots"]))
    return QiskitSimulator(shots=sim["shots"])


# ─────────────────────────────────────────────────────────────────────
#  Hamiltonian builder  (unchanged from the original demo)
# ─────────────────────────────────────────────────────────────────────

def build_tfim_hamiltonian(n_qubits: int, J: float, h: float) -> qml.Hamiltonian:
    """H = -J * sum_i Z_i Z_{i+1}  -  h * sum_i X_i"""
    coeffs, ops = [], []
    for i in range(n_qubits - 1):
        coeffs.append(-J)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    for i in range(n_qubits):
        coeffs.append(-h)
        ops.append(qml.PauliX(i))
    return qml.Hamiltonian(coeffs, ops)


# ─────────────────────────────────────────────────────────────────────
#  Run a trajectory  (unchanged from the original demo)
# ─────────────────────────────────────────────────────────────────────

def run_trajectory(hamiltonian, time_points, strategy, backend,
                    n_steps=6, initial_state: InitialState | None = None):
    trajectory = TimeEvolutionTrajectory(
        hamiltonian=hamiltonian,
        time_points=time_points,
        observable=qml.PauliZ(0),
        backend=backend,
        trotterization_strategy=strategy,
        n_steps=n_steps,
        order=1,
        initial_state=initial_state or ZerosState(),
    )
    trajectory.create_programs()
    trajectory.run(blocking=True)
    results = trajectory.aggregate_results()
    sorted_t = sorted(results.keys())
    return sorted_t, [results[t] for t in sorted_t]


def plot_dynamics(times_exact, mag_exact, times_qdrift, mag_qdrift, title):
    """Return a matplotlib Figure (instead of saving to disk) for Streamlit."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(times_exact, mag_exact, "o-", color="#3b82f6",
            label="Exact Trotterization", linewidth=2, markersize=7)
    ax.plot(times_qdrift, mag_qdrift, "s--", color="#f97316",
            label="QDrift (stochastic)", linewidth=2, markersize=7)
    ax.axhline(0, color="black", linestyle="-", alpha=0.2)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel(r"Magnetization $\langle Z_0 \rangle$")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


@dataclass
class ExperimentResult:
    name: str
    J: float
    h: float
    figure: object
    runtime_exact_s: float
    runtime_qdrift_s: float


# ─────────────────────────────────────────────────────────────────────
#  Entry point used by the Streamlit app
# ─────────────────────────────────────────────────────────────────────

def run_from_config(cfg: dict, progress_callback=None) -> list[ExperimentResult]:
    """Run every experiment defined in the data file. Returns one result per experiment."""
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

        fig = plot_dynamics(
            t_exact, m_exact, t_qdrift, m_qdrift,
            title=f"{exp['name']} (J={exp['J']}, h={exp['h']})",
        )

        results.append(ExperimentResult(
            name=exp["name"], J=exp["J"], h=exp["h"],
            figure=fig,
            runtime_exact_s=runtime_exact,
            runtime_qdrift_s=runtime_qdrift,
        ))

    return results
