"""
Transverse-Field Ising Model (TFIM) spin dynamics via Divi.

Demonstrates ``TimeEvolutionTrajectory``: instead of looping over time points
one-by-one, the trajectory API creates one program per time point and executes
them together on the selected backend.

Tracks the magnetization ⟨Z₀⟩ for three physical regimes (ferromagnetic,
paramagnetic, Néel-state) and compares Exact Trotterization against QDrift.
"""

import time
from dataclasses import dataclass

import numpy as np
import pennylane as qml

from divi.backends import MaestroSimulator, QoroService, JobConfig
from divi.hamiltonians import ExactTrotterization, QDrift
from divi.qprog import (
    CustomPerQubitState,
    InitialState,
    TimeEvolutionTrajectory,
    ZerosState,
)
from plotting import absolute_trajectory_error, plot_dynamics, plot_trajectory_error


# ─────────────────────────────────────────────────────────────────────
#  Hamiltonian builder
# ─────────────────────────────────────────────────────────────────────

def build_tfim_hamiltonian(n_qubits: int, J: float, h: float) -> qml.Hamiltonian:
    """Build the Transverse-Field Ising Model (TFIM) Hamiltonian.

    H = -J Σ_i Z_i Z_{i+1}  -  h Σ_i X_i
    """
    coeffs, ops = [], []
    for i in range(n_qubits - 1):
        coeffs.append(-J)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    for i in range(n_qubits):
        coeffs.append(-h)
        ops.append(qml.PauliX(i))
    return qml.Hamiltonian(coeffs, ops)


# ─────────────────────────────────────────────────────────────────────
#  Run a trajectory (all time points in parallel)
# ─────────────────────────────────────────────────────────────────────

def run_trajectory(hamiltonian, time_points, strategy, backend,
                   n_steps=6, initial_state: InitialState | None = None):
    """Create and run a TimeEvolutionTrajectory, return sorted (times, mags)."""
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

    results = trajectory.aggregate_results()  # dict {t: <O>(t)}
    sorted_t = sorted(results.keys())
    return sorted_t, [results[t] for t in sorted_t]


# ─────────────────────────────────────────────────────────────────────
#  Experiment specs (declared once, looped over)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Experiment:
    name: str
    J: float
    h: float
    initial_state: InitialState
    filename: str


def run_experiment(
    exp: Experiment, n_qubits: int, n_steps: int, time_points, backend
) -> None:
    """Build H, run Exact + QDrift trajectories, save the comparison plot."""
    print(f"\n=== {exp.name} (J={exp.J}, h={exp.h}) ===")
    H = build_tfim_hamiltonian(n_qubits=n_qubits, J=exp.J, h=exp.h)

    t0 = time.time()
    t_exact, m_exact = run_trajectory(
        H, time_points, ExactTrotterization(), backend,
        n_steps=n_steps, initial_state=exp.initial_state,
    )
    print(f"  Exact   ({len(time_points)} points): {time.time() - t0:.1f}s")

    t0 = time.time()
    t_qdrift, m_qdrift = run_trajectory(
        H, time_points, QDrift(sampling_budget=10), backend,
        n_steps=n_steps, initial_state=exp.initial_state,
    )
    print(f"  QDrift  ({len(time_points)} points): {time.time() - t0:.1f}s")

    plot_dynamics(
        t_exact, m_exact, t_qdrift, m_qdrift,
        title=f"Spin Dynamics: {exp.name} (J={exp.J}, h={exp.h})",
        filename=exp.filename,
    )
    plot_trajectory_error(
        t_exact,
        absolute_trajectory_error(m_exact, m_qdrift),
        exp.name,
        exp.filename.replace(".png", "_error.png"),
    )


# =====================================================================
#  Main
# =====================================================================

if __name__ == "__main__":
    N_QUBITS = 6
    N_STEPS = 5
    T_MAX = 3.0
    N_POINTS = 6
    SHOTS = 5_000
    USE_CLOUD = False

    backend = (
        QoroService(job_config=JobConfig(shots=SHOTS))
        if USE_CLOUD
        else MaestroSimulator(shots=SHOTS)
    )

    # TimeEvolutionTrajectory needs t > 0 — skip t=0.
    time_points = np.linspace(0.01, T_MAX, N_POINTS).tolist()

    experiments = [
        Experiment(
            name="Ferromagnetic Phase",
            J=1.0, h=0.2,
            initial_state=ZerosState(),
            filename="dynamics_ferromagnetic.png",
        ),
        Experiment(
            name="Paramagnetic Phase",
            J=1.0, h=2.0,
            initial_state=ZerosState(),
            filename="dynamics_paramagnetic.png",
        ),
        Experiment(
            name="Néel State Dynamics",
            J=1.0, h=0.5,
            initial_state=CustomPerQubitState("101010"),
            filename="dynamics_neel.png",
        ),
    ]

    for exp in experiments:
        run_experiment(exp, N_QUBITS, N_STEPS, time_points, backend)
