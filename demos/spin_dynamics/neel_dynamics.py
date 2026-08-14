# =============================================================================
#  Néel State Dynamics — QASM Export / Reload Demo
# =============================================================================
#
#  Two-phase workflow:
#
#    Phase 1 (export):  Build Trotter circuits and save them as QASM files.
#       python neel_dynamics.py export --n-qubits 15
#
#    Run:               Run exact baseline, reload (compressed) QASM circuits,
#                       run QDrift, and compare all three.
#       python neel_dynamics.py run --n-qubits 15 --compressed-dir compressed_circuits
#
#  The phases are independent — you can export, compress the QASM files
#  externally, and run later.
#
# =============================================================================

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

from divi.backends import MaestroSimulator, QoroService, JobConfig
from divi.circuits import qscript_to_meta, dag_to_qasm_body
from divi.qprog import CustomPerQubitState, TimeEvolution
from divi.hamiltonians import ExactTrotterization, QDrift


# ── Hamiltonian ──────────────────────────────────────────────────────

def build_tfim_hamiltonian(n_qubits: int, J: float, h: float) -> qml.Hamiltonian:
    """Build the Transverse-Field Ising Model (TFIM) Hamiltonian.

    H = -J * Sum_{i}(Z_i Z_{i+1}) - h * Sum_{i}(X_i)
    """
    coeffs, ops = [], []
    for i in range(n_qubits - 1):
        coeffs.append(-J)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    for i in range(n_qubits):
        coeffs.append(-h)
        ops.append(qml.PauliX(i))
    return qml.Hamiltonian(coeffs, ops)


# ── Circuit builder ──────────────────────────────────────────────────

def build_neel_circuit_tape(hamiltonian, n_qubits, neel_state, time, n_steps, order=1):
    """Build a PennyLane QuantumScript for Néel-state time evolution.

    Returns a tape with initial-state prep + Trotter evolution + probs measurement.
    """
    ops = []

    # Initial state: Néel pattern (e.g. |101010...>)
    for w, char in enumerate(neel_state):
        if char == "1":
            ops.append(qml.PauliX(wires=w))

    # Trotterised evolution e^{-iHt}
    n_terms = len(hamiltonian)
    if n_terms >= 2:
        ops.append(
            qml.adjoint(
                qml.TrotterProduct(hamiltonian, time=time, n=n_steps, order=order)
            )
        )
    else:
        ops.append(qml.evolve(hamiltonian, coeff=time))

    return qml.tape.QuantumScript(ops=ops, measurements=[qml.probs()])


# ── Helper: load QASM folder and compute <Z_0> ───────────────────────

def load_and_execute_qasm(qasm_dir, times, backend, neel_state, n_qubits):
    """Load QASM files from a directory, execute them, and return magnetizations.

    Returns:
        list of <Z_0> values, one per time point (including t=0).
    """
    circuits = {}
    time_by_label = {}
    for i, t in enumerate(times):
        if t == 0.0:
            continue
        filepath = os.path.join(qasm_dir, f"neel_t{i:03d}.qasm")
        if not os.path.exists(filepath):
            print(f"   ⚠️  Missing: {filepath}")
            continue
        label = f"neel_t{i:03d}"
        with open(filepath, "r") as f:
            circuits[label] = f.read()
        time_by_label[label] = t

    print(f"   📦 Loaded {len(circuits)} QASM circuits from {qasm_dir}/")
    print(f"   🚀 Submitting to backend...")

    result = backend.submit_circuits(circuits)

    # Cloud backend: poll for completion and fetch results if needed
    if result is not None and result.results is None and hasattr(result, 'job_id') and result.job_id:
        print(f"   ⏳ Job {result.job_id} submitted, polling for completion...")
        backend.poll_job_status(result, loop_until_complete=True, verbose=True)
        result = backend.get_job_results(result)

    if result is None or result.results is None:
        print(f"   ❌ No results returned after polling.")
        if result is not None:
            for attr, val in vars(result).items():
                val_str = repr(val)
                if len(val_str) > 200:
                    val_str = val_str[:200] + "..."
                print(f"      .{attr} = {val_str}")
        return [0.0] * len(times)

    # Compute <Z_0> from counts
    mag_by_time = {}
    for entry in result.results:
        label = entry["label"]
        counts = entry["results"]
        t = time_by_label[label]

        total_shots = sum(counts.values())
        p0, p1 = 0, 0
        for bitstring, count in counts.items():
            # Qiskit: bitstring is reversed — qubit 0 is the rightmost character
            if bitstring[-1] == "0":
                p0 += count
            else:
                p1 += count
        mag_by_time[t] = (p0 - p1) / total_shots

    # Build ordered list (prepend t=0)
    magnetizations = []
    for t in times:
        if t == 0.0:
            magnetizations.append(-1.0 if neel_state[0] == "1" else 1.0)
        else:
            magnetizations.append(mag_by_time.get(t, 0.0))

    return magnetizations


# =====================================================================
#  Phase 1 — EXPORT: Build circuits and save as QASM
# =====================================================================

def phase_export(args):
    """Build one Exact-Trotter circuit per time step and save as QASM."""
    N_QUBITS = args.n_qubits
    N_STEPS = args.n_steps
    ORDER = args.order
    T_MAX = args.t_max
    N_POINTS = args.n_points
    output_dir = args.output_dir

    times = np.linspace(0, T_MAX, N_POINTS)
    neel_state = "10" * (N_QUBITS // 2) + ("1" if N_QUBITS % 2 else "")

    H = build_tfim_hamiltonian(n_qubits=N_QUBITS, J=1.0, h=0.5)

    print(f"   ⚙️  Hamiltonian: {N_QUBITS}-qubit TFIM (J=1.0, h=0.5)")
    print(f"   ⚙️  Néel state:  |{neel_state}⟩")
    print(f"   ⚙️  Time range:  0 → {T_MAX} ({N_POINTS} points, {N_STEPS} Trotter steps, order {ORDER})")

    os.makedirs(output_dir, exist_ok=True)

    # Clean stale .qasm files from previous runs
    for f in os.listdir(output_dir):
        if f.endswith(".qasm"):
            os.remove(os.path.join(output_dir, f))

    # Save time-point metadata for the reload step.
    times_file = os.path.join(output_dir, "times.npy")
    np.save(times_file, times)

    count = 0
    for i, t in enumerate(times):
        if t == 0.0:
            continue

        tape = build_neel_circuit_tape(H, N_QUBITS, neel_state, t, N_STEPS, order=ORDER)

        # tape → MetaCircuit → DAG → QASM body. Both helpers are public; the
        # old single-shot circuit_body_to_qasm was retired in favour of this
        # split (build a DAG once, emit QASM at any precision later).
        meta = qscript_to_meta(tape, precision=10, parameter_order=())
        _, dag = meta.circuit_bodies[0]
        body_qasm = dag_to_qasm_body(dag, precision=10)

        # Append a measurement on every qubit.
        measure_lines = "".join(
            f"measure q[{w}] -> c[{w}];\n" for w in range(N_QUBITS)
        )
        qasm_header = (
            "OPENQASM 2.0;\n"
            'include "qelib1.inc";\n'
            f"qreg q[{N_QUBITS}];\n"
            f"creg c[{N_QUBITS}];\n"
        )
        full_qasm = qasm_header + body_qasm + measure_lines

        filepath = os.path.join(output_dir, f"neel_t{i:03d}.qasm")
        with open(filepath, "w") as f:
            f.write(full_qasm)
        count += 1

    print(f"\n   💾 Saved {count} QASM circuits to {output_dir}/")
    print(f"   💾 Time metadata saved to {times_file}")
    print(f"\n   Next step: python neel_dynamics.py run --n-qubits {N_QUBITS}")


# =====================================================================
#  RUN: Exact baseline, compressed QASM reload, QDrift
# =====================================================================

def phase_run(args):
    """Run exact baseline, reload compressed circuits, run QDrift, and plot."""
    N_QUBITS = args.n_qubits
    N_STEPS = args.n_steps
    ORDER = args.order
    T_MAX = args.t_max
    N_POINTS = args.n_points
    SHOTS = args.shots
    SAMPLING_BUDGET = args.sampling_budget
    original_dir = args.output_dir
    compressed_dir = args.compressed_dir

    neel_state = "10" * (N_QUBITS // 2) + ("1" if N_QUBITS % 2 else "")

    # --- Backends ---
    local_backend = MaestroSimulator(shots=SHOTS, track_depth=True)
    print("Local backend: MaestroSimulator (statevector reference)")
    if args.cloud:
        cloud_backend = QoroService(job_config=JobConfig(shots=SHOTS, force_sampling=True))
        print("Cloud backend: QoroService")
    else:
        cloud_backend = None

    # Load time metadata
    times_file = os.path.join(original_dir, "times.npy")
    if not os.path.exists(times_file):
        print(f"   ❌ Time metadata not found at {times_file}")
        print(f"   Run 'python neel_dynamics.py export --n-qubits {N_QUBITS}' first.")
        return
    times = np.load(times_file)

    H = build_tfim_hamiltonian(n_qubits=N_QUBITS, J=1.0, h=0.5)
    observable = qml.PauliZ(0)
    hw_backend = cloud_backend if cloud_backend else local_backend
    hw_label = "cloud hardware" if cloud_backend else "local (no --cloud)"

    # ── 1. Exact SV baseline (local — ground truth) ───────────────
    print("\n" + "=" * 70)
    print("  1. Exact Trotterization — local SV (ground truth)")
    print("=" * 70)

    mag_sv = []
    for t in times:
        if t == 0.0:
            mag_sv.append(-1.0 if neel_state[0] == "1" else 1.0)
            continue

        te = TimeEvolution(
            hamiltonian=H,
            time=t,
            n_steps=N_STEPS,
            order=ORDER,
            initial_state=CustomPerQubitState(neel_state),
            observable=observable,
            backend=local_backend,
            trotterization_strategy=ExactTrotterization(),
        )
        te.run()
        mag_sv.append(te.results)

    print(f"   ✅ SV baseline complete")
    print(f"   📏 Avg circuit depth: {local_backend.average_depth():.0f} (std: {local_backend.std_depth():.0f})")
    local_backend.clear_depth_history()

    # ── 2. Exact Trotterization — cloud/hardware ──────────────────
    print("\n" + "=" * 70)
    print(f"  2. Exact Trotterization — {hw_label}")
    print("=" * 70)

    mag_exact_hw = []
    for t in times:
        if t == 0.0:
            mag_exact_hw.append(-1.0 if neel_state[0] == "1" else 1.0)
            continue

        te = TimeEvolution(
            hamiltonian=H,
            time=t,
            n_steps=N_STEPS,
            order=ORDER,
            initial_state=CustomPerQubitState(neel_state),
            observable=observable,
            backend=hw_backend,
            trotterization_strategy=ExactTrotterization(),
        )
        te.run()
        mag_exact_hw.append(te.results)

    print(f"   ✅ Exact Trotterization (hardware) complete")
    print(f"   📏 Avg circuit depth: {hw_backend.average_depth():.0f} (std: {hw_backend.std_depth():.0f})")
    hw_backend.clear_depth_history()

    # ── 3. Compressed QASM reload — cloud/hardware ────────────────
    mag_compressed = None
    if compressed_dir and os.path.isdir(compressed_dir):
        print("\n" + "=" * 70)
        print(f"  3. Compressed circuits — {hw_label} ({compressed_dir}/)")
        print("=" * 70)

        mag_compressed = load_and_execute_qasm(
            compressed_dir, times, hw_backend, neel_state, N_QUBITS
        )
        print(f"   ✅ Compressed circuits complete")
        print(f"   📏 Avg circuit depth: {hw_backend.average_depth():.0f} (std: {hw_backend.std_depth():.0f})")
        hw_backend.clear_depth_history()
    else:
        print(f"\n   ⏭️  No compressed circuit directory provided, skipping.")
        print(f"      Use --compressed-dir <path> to include compressed results.")

    # ── 4. QDrift — cloud/hardware ────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  4. QDrift — {hw_label} (sampling_budget={SAMPLING_BUDGET})")
    print("=" * 70)

    mag_qdrift = []
    for t in times:
        if t == 0.0:
            mag_qdrift.append(-1.0 if neel_state[0] == "1" else 1.0)
            continue

        te = TimeEvolution(
            hamiltonian=H,
            time=t,
            n_steps=N_STEPS,
            order=ORDER,
            initial_state=CustomPerQubitState(neel_state),
            observable=observable,
            backend=hw_backend,
            trotterization_strategy=QDrift(sampling_budget=SAMPLING_BUDGET, keep_fraction=0.75, sampling_strategy="weighted"),
        )
        te.run()
        mag_qdrift.append(te.results)

    print(f"   ✅ QDrift complete")
    print(f"   📏 Avg circuit depth: {hw_backend.average_depth():.0f} (std: {hw_backend.std_depth():.0f})")
    hw_backend.clear_depth_history()

    # ── Plot ──────────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))

    plt.plot(
        times, mag_sv, "o-", color="#6366f1",
        label="Exact SV (ground truth)", linewidth=2.5, markersize=8,
    )

    plt.plot(
        times, mag_exact_hw, "D-", color="#3b82f6",
        label=f"Exact Trotter ({hw_label})", linewidth=2, markersize=7,
    )

    if mag_compressed is not None:
        plt.plot(
            times, mag_compressed, "^-", color="#10b981",
            label=f"Compressed ({hw_label})", linewidth=2, markersize=8,
        )

    plt.plot(
        times, mag_qdrift, "s--", color="#f97316",
        label=f"QDrift ({hw_label}, budget={SAMPLING_BUDGET})", linewidth=2, markersize=7,
    )

    plt.axhline(0, color="black", linestyle="-", alpha=0.2)
    plt.title(
        f"Néel State |{neel_state}⟩ Dynamics (J=1.0, h=0.5)",
        fontsize=14, fontweight="bold",
    )
    plt.xlabel("Time (t)", fontsize=12)
    plt.ylabel(r"Magnetization $\langle Z_0 \rangle$", fontsize=12)
    plt.ylim(-1.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig("dynamics_neel.png", dpi=300)
    print(f"\n   📈 Plot saved to dynamics_neel.png")

    print("\n" + "=" * 70)
    print("  ✅ Done!")
    print("=" * 70)


# =====================================================================
#  CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Néel State Dynamics — QASM Export / Reload Demo"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared arguments
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--n-qubits", type=int, default=15, help="Number of qubits (default: 15)")
    shared.add_argument("--n-steps", type=int, default=5, help="Trotter steps (default: 5)")
    shared.add_argument("--order", type=int, default=2, help="Suzuki-Trotter order (default: 2)")
    shared.add_argument("--t-max", type=float, default=3.0, help="Max evolution time (default: 3.0)")
    shared.add_argument("--n-points", type=int, default=15, help="Number of time points (default: 15)")
    shared.add_argument("--output-dir", type=str, default="qasm_circuits", help="QASM output directory (default: qasm_circuits)")

    # Export subcommand
    export_parser = subparsers.add_parser("export", parents=[shared], help="Build circuits and save as QASM")
    export_parser.set_defaults(func=phase_export)

    # Run subcommand
    run_parser = subparsers.add_parser("run", parents=[shared], help="Run exact, reload compressed QASM, run QDrift, and compare")
    run_parser.add_argument("--shots", type=int, default=10_000, help="Measurement shots (default: 10000)")
    run_parser.add_argument("--sampling-budget", type=int, default=50, help="QDrift sampling budget (default: 50)")
    run_parser.add_argument("--compressed-dir", type=str, default=None, help="Directory with compressed QASM circuits (optional)")
    run_parser.add_argument("--cloud", action="store_true", help="Run hardware comparisons on QoroService (otherwise local-only)")
    run_parser.set_defaults(func=phase_run)

    args = parser.parse_args()
    args.func(args)
