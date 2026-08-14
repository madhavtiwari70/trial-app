# =============================================================================
#  Economic Load Dispatch with Prohibited Operating Zones via PCE-VQE
# =============================================================================
#
#  Solves the Economic Load Dispatch problem for a 3-generator microgrid
#  using the Divi quantum SDK.
#
#  Scroll to the bottom to see the main flow — it reads like plain English!
#
# =============================================================================

import dimod
import pennylane as qml
from qiskit.circuit.library import CXGate, RYGate, RZGate
import time

from divi.backends import MaestroSimulator, QoroService, JobConfig
from divi.qprog import PCE, GenericLayerAnsatz
from divi.qprog.problems import BinaryOptimizationProblem
from divi.qprog.optimizers import PymooMethod, PymooOptimizer
from divi.hamiltonians import qubo_to_matrix


# ─────────────────────────────────────────────────────────────────────
#  STEP 1 — Define the generators
# ─────────────────────────────────────────────────────────────────────

def define_generators():
    """Define the 3-generator microgrid.

    Each generator has:
      - a, b, c:          fuel cost curve  Cost = a + b·P + c·P²
      - P_min, P_max:     operating range in MW
      - poz_low, poz_high: prohibited operating zone (vibration band)

    Returns a list of generator dicts.
    """
    return [
        {
            "name": "Gen 1", "a": 20, "b": 2.0, "c": 0.010,
            "P_min": 40, "P_max": 115,
            "poz_low": 60, "poz_high": 75,
        },
        {
            "name": "Gen 2", "a": 15, "b": 1.5, "c": 0.020,
            "P_min": 20, "P_max": 95,
            "poz_low": 40, "poz_high": 55,
        },
        {
            "name": "Gen 3", "a": 25, "b": 1.8, "c": 0.015,
            "P_min": 30, "P_max": 105,
            "poz_low": 50, "poz_high": 65,
        },
    ]



def define_generators_large():
    """Define a six-generator variant for exploring the same formulation."""
    return define_generators() + [
        {
            "name": "Gen 4", "a": 18, "b": 1.7, "c": 0.012,
            "P_min": 35, "P_max": 110,
            "poz_low": 55, "poz_high": 70,
        },
        {
            "name": "Gen 5", "a": 22, "b": 2.2, "c": 0.009,
            "P_min": 50, "P_max": 120,
            "poz_low": 65, "poz_high": 80,
        },
        {
            "name": "Gen 6", "a": 12, "b": 1.3, "c": 0.018,
            "P_min": 25, "P_max": 90,
            "poz_low": 45, "poz_high": 60,
        },
    ]


# ─────────────────────────────────────────────────────────────────────
#  STEP 2 — Build the optimisation problem (QUBO)
# ─────────────────────────────────────────────────────────────────────

STEP_MW = 5           # power resolution per qubit level
N_QUBITS_PER_GEN = 4  # 2^4 = 16 discrete levels per generator
BIT_WEIGHTS = [2**b for b in range(N_QUBITS_PER_GEN)]  # [1, 2, 4, 8]


def fuel_cost(gen, power):
    """Compute fuel cost ($) for a generator at a given power (MW)."""
    return gen["a"] + gen["b"] * power + gen["c"] * power ** 2


def _qubit_name(gen_idx, bit_idx):
    """Variable name for qubit `bit_idx` of generator `gen_idx`."""
    return f"q_{gen_idx}_{bit_idx}"


def decode_power(gen_idx, generators, bit_values):
    """Convert binary qubit values back to MW for one generator."""
    integer_val = sum(
        BIT_WEIGHTS[b] * bit_values[_qubit_name(gen_idx, b)]
        for b in range(N_QUBITS_PER_GEN)
    )
    return generators[gen_idx]["P_min"] + STEP_MW * integer_val


def build_qubo(generators, demand, penalty_lambda=2000, poz_mu=5000):
    """Build the Binary Quadratic Model (BQM) for the ELD problem.

    The BQM encodes three things as a single energy function:
      1. Fuel cost    — the objective we want to minimise
      2. Demand penalty — forces total generation to equal demand
      3. POZ penalty  — forbids generators from operating in vibration bands

    Args:
        generators:     list of generator dicts from define_generators()
        demand:         target load in MW
        penalty_lambda: weight for the demand constraint
        poz_mu:         weight for prohibited operating zones

    Returns:
        bqm:         the dimod BinaryQuadraticModel
        var_names:   ordered list of qubit variable names
    """
    bqm = dimod.BinaryQuadraticModel(vartype="BINARY")
    var_names = []

    for g in range(len(generators)):
        for b in range(N_QUBITS_PER_GEN):
            var_names.append(_qubit_name(g, b))

    # ── 1. Fuel cost objective ──
    cost_offset = 0.0
    for g, gen in enumerate(generators):
        a, b_coeff, c = gen["a"], gen["b"], gen["c"]
        p_min = gen["P_min"]
        cost_offset += a + b_coeff * p_min + c * p_min ** 2

        for k in range(N_QUBITS_PER_GEN):
            w_k = STEP_MW * BIT_WEIGHTS[k]
            var_k = _qubit_name(g, k)
            bqm.add_linear(var_k, b_coeff * w_k + c * (2 * p_min * w_k + w_k ** 2))
            for l in range(k + 1, N_QUBITS_PER_GEN):
                w_l = STEP_MW * BIT_WEIGHTS[l]
                bqm.add_quadratic(var_k, _qubit_name(g, l), c * 2 * w_k * w_l)
    bqm.offset += cost_offset

    # ── 2. Demand constraint:  λ · (P1 + P2 + P3 − demand)² ──
    d_const = sum(gen["P_min"] for gen in generators) - demand
    bqm.offset += penalty_lambda * d_const ** 2

    d_terms = []
    for g in range(len(generators)):
        for k in range(N_QUBITS_PER_GEN):
            w = STEP_MW * BIT_WEIGHTS[k]
            d_terms.append((_qubit_name(g, k), w))

    for var_i, w_i in d_terms:
        bqm.add_linear(var_i, penalty_lambda * (2 * d_const * w_i + w_i ** 2))
    for i in range(len(d_terms)):
        vi, wi = d_terms[i]
        for j in range(i + 1, len(d_terms)):
            vj, wj = d_terms[j]
            bqm.add_quadratic(vi, vj, penalty_lambda * 2 * wi * wj)

    # ── 3. POZ penalty:  μ · (1 − q_msb) · q_2nd  per generator ──
    for g in range(len(generators)):
        q_msb = _qubit_name(g, 3)
        q_2nd = _qubit_name(g, 2)
        bqm.add_linear(q_2nd, poz_mu)
        bqm.add_quadratic(q_msb, q_2nd, -poz_mu)

    return bqm, var_names


# ─────────────────────────────────────────────────────────────────────
#  STEP 3 — Find the best classical solution (brute force)
# ─────────────────────────────────────────────────────────────────────

def classical_brute_force(generators, demand, bqm):
    """Enumerate all 4,096 configurations and return the cheapest valid one.

    Returns:
        (P1, P2, P3, cost) for the best valid dispatch.
    """
    best = None

    for i1 in range(16):
        p1 = generators[0]["P_min"] + STEP_MW * i1
        poz1 = generators[0]["poz_low"] <= p1 <= generators[0]["poz_high"]
        for i2 in range(16):
            p2 = generators[1]["P_min"] + STEP_MW * i2
            poz2 = generators[1]["poz_low"] <= p2 <= generators[1]["poz_high"]
            for i3 in range(16):
                p3 = generators[2]["P_min"] + STEP_MW * i3
                poz3 = generators[2]["poz_low"] <= p3 <= generators[2]["poz_high"]

                if p1 + p2 + p3 != demand:
                    continue
                if poz1 or poz2 or poz3:
                    continue

                cost = sum(fuel_cost(generators[g], p)
                           for g, p in enumerate([p1, p2, p3]))
                if best is None or cost < best[3]:
                    best = (p1, p2, p3, cost)

    return best


def classical_sa_solve(generators, demand, bqm, num_reads=1000):
    """Classical simulated annealing baseline — works for any number of generators."""
    sampler = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=num_reads)

    best = None
    for sample, energy in sampleset.data(["sample", "energy"]):
        powers = []
        valid = True
        for g, gen in enumerate(generators):
            p = gen["P_min"] + STEP_MW * sum(
                BIT_WEIGHTS[b] * int(sample.get(_qubit_name(g, b), 0))
                for b in range(N_QUBITS_PER_GEN)
            )
            if gen["poz_low"] <= p <= gen["poz_high"]:
                valid = False
                break
            powers.append(p)

        if not valid or abs(sum(powers) - demand) > STEP_MW:
            continue

        cost = sum(fuel_cost(gen, p) for gen, p in zip(generators, powers))
        if best is None or cost < best[-1]:
            best = tuple(powers) + (cost,)

    return best


# ─────────────────────────────────────────────────────────────────────
#  STEP 4 — Solve with quantum computing (PCE-VQE)
# ─────────────────────────────────────────────────────────────────────

def solve_with_pce(bqm, n_layers=3, max_iterations=20, alpha=3.0,
                   population_size=30, shots=10000, backend=None):
    """Run the PCE-VQE quantum solver on the BQM.

    PCE (Pauli Correlation Encoding) compresses 12 QUBO variables into
    just 5 qubits using polynomial encoding — far fewer than the 12
    qubits QAOA would need.

    Args:
        bqm:              the Binary Quadratic Model to solve
        n_layers:         depth of the variational quantum circuit
        max_iterations:   number of Differential Evolution generations
        alpha:            binary activation hardness (higher = sharper)
        population_size:  DE population per generation
        shots:            measurement samples per circuit evaluation
        backend:          Divi backend (MaestroSimulator or QoroService)

    Returns:
        pce_solver:  the solved PCE object (access .solution, .get_top_solutions)
    """
    if backend is None:
        backend = MaestroSimulator(shots=shots)

    qubo_mat = qubo_to_matrix(bqm)

    ansatz = GenericLayerAnsatz(
        gate_sequence=[RYGate, RZGate],
        entangler=CXGate,
        entangling_layout="all-to-all",
    )

    pce_solver = PCE(
        BinaryOptimizationProblem(qubo_mat),
        ansatz=ansatz,
        n_layers=n_layers,
        encoding_type="poly",
        optimizer=PymooOptimizer(method=PymooMethod.DE,
                                 population_size=population_size),
        max_iterations=max_iterations,
        alpha=alpha,
        backend=backend,
    )

    print(f"\n   PCE qubits: {pce_solver.n_qubits}  "
          f"(poly encoding of {len(bqm.variables)} variables)")
    pce_solver.run()
    return pce_solver


# ─────────────────────────────────────────────────────────────────────
#  STEP 5 — Repair quantum solutions to make them feasible
# ─────────────────────────────────────────────────────────────────────

def repair_solution(powers, generators, demand):
    """Fix a near-feasible quantum solution so it meets all constraints.

    Stage 1 — Snap any generator inside a Prohibited Operating Zone
              to the nearest allowed power level.
    Stage 2 — Greedily adjust generators (cheapest first) until total
              generation exactly matches demand.

    Args:
        powers:     list of MW values [P1, P2, P3]
        generators: list of generator dicts
        demand:     target load in MW

    Returns:
        Repaired [P1, P2, P3] list, or None if repair is impossible.
    """
    ps = list(powers)

    # Stage 1: fix POZ violations
    for g, gen in enumerate(generators):
        if gen["poz_low"] <= ps[g] <= gen["poz_high"]:
            # Find all allowed power levels for this generator
            allowed = [
                gen["P_min"] + STEP_MW * idx
                for idx in range(2 ** N_QUBITS_PER_GEN)
                if not (gen["poz_low"]
                        <= gen["P_min"] + STEP_MW * idx
                        <= gen["poz_high"])
            ]
            ps[g] = min(allowed, key=lambda lv: abs(lv - ps[g]))

    # Stage 2: fix demand mismatch
    for _ in range(50):
        gap = demand - sum(ps)
        if gap == 0:
            break

        step = STEP_MW if gap > 0 else -STEP_MW
        best_g, best_cost = None, float("inf")

        for g, gen in enumerate(generators):
            new_p = ps[g] + step
            if new_p < gen["P_min"] or new_p > gen["P_max"]:
                continue
            if gen["poz_low"] <= new_p <= gen["poz_high"]:
                continue
            marginal = abs(fuel_cost(gen, new_p) - fuel_cost(gen, ps[g]))
            if marginal < best_cost:
                best_cost = marginal
                best_g = g

        if best_g is None:
            return None
        ps[best_g] += step

    return ps if sum(ps) == demand else None


def dispatch_feasibility_report(powers, generators, demand):
    """Return constraint diagnostics for a decoded generator dispatch."""
    powers = [int(power) for power in powers]
    prohibited = [
        i
        for i, (power, gen) in enumerate(zip(powers, generators))
        if gen["poz_low"] <= power <= gen["poz_high"]
    ]
    out_of_bounds = [
        i
        for i, (power, gen) in enumerate(zip(powers, generators))
        if power < gen["P_min"] or power > gen["P_max"]
    ]
    total = sum(powers)
    return {
        "total_generation": total,
        "demand_gap": demand - total,
        "prohibited_generators": prohibited,
        "out_of_bounds_generators": out_of_bounds,
        "feasible": total == demand and not prohibited and not out_of_bounds,
    }


def penalty_sweep(generators, demand, penalties=(200, 2_000, 20_000)):
    """Show how the demand-penalty weight changes the QUBO ground sample."""
    rows = []
    for penalty in penalties:
        bqm, _ = build_qubo(generators, demand, penalty_lambda=penalty)
        sample = dimod.ExactSolver().sample(bqm).first.sample
        powers = [decode_power(g, generators, sample) for g in range(len(generators))]
        rows.append((penalty, dispatch_feasibility_report(powers, generators, demand)))
    return rows


def print_dispatch_report(label, powers, generators, demand):
    """Print the demand and generator-constraint status of one dispatch."""
    report = dispatch_feasibility_report(powers, generators, demand)
    print(f"\n   {label}")
    print(
        f"   total={report['total_generation']:.0f} MW, "
        f"demand gap={report['demand_gap']:.0f} MW"
    )
    print(
        f"   prohibited zones: {report['prohibited_generators'] or 'none'}; "
        f"limits: {report['out_of_bounds_generators'] or 'none'}; "
        f"feasible: {'yes' if report['feasible'] else 'no'}"
    )


def find_best_repaired_solution(pce_solver, bqm, generators, demand, top_n=20):
    """Scan the top quantum candidates and return the best repaired dispatch.

    For each candidate in the quantum probability distribution:
      1. Decode the bitstring to MW values
      2. Repair any constraint violations
      3. Keep the cheapest valid result

    Returns:
        (powers, cost, probability) or None
    """
    top_solutions = pce_solver.get_top_solutions(n=top_n, include_decoded=True)
    best = None

    print("\n   Top quantum candidates:")
    for i, sol in enumerate(top_solutions, 1):
        if sol.decoded is None:
            continue

        sample = {var: int(val) for var, val in zip(bqm.variables, sol.decoded)}
        ps = [decode_power(g, generators, sample) for g in range(len(generators))]
        report = dispatch_feasibility_report(ps, generators, demand)
        cost = sum(fuel_cost(generators[g], ps[g]) for g in range(len(generators)))
        tot = sum(ps)
        valid = (
            tot == demand
            and all(
                not (generators[g]["poz_low"] <= ps[g] <= generators[g]["poz_high"])
                for g in range(len(generators))
            )
        )
        print(f"     {i:2d}. P=[{ps[0]:.0f},{ps[1]:.0f},{ps[2]:.0f}]  "
              f"Tot={tot:.0f}  Cost={cost:.0f}$  "
              f"Prob={sol.prob:.2%}  {'✅' if valid else '❌'}")
        if i == 1:
            print(
                f"         demand gap={report['demand_gap']:.0f} MW; "
                f"prohibited zones={report['prohibited_generators'] or 'none'}"
            )

        repaired = repair_solution(ps, generators, demand)
        if repaired is not None:
            rep_cost = sum(fuel_cost(generators[g], repaired[g])
                          for g in range(len(generators)))
            if best is None or rep_cost < best[1]:
                best = (repaired, rep_cost, sol.prob)

    return best


# ─────────────────────────────────────────────────────────────────────
#  STEP 6 — Compare quantum vs classical results
# ─────────────────────────────────────────────────────────────────────

def print_comparison(quantum_result, classical_result):
    """Print a side-by-side comparison of quantum and classical solutions."""
    q_powers, q_cost, q_prob = quantum_result
    c_powers, c_cost = classical_result[:-1], classical_result[-1]

    c_str = ", ".join(f"P{i+1}={p:.0f}" for i, p in enumerate(c_powers))
    q_str = ", ".join(f"P{i+1}={p:.0f}" for i, p in enumerate(q_powers))

    print("\n" + "=" * 70)
    print("  🔬 Quantum vs Classical Comparison")
    print("=" * 70)
    print(f"\n   Classical optimum:  {c_str} MW  → Cost = {c_cost:.1f} $")
    print(f"   PCE-VQE result:     {q_str} MW  → Cost = {q_cost:.1f} $")

    if abs(q_cost - c_cost) < 0.1:
        print("\n   🎉 PCE-VQE found the global optimum!")
    else:
        print(f"\n   ⚡ Gap from optimum: {q_cost - c_cost:.1f} $")
        print("      Try increasing n_layers or max_iterations.")

    print("\n" + "=" * 70)


# =====================================================================
#  MAIN — The high-level flow (start reading here!)
# =====================================================================

if __name__ == "__main__":
    DEMAND = 195  # MW — how much power the grid needs

    # --- Backend selection ---
    USE_QORO_SERVICE = False  # Optional backend; credentials must be configured.

    if USE_QORO_SERVICE:
        backend = QoroService(job_config=JobConfig(shots=10_000))
        print("Using QoroService backend")
    else:
        backend = MaestroSimulator(shots=10_000)
        print("Using local MaestroSimulator")

    # 1. Define the generators and their constraints
    generators = define_generators()

    # 2. Encode the problem as a QUBO (quantum-ready format)
    bqm, var_names = build_qubo(generators, demand=DEMAND)
    print(f"Built QUBO: {len(var_names)} variables, {len(bqm.quadratic)} interactions")
    print("Demand-penalty sweep (QUBO ground samples):")
    for penalty, report in penalty_sweep(generators, DEMAND):
        print(
            f"  λ={penalty:>5}: gap={report['demand_gap']:>4.0f} MW, "
            f"prohibited={report['prohibited_generators'] or 'none'}"
        )

    # 3. Find the classical optimum (for comparison)
    classical_best = classical_brute_force(generators, DEMAND, bqm)
    print(f"Classical optimum: P1={classical_best[0]}, P2={classical_best[1]}, "
          f"P3={classical_best[2]} MW  → Cost = {classical_best[3]:.1f} $")

    # 4. Solve with quantum computing
    print("\n🚀 Running quantum solver (PCE-VQE)...")
    t0 = time.time()
    pce_solver = solve_with_pce(bqm, backend=backend)
    local_time = time.time() - t0

    # 5. Repair the quantum solution to make it fully valid
    result = find_best_repaired_solution(pce_solver, bqm, generators, DEMAND)

    if result is not None:
        powers, cost, prob = result
        print(f"\n   → Best repaired solution: P1={powers[0]:.0f}, "
              f"P2={powers[1]:.0f}, P3={powers[2]:.0f} MW, "
              f"Cost={cost:.1f} $  (quantum seed prob={prob:.2%})")
        print_dispatch_report("Repaired-solution feasibility", powers, generators, DEMAND)

        # 6. Compare quantum vs classical
        print_comparison(result, classical_best)
    else:
        print("\n   ⚠️  No valid solution found. Try increasing max_iterations.")

    print("\n" + "=" * 70)
    print("  Six-generator variant (24 binary variables, about 8 PCE qubits)")
    print("=" * 70)

    generators_large = define_generators_large()
    DEMAND_LARGE = 390  # MW — scales with added capacity

    bqm_large, var_names_large = build_qubo(generators_large, demand=DEMAND_LARGE)
    print(f"Built QUBO: {len(var_names_large)} variables "
          f"(was {len(var_names)}) — {len(bqm_large.quadratic)} interactions")

    classical_large = classical_sa_solve(generators_large, DEMAND_LARGE, bqm_large, num_reads=100)
    if classical_large:
        print(f"Classical SA baseline: Cost = {classical_large[-1]:.1f} $")

    print("\nRunning the six-generator PCE-VQE variant...")

    t0 = time.time()
    pce_solver_large = solve_with_pce(
        bqm_large,
        n_layers=3,
        max_iterations=10,
        population_size=50,
        backend=backend,
    )
    large_time = time.time() - t0

    result_large = find_best_repaired_solution(
        pce_solver_large, bqm_large, generators_large, DEMAND_LARGE
    )

    if result_large is not None:
        powers, cost, prob = result_large
        gen_str = ", ".join(f"P{i+1}={p:.0f}" for i, p in enumerate(powers))
        print(f"\n   → Best repaired solution: {gen_str} MW")
        print(f"     Total={sum(powers):.0f} MW, Cost={cost:.1f} $, "
              f"quantum seed prob={prob:.2%}")
        if classical_large:
            print_comparison(result_large, classical_large)

    print(f"\n   Three-generator run: {local_time:.1f}s ({len(var_names)} variables)")
    print(f"   Six-generator run: {large_time:.1f}s ({len(var_names_large)} variables)")
