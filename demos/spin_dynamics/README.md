# Spin Dynamics of the Transverse-Field Ising Model

These scripts simulate a one-dimensional transverse-field Ising chain,

$$ H = -J \sum_i Z_i Z_{i+1} - h \sum_i X_i. $$

`spin_dynamics.py` compares exact Trotterization and QDrift across several time points using `TimeEvolutionTrajectory`. `neel_dynamics.py` additionally demonstrates QASM export, optional external circuit processing, and reloading the circuits for execution.

## Run locally

```bash
python spin_dynamics.py
python neel_dynamics.py export --n-qubits 15
python neel_dynamics.py run --n-qubits 15
```

`--cloud` is optional for the hardware-targeting comparison and requires configured QoroService credentials. Local statevector and simulator results remain available without it.
