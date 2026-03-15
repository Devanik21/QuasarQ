# Quasarq

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/QuasarQ?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/QuasarQ?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> A quantum-classical hybrid reasoning engine — exploring quantum-inspired algorithms for optimisation, search, and machine learning.

---

**Topics:** `astronomical-spectroscopy` · `astronomy` · `astrophysics-ml` · `data-science` · `deep-learning` · `dimensionality-reduction` · `machine-learning` · `neural-networks` · `signal-processing` · `quasar-spectrum-analysis`

## Overview

QuasarQ is a research and educational project exploring quantum-inspired and quantum-classical hybrid
algorithms for optimisation, combinatorial search, and machine learning acceleration. It implements
a collection of quantum-inspired classical algorithms — Simulated Quantum Annealing, Quantum-Inspired
Evolutionary Algorithms (QIEA), and variational quantum circuit simulations — alongside interfaces
to real quantum hardware via IBM Qiskit and Google Cirq.

The project is structured around three progressively advanced tracks. The first track covers
quantum computing fundamentals: qubit representation, gate operations (H, CNOT, Toffoli, phase gates),
quantum circuits, and measurement — all implemented in pure NumPy for pedagogical transparency without
hiding the linear algebra behind a simulator abstraction. The second track implements quantum-inspired
classical algorithms that exhibit superlinear convergence on specific optimisation problems without
requiring quantum hardware. The third track demonstrates Variational Quantum Eigensolver (VQE) and
Quantum Approximate Optimisation Algorithm (QAOA) on real IBM quantum processors via Qiskit.

QuasarQ is explicitly designed to be a learning environment: every implementation is documented with
the mathematical formalism alongside the code, connecting the abstract quantum mechanics to the
practical computational structures.

---

## Motivation

Quantum computing is transitioning from theoretical curiosity to practical engineering tool. Understanding
it requires simultaneously grasping quantum mechanics, linear algebra, and algorithm design — a combination
that most educational resources address only superficially. QuasarQ was built to provide depth at all
three levels simultaneously, with implementations that are transparent enough to learn from and realistic
enough to apply.

---

## Architecture

```
QuasarQ Architecture
        │
  ┌─────────────────────────────────────────────┐
  │  Track 1: Quantum Fundamentals (NumPy)      │
  │  Qubits, gates, circuits, measurement       │
  └─────────────────────────────────────────────┘
        │
  ┌─────────────────────────────────────────────┐
  │  Track 2: Quantum-Inspired Classical        │
  │  Simulated Quantum Annealing (SQA)          │
  │  Quantum-Inspired Evolutionary Algo (QIEA)  │
  │  Quantum Walk-based search                  │
  └─────────────────────────────────────────────┘
        │
  ┌─────────────────────────────────────────────┐
  │  Track 3: Real Quantum Hardware             │
  │  Qiskit: VQE, QAOA on IBM backends          │
  │  Cirq: variational circuits (Google)        │
  └─────────────────────────────────────────────┘
```

---

## Features

### Pure NumPy Quantum Simulator
Custom statevector quantum circuit simulator implemented in NumPy — no Qiskit, no abstraction layers — exposing the raw tensor product structure and unitary gate matrices for maximum pedagogical transparency.

### Quantum Gate Library
Implementation of all standard single- and multi-qubit gates: Pauli X/Y/Z, Hadamard, phase S/T, CNOT, CZ, Toffoli, SWAP, and parameterised rotation gates (Rx, Ry, Rz).

### Simulated Quantum Annealing
Path-Integral Monte Carlo-based SQA implementation for combinatorial optimisation problems (TSP, graph colouring, max-cut) — demonstrating quantum tunnelling through energy barriers.

### Quantum-Inspired Evolutionary Algorithm
QIEA implementation using quantum rotation gates on probability amplitude representations of chromosomes, achieving faster convergence than classical GAs on binary optimisation benchmarks.

### VQE Implementation
Variational Quantum Eigensolver for ground state energy estimation of small molecular Hamiltonians (H₂, LiH), using parameterised ansatz circuits and classical gradient-free optimisation (COBYLA, SPSA).

### QAOA for Combinatorial Problems
Quantum Approximate Optimisation Algorithm for Max-Cut and graph partitioning, with configurable circuit depth p and classical optimisation of variational angles (γ, β).

### Quantum Circuit Visualisation
ASCII and Matplotlib-based circuit diagram rendering, qubit state Bloch sphere visualisation, and probability distribution bar charts after measurement.

### IBM Quantum Integration
Qiskit backend connection to IBM Quantum cloud processors (ibm_brisbane, ibm_kyoto) via IBMQ account credentials for running real quantum experiments.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **NumPy** | Core quantum simulation | Statevector simulation, unitary matrix operations, tensor products |
| **Qiskit** | IBM quantum hardware | VQE, QAOA, circuit transpilation for real hardware |
| **SciPy** | Classical optimisation | COBYLA, SPSA for variational parameter optimisation |
| **Matplotlib** | Visualisation | Circuit diagrams, Bloch spheres, probability histograms |
| **Cirq (optional)** | Google quantum | Variational circuits on Google Quantum AI processors |
| **pandas** | Benchmark results | Algorithm comparison tables |

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JavaScript projects)
- A virtual environment manager (`venv`, `conda`, or equivalent)
- API keys as listed in the Configuration section

### Installation

```bash
git clone https://github.com/Devanik21/QuasarQ.git
cd QuasarQ
python -m venv venv && source venv/bin/activate
pip install numpy scipy matplotlib pandas jupyter
pip install qiskit qiskit-ibm-runtime  # for IBM hardware access
# Optional: Google Cirq
# pip install cirq

jupyter notebook
```

---

## Usage

```bash
# Run the quantum fundamentals notebook
jupyter notebook notebooks/01_quantum_fundamentals.ipynb

# Run SQA on Max-Cut problem
python sqa_maxcut.py --nodes 20 --edges 40 --tunneling_field 1.0

# Run VQE for H2 ground state
python vqe_h2.py --backend statevector_simulator --shots 1024

# Connect to IBM Quantum
python ibm_qaoa.py --backend ibm_brisbane --depth 3
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `IBM_QUANTUM_TOKEN` | `(optional)` | IBM Quantum API token for real hardware access |
| `DEFAULT_BACKEND` | `statevector_simulator` | Simulation backend: statevector, qasm, ibm_brisbane |
| `NUM_QUBITS` | `8` | Default number of qubits for demo circuits |
| `SQA_TROTTER_STEPS` | `20` | Suzuki-Trotter decomposition steps for SQA |

> Copy `.env.example` to `.env` and populate required values before running.

---

## Project Structure

```
QuasarQ/
├── README.md
├── requirements.txt
├── app.py
└── ...
```

---

## Roadmap

- [ ] Quantum machine learning: Quantum Kernel SVM and quantum neural network implementations
- [ ] Quantum error correction: Shor code and surface code simulation
- [ ] Quantum cryptography: BB84 protocol and quantum key distribution simulation
- [ ] Noisy circuit simulation with depolarising and amplitude damping channels
- [ ] Interactive web tutorial built with Streamlit for browser-based quantum learning

---

## Contributing

Contributions, issues, and suggestions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-idea`
3. Commit your changes: `git commit -m 'feat: add your idea'`
4. Push to your branch: `git push origin feature/your-idea`
5. Open a Pull Request with a clear description

Please follow conventional commit messages and add documentation for new features.

---

## Notes

Real quantum hardware access requires an IBM Quantum account (free tier available at quantum.ibm.com). All simulations run locally without any account. Quantum computation results on real hardware may differ from ideal simulation due to noise and decoherence.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with curiosity, depth, and care — because good projects deserve good documentation.*
