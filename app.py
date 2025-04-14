import streamlit as st
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Quantum Playground", layout="wide")
st.title("🔮 Quantum Playground")

st.sidebar.title("🛠️ Build Your Quantum Circuit")
num_qubits = st.sidebar.slider("Number of Qubits", 1, 3, 1)

qc = QuantumCircuit(num_qubits)

st.sidebar.subheader("Apply Gates")
for i in range(num_qubits):
    if st.sidebar.checkbox(f"Hadamard on qubit {i}"):
        qc.h(i)
    if st.sidebar.checkbox(f"Pauli-X on qubit {i}"):
        qc.x(i)
    if st.sidebar.checkbox(f"Pauli-Z on qubit {i}"):
        qc.z(i)

st.sidebar.subheader("Measurement")
measure = st.sidebar.button("Measure Qubits")

if measure:
    qc.measure_all()

st.subheader("Quantum Circuit")
st.text(qc.draw())

st.subheader("Bloch Sphere Visualization")
backend = Aer.get_backend('statevector_simulator')
job = execute(qc, backend)
result = job.result()
statevector = result.get_statevector()

fig = plot_bloch_multivector(statevector)
st.pyplot(fig)

if measure:
    st.subheader("Measurement Results")
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1024)
    result = job.result()
    counts = result.get_counts()
    fig2 = plot_histogram(counts)
    st.pyplot(fig2)
