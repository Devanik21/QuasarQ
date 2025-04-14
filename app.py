import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import google.generativeai as genai
from scipy.integrate import solve_ivp

# Page configuration
st.set_page_config(
    page_title="Quantum Day Interactive Explorer",
    page_icon="⚛️",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'wavefunction' not in st.session_state:
    st.session_state.wavefunction = None
if 'potential_type' not in st.session_state:
    st.session_state.potential_type = "Harmonic Oscillator"
if 'animation' not in st.session_state:
    st.session_state.animation = None

# Sidebar configuration
with st.sidebar:
    st.title("⚛️ Quantum Explorer")
    st.subheader("Happy Quantum Day! 🎉")
    
    # Gemini API Key input
    api_key = st.text_input("Enter your Gemini API Key", type="password")
    
    # Only configure the API if a key is provided
    if api_key:
        try:
            genai.configure(api_key=api_key)
            st.success("API key configured successfully!")
        except Exception as e:
            st.error(f"Error configuring API: {e}")
    
    # Quantum simulation parameters
    st.subheader("Simulation Parameters")
    potential_type = st.selectbox(
        "Potential Type",
        ["Harmonic Oscillator", "Infinite Square Well", "Quantum Tunneling"]
    )
    
    n_state = st.slider("Quantum State (n)", 1, 5, 1)
    
    # Animation speed
    animation_speed = st.slider("Animation Speed", 0.1, 2.0, 1.0)
    
    # Button to run simulation
    if st.button("Run Simulation"):
        st.session_state.potential_type = potential_type
        st.session_state.n_state = n_state
        st.session_state.animation_speed = animation_speed

# Main content
st.title("Quantum Mechanics Interactive Explorer")
st.markdown("""
Explore fundamental quantum mechanical concepts through interactive simulations and AI-powered explanations.
""")

# Initialize tabs
tab1, tab2, tab3 = st.tabs(["Wavefunction Simulator", "Quantum Concepts", "Ask Gemini"])

with tab1:
    st.header("Wavefunction Visualization")
    
    # Helper functions for quantum simulations
    def harmonic_oscillator_wavefunction(x, n):
        """Generate harmonic oscillator wavefunction for state n"""
        # Constants
        m = 1.0  # mass
        omega = 1.0  # angular frequency
        hbar = 1.0  # reduced Planck's constant
        
        # Normalization and Hermite polynomial calculation
        def hermite(x, n):
            if n == 0:
                return np.ones_like(x)
            elif n == 1:
                return 2 * x
            else:
                return 2 * x * hermite(x, n-1) - 2 * (n-1) * hermite(x, n-2)
        
        # Characteristic length for harmonic oscillator
        alpha = np.sqrt(m * omega / hbar)
        
        # Normalization constant
        N = (alpha / np.sqrt(np.pi * 2**n * math.factorial(n)))**0.5
        
        # Wavefunction
        psi = N * hermite(alpha * x, n) * np.exp(-(alpha * x)**2 / 2)
        
        return psi
    
    def infinite_square_well(x, n, L=10):
        """Generate infinite square well wavefunction for state n"""
        psi = np.sqrt(2/L) * np.sin(n * np.pi * x / L)
        # Set values outside the well to zero
        psi[(x < 0) | (x > L)] = 0
        return psi
    
    def quantum_tunneling(x, E=0.5, V0=1.0, a=5.0):
        """Generate wavefunction for quantum tunneling through a barrier"""
        # Simple approximation for visualization purposes
        k1 = np.sqrt(2 * E)  # wave number in regions I and III
        k2 = np.sqrt(2 * (V0 - E)) if E < V0 else np.sqrt(2 * (E - V0)) * 1j  # wave number/decay constant in region II
        
        psi = np.zeros_like(x, dtype=complex)
        
        # Region I: x < 0
        region1 = x < 0
        psi[region1] = np.exp(1j * k1 * x[region1]) + 0.5 * np.exp(-1j * k1 * x[region1])
        
        # Region II: 0 <= x <= a (barrier)
        region2 = (x >= 0) & (x <= a)
        if E < V0:  # tunneling case
            psi[region2] = 0.3 * np.exp(-k2 * x[region2]) + 0.3 * np.exp(k2 * x[region2])
        else:  # over-barrier case
            psi[region2] = 0.3 * np.exp(1j * k2 * x[region2]) + 0.3 * np.exp(-1j * k2 * x[region2])
        
        # Region III: x > a
        region3 = x > a
        psi[region3] = 0.25 * np.exp(1j * k1 * x[region3])
        
        return psi
    
    # Generate wavefunction based on selected potential
    x = np.linspace(-10, 10, 1000)
    
    if st.session_state.potential_type == "Harmonic Oscillator":
        n = st.session_state.get('n_state', 1)
        psi = harmonic_oscillator_wavefunction(x, n)
        potential = 0.5 * x**2  # V(x) = 0.5*m*ω²*x²
        title = f"Harmonic Oscillator - State n={n}"
        y_range = [-0.8, 0.8]
        
    elif st.session_state.potential_type == "Infinite Square Well":
        L = 10  # Well width
        x = np.linspace(-2, L+2, 1000)
        n = st.session_state.get('n_state', 1)
        psi = infinite_square_well(x, n, L)
        # Create potential function (infinity outside well)
        potential = np.ones_like(x) * 5
        potential[(x >= 0) & (x <= L)] = 0
        title = f"Infinite Square Well - State n={n}"
        y_range = [-0.6, 0.6]
        
    elif st.session_state.potential_type == "Quantum Tunneling":
        x = np.linspace(-10, 15, 1000)
        psi = quantum_tunneling(x)
        # Create potential barrier
        potential = np.zeros_like(x)
        potential[(x >= 0) & (x <= 5)] = 1.0
        title = "Quantum Tunneling Through a Barrier"
        y_range = [-1.2, 1.2]
    
    # Convert complex wavefunction to real for visualization
    if np.iscomplexobj(psi):
        psi_real = psi.real
        psi_imag = psi.imag
        psi_prob = np.abs(psi)**2
    else:
        psi_real = psi
        psi_imag = np.zeros_like(psi)
        psi_prob = psi**2
    
    # Create figure for visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot potential
    ax.plot(x, potential, 'k--', label='Potential V(x)')
    
    # Plot wavefunction
    line_real, = ax.plot(x, psi_real, 'b-', label='Re(ψ)')
    
    if np.iscomplexobj(psi):
        line_imag, = ax.plot(x, psi_imag, 'r-', label='Im(ψ)')
    
    line_prob, = ax.plot(x, psi_prob, 'g-', label='|ψ|²')
    
    ax.set_ylim(y_range)
    ax.set_xlim(min(x), max(x))
    ax.set_title(title)
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('ψ(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Function to animate wavefunction (time evolution)
    def update(frame, line_real, line_imag=None, psi=None):
        # Simple time evolution (phase rotation for visualization)
        omega = st.session_state.get('animation_speed', 1.0)
        evolved_psi = psi * np.exp(-1j * omega * frame * 0.1)
        
        line_real.set_ydata(evolved_psi.real)
        
        if line_imag is not None and np.iscomplexobj(psi):
            line_imag.set_ydata(evolved_psi.imag)
            
        return line_real, line_imag
    
    # Create animation if not already created
    if np.iscomplexobj(psi):
        anim = FuncAnimation(
            fig, update, frames=100, interval=50, blit=True,
            fargs=(line_real, line_imag, psi)
        )
    else:
        # For real wavefunctions, we still animate but only update the real part
        anim = FuncAnimation(
            fig, update, frames=100, interval=50, blit=True,
            fargs=(line_real, None, psi)
        )
    
    st.session_state.animation = anim
    st.pyplot(fig)
    
    # Information about the selected quantum system
    if st.session_state.potential_type == "Harmonic Oscillator":
        st.markdown("""
        ### Harmonic Oscillator
        The quantum harmonic oscillator is a quantum-mechanical analog of the classical harmonic oscillator.
        
        **Key properties:**
        - Energy levels: E_n = ℏω(n + 1/2)
        - Equally spaced energy levels
        - Zero-point energy: E_0 = ℏω/2
        - Wavefunction involves Hermite polynomials
        """)
    
    elif st.session_state.potential_type == "Infinite Square Well":
        st.markdown("""
        ### Infinite Square Well
        Also known as the particle in a box, this is one of the simplest quantum systems.
        
        **Key properties:**
        - Energy levels: E_n = (n²π²ℏ²)/(2mL²)
        - Wavefunctions are sinusoidal inside the well and zero outside
        - No tunneling occurs due to infinite potential barriers
        """)
    
    elif st.session_state.potential_type == "Quantum Tunneling":
        st.markdown("""
        ### Quantum Tunneling
        A quantum mechanical phenomenon where particles can penetrate a potential barrier that would be impossible to overcome according to classical mechanics.
        
        **Key properties:**
        - Transmission coefficient depends on barrier height and width
        - Critical for many physical processes: alpha decay, nuclear fusion, etc.
        - Basis for technologies like tunneling diodes and STM microscopy
        """)

with tab2:
    st.header("Key Quantum Concepts")
    
    concept = st.selectbox(
        "Select a quantum concept to explore:",
        ["Wave-Particle Duality", "Uncertainty Principle", "Quantum Superposition", 
         "Quantum Entanglement", "Quantum Measurement"]
    )
    
    if concept == "Wave-Particle Duality":
        st.subheader("Wave-Particle Duality")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Wave-particle duality refers to the concept that every particle or quantum entity exhibits both wave and particle properties.
            
            - Light can behave as electromagnetic waves (diffraction, interference) or as particles called photons (photoelectric effect)
            - Electrons can behave as particles with charge and mass, or as waves that create diffraction patterns
            - The de Broglie wavelength relates a particle's momentum to its wavelength: λ = h/p
            
            The double-slit experiment dramatically demonstrates this duality: individual particles build up an interference pattern over time.
            """)
        
        with col2:
            # Simple visualization for double slit pattern
            fig, ax = plt.subplots(figsize=(5, 4))
            x = np.linspace(-5, 5, 1000)
            # Simulating double-slit interference pattern
            intensity = np.sin(3*x)**2 * np.sin(0.5*x)**2 / (0.1 + x**2)
            ax.plot(x, intensity)
            ax.set_title("Double-slit Interference Pattern")
            ax.set_xlabel("Position")
            ax.set_ylabel("Intensity")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    elif concept == "Uncertainty Principle":
        st.subheader("Heisenberg Uncertainty Principle")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            The Heisenberg uncertainty principle states that there is a fundamental limit to the precision with which complementary variables can be known simultaneously.
            
            The most famous form relates position and momentum:
            
            $$\\Delta x \\cdot \\Delta p \\geq \\frac{\\hbar}{2}$$
            
            Where:
            - Δx is the uncertainty in position
            - Δp is the uncertainty in momentum
            - ℏ is the reduced Planck constant
            
            This is not due to measurement limitations but is a fundamental property of quantum systems.
            """)
        
        with col2:
            # Visualization of uncertainty relation
            fig, ax = plt.subplots(figsize=(5, 4))
            
            # Generate data for visualization
            dx = np.linspace(0.1, 5, 100)
            dp = 0.5/dx  # Proportional to 1/dx to show the relationship
            
            ax.plot(dx, dp)
            ax.set_title("Position-Momentum Uncertainty Relation")
            ax.set_xlabel("Δx (Position Uncertainty)")
            ax.set_ylabel("Δp (Momentum Uncertainty)")
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 5)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    elif concept == "Quantum Superposition":
        st.subheader("Quantum Superposition")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Quantum superposition is the principle that quantum systems can exist in multiple states simultaneously until measured.
            
            - A quantum system can be described as a combination of multiple states
            - The wavefunction gives probability amplitudes for each possible state
            - Upon measurement, the system "collapses" to one definite state
            - Mathematically represented as: |ψ⟩ = α|0⟩ + β|1⟩
            
            This principle is fundamental to quantum computing, where qubits can represent both 0 and 1 simultaneously.
            """)
        
        with col2:
            # Simple visualization for superposition
            fig, ax = plt.subplots(figsize=(5, 4))
            
            theta = np.linspace(0, 2*np.pi, 100)
            x = np.sin(theta)
            y = np.cos(theta)
            z = np.sin(2*theta)
            
            ax.plot(theta, x, 'b-', label='State |0⟩')
            ax.plot(theta, y, 'r-', label='State |1⟩')
            ax.plot(theta, 0.7*x + 0.3*y, 'g-', label='Superposition')
            ax.set_title("Quantum State Superposition")
            ax.set_xlabel("Phase")
            ax.set_ylabel("Amplitude")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    elif concept == "Quantum Entanglement":
        st.subheader("Quantum Entanglement")
        
        st.markdown("""
        Quantum entanglement occurs when pairs or groups of particles interact in ways such that the quantum state of each particle cannot be described independently of the others.
        
        - Entangled particles remain connected so that actions performed on one affect the other, regardless of distance
        - Einstein referred to this as "spooky action at a distance"
        - Entanglement is the basis for quantum teleportation and many quantum computing operations
        - Bell's inequality provides a way to test for entanglement vs. classical "hidden variable" theories
        
        Entanglement has been verified experimentally many times and is one of the most counterintuitive aspects of quantum mechanics.
        """)
        
        # Visualization for entangled states
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Simulating correlated measurements
        np.random.seed(42)
        n_measurements = 100
        
        # Generate correlated random values
        shared_random = np.random.normal(0, 1, n_measurements)
        alice_results = np.sign(shared_random + 0.1*np.random.normal(0, 1, n_measurements))
        bob_results = np.sign(shared_random + 0.1*np.random.normal(0, 1, n_measurements))
        
        # Plot correlation
        ax.scatter(alice_results, bob_results, alpha=0.5)
        ax.set_title("Simulated Correlation in Entangled Particle Measurements")
        ax.set_xlabel("Alice's Measurement")
        ax.set_ylabel("Bob's Measurement")
        ax.set_xticks([-1, 1])
        ax.set_yticks([-1, 1])
        ax.set_xticklabels(["-1/2", "+1/2"])
        ax.set_yticklabels(["-1/2", "+1/2"])
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    elif concept == "Quantum Measurement":
        st.subheader("Quantum Measurement")
        
        st.markdown("""
        Quantum measurement refers to the process by which quantum systems are observed, causing their wavefunctions to collapse into definite states.
        
        - Before measurement, a quantum system exists in a superposition of possible states
        - The act of measurement causes the system to "collapse" to a specific eigenstate
        - The probability of measuring a particular eigenvalue is given by |⟨φ|ψ⟩|²
        - This collapse is instantaneous and non-local
        
        Various interpretations of quantum mechanics (Copenhagen, Many-Worlds, etc.) offer different perspectives on what happens during measurement.
        """)
        
        # Visualization for measurement collapse
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Generate example wavefunction
        x = np.linspace(-6, 6, 1000)
        psi = np.exp(-(x+2)**2/2) + np.exp(-(x-2)**2/2)  # Superposition of two Gaussians
        psi = psi / np.sqrt(np.sum(psi**2))  # Normalize
        
        # Generate collapsed states
        collapsed1 = 1.5 * np.exp(-(x+2)**2/2)
        collapsed1 = collapsed1 / np.sqrt(np.sum(collapsed1**2))
        
        collapsed2 = 1.5 * np.exp(-(x-2)**2/2)
        collapsed2 = collapsed2 / np.sqrt(np.sum(collapsed2**2))
        
        # Plot
        ax.plot(x, psi**2, 'b-', label='Before measurement (probability)')
        ax.plot(x, collapsed1**2, 'r--', label='Possible outcome 1')
        ax.plot(x, collapsed2**2, 'g--', label='Possible outcome 2')
        ax.fill_between(x, 0, psi**2, color='b', alpha=0.2)
        
        # Add arrow to indicate collapse
        ax.annotate('', xy=(0, 0.15), xytext=(0, 0.25),
                   arrowprops=dict(facecolor='black', shrink=0.05))
        ax.text(0, 0.3, 'Measurement', ha='center')
        
        ax.set_title("Wavefunction Collapse During Measurement")
        ax.set_xlabel("Position")
        ax.set_ylabel("Probability Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

with tab3:
    st.header("Ask Gemini About Quantum Physics")
    
    if not api_key:
        st.warning("Please enter your Gemini API key in the sidebar to use this feature.")
    else:
        query = st.text_input("Ask a question about quantum mechanics:", 
                         "What is the significance of Schrödinger's Cat thought experiment?")
        
        if st.button("Ask Gemini"):
            try:
                with st.spinner("Gemini is thinking..."):
                    # Configure the model
                    model = genai.GenerativeModel('gemini-pro')
                    
                    # Generate response
                    response = model.generate_content(
                        f"Explain the following quantum mechanics concept in detail, with clear explanations "
                        f"suitable for someone with basic physics knowledge: {query}"
                    )
                    
                    # Display response
                    st.markdown("### Gemini's Response:")
                    st.markdown(response.text)
                    
                    # Add citation
                    st.markdown("---")
                    st.caption("Generated by Google Gemini API")
            except Exception as e:
                st.error(f"Error communicating with Gemini API: {e}")
                st.info("Make sure your API key is correct and that you have access to the Gemini API.")

# Footer
st.markdown("---")
st.markdown("Created for Quantum Day 2025 | ⚛️ Happy exploring the quantum realm!")
