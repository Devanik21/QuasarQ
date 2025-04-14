import streamlit as st
import google.generativeai as genai

# Page configuration
st.set_page_config(
    page_title="Quantum Day Interactive Explorer",
    page_icon="⚛️",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'potential_type' not in st.session_state:
    st.session_state.potential_type = "Harmonic Oscillator"

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
    animation_speed = st.slider("Animation Speed", 0.1, 2.0, 1.0)

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
    st.header("Wavefunction Simulator")
    st.markdown(f"Simulating quantum state: **{st.session_state.potential_type}** with n = {st.session_state.get('n_state', 1)}")
    st.info("Visualization and animation features have been removed for this version.")

    if st.session_state.potential_type == "Harmonic Oscillator":
        st.markdown("""
        ### Harmonic Oscillator
        The quantum harmonic oscillator is a quantum-mechanical analog of the classical harmonic oscillator.
        
        **Key properties:**
        - Energy levels: E_n = ℓω(n + 1/2)
        - Equally spaced energy levels
        - Zero-point energy: E_0 = ℓω/2
        - Wavefunction involves Hermite polynomials
        """)

    elif st.session_state.potential_type == "Infinite Square Well":
        st.markdown("""
        ### Infinite Square Well
        Also known as the particle in a box, this is one of the simplest quantum systems.
        
        **Key properties:**
        - Energy levels: E_n = (n²π²ℓ²)/(2mL²)
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

    explanations = {
        "Wave-Particle Duality": "Wave-particle duality refers to the concept that every particle or quantum entity exhibits both wave and particle properties.",
        "Uncertainty Principle": "The Heisenberg uncertainty principle states that there is a fundamental limit to the precision with which complementary variables can be known simultaneously.",
        "Quantum Superposition": "Quantum superposition is the principle that quantum systems can exist in multiple states simultaneously until measured.",
        "Quantum Entanglement": "Quantum entanglement occurs when particles are correlated in such a way that the quantum state of one cannot be described independently of the other.",
        "Quantum Measurement": "Quantum measurement refers to the process by which quantum systems are observed, causing their wavefunctions to collapse into definite states."
    }

    st.markdown(f"### {concept}\n{explanations[concept]}")
    st.info("Visual explanations have been removed for this version.")

with tab3:
    st.header("Ask Gemini About Quantum Physics")

    if not api_key:
        st.warning("Please enter your Gemini API key in the sidebar to use this feature.")
    else:
        query = st.text_input("Ask a question about quantum mechanics:", "What is the significance of Schrödinger's Cat thought experiment?")

        if st.button("Ask Gemini"):
            try:
                with st.spinner("Gemini is thinking..."):
                    model = genai.GenerativeModel('gemini-pro')
                    response = model.generate_content(
                        f"Explain the following quantum mechanics concept in detail, with clear explanations suitable for someone with basic physics knowledge: {query}"
                    )

                    st.markdown("### Gemini's Response:")
                    st.markdown(response.text)
                    st.markdown("---")
                    st.caption("Generated by Google Gemini API")
            except Exception as e:
                st.error(f"Error communicating with Gemini API: {e}")
                st.info("Make sure your API key is correct and that you have access to the Gemini API.")

# Footer
st.markdown("---")
st.markdown("Created for Quantum Day 2025 | ⚛️ Happy exploring the quantum realm!")
