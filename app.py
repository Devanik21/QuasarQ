import streamlit as st
import google.generativeai as genai
import random

# Page configuration
st.set_page_config(
    page_title=" Quantum Day Interactive Explorer",
    page_icon="⚛️",
    layout="wide"
)

# Initialize session state
if 'potential_type' not in st.session_state:
    st.session_state.potential_type = "Harmonic Oscillator"

# Sidebar configuration
with st.sidebar:
    st.title("🌌 Quantum Explorer Portal")
    st.success("Welcome, Quantum Adventurer! ✨")

    # Gemini API Key input
    api_key = st.text_input("🔐 Enter Gemini API Key", type="password")

    if api_key:
        try:
            genai.configure(api_key=api_key)
            st.success("🔓 API Key Verified!")
        except Exception as e:
            st.error(f"❌ API Error: {e}")

    st.subheader("🧪 Simulation Setup")
    st.session_state.potential_type = st.selectbox("🎯 Choose Potential", ["Harmonic Oscillator", "Infinite Square Well", "Quantum Tunneling"])
    st.session_state.n_state = st.slider("🎚️ Quantum State (n)", 1, 5, 1)
    st.session_state.animation_speed = st.slider("⚡ Animation Speed", 0.1, 2.0, 1.0)

# Main Interface
st.title("⚛️ Quantum Mechanics Interactive Explorer")
st.markdown("""
🚀 Dive into the mysteries of the quantum realm. Customize your simulation, explore key principles, and ask Gemini anything about the world of quantum physics!
""")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔬 Simulator", "📚 Concepts", "🤖 Ask Gemini"])

with tab1:
    st.header("🌀 Wavefunction Simulator")
    st.info("Visualization turned off for performance. Use this portal to mentally simulate quantum systems ✨")

    tips = {
        "Harmonic Oscillator": "Think of a springy quantum trampoline — particles dance in smooth energy steps!",
        "Infinite Square Well": "Like a bouncy ball trapped in a perfect quantum box. No escape, pure math.",
        "Quantum Tunneling": "Defy classical logic — walk through walls! Quantum particles do it all the time."
    }

    st.markdown(f"### ⚙️ Selected System: {st.session_state.potential_type}")
    st.markdown(f"**Quantum State (n):** {st.session_state.n_state}")
    st.markdown(f"💡 *Quantum Insight:* {tips.get(st.session_state.potential_type, '')}*")

with tab2:
    st.header("📖 Explore Quantum Concepts")
    concept = st.selectbox("🔎 Choose a concept to explore:", [
        "Wave-Particle Duality", "Uncertainty Principle", "Quantum Superposition",
        "Quantum Entanglement", "Quantum Measurement"
    ])

    concept_explanations = {
        "Wave-Particle Duality": "🎭 Everything in the quantum world plays dual roles — wave AND particle! From light to electrons, they do it all.",
        "Uncertainty Principle": "🤯 You can't know it all. The more you know about a particle's position, the less you know about its momentum.",
        "Quantum Superposition": "🌀 Schrödinger's cat lives in multiverses. A quantum state can be 0 and 1 until you peek!",
        "Quantum Entanglement": "🧠 Telepathy for particles? Entangled particles share state instantly, across space!",
        "Quantum Measurement": "🎯 Collapse! When we observe a quantum system, it picks a state — randomly but precisely."
    }

    st.subheader(f"🧠 {concept}")
    st.markdown(concept_explanations[concept])

    quantum_quotes = [
        "\"God does not play dice with the universe.\" – Einstein",
        "\"Anyone who is not shocked by quantum theory has not understood it.\" – Niels Bohr",
        "\"If you think you understand quantum mechanics, you don't understand quantum mechanics.\" – Richard Feynman"
    ]
    st.caption(random.choice(quantum_quotes))

with tab3:
    st.header("🤖 Ask Gemini Anything About Quantum Physics")

    if not api_key:
        st.warning("🔐 Please enter your Gemini API key in the sidebar.")
    else:
        question = st.text_input("❓ What do you want to know?", "What is quantum decoherence?")

        if st.button("💬 Ask Gemini"):
            try:
                with st.spinner("🧠 Gemini is thinking..."):
                    model = genai.GenerativeModel('gemini-pro')
                    response = model.generate_content(
                        f"Explain this quantum physics concept for curious minds: {question}"
                    )
                    st.markdown("### 📘 Gemini's Answer")
                    st.markdown(response.text)
                    st.caption("Powered by Google Gemini API")
            except Exception as e:
                st.error(f"Gemini API error: {e}")

# Footer
st.markdown("---")
st.markdown("✨ Created for Quantum Day 2025 | Explore, Learn, Question ✨")
