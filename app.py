import streamlit as st
from main import query_chroma, query_neo4j, build_prompt, llm
from datetime import datetime
import base64
import speech_recognition as sr

# ===== PAGE CONFIG =====
st.set_page_config(page_title="MediSage - Smart Health Assistant", layout="centered")

# ===== CUSTOM CSS =====
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
        margin: auto;
        background: rgba(0, 0, 0, 0.6);
        border-radius: 20px;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.2);
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 10px;
        border: 2px solid #00BFFF;
        background-color: #121212;
        color: white;
    }
    .askneo-title {
        font-size: 2.8rem;
        text-align: center;
        font-weight: bold;
        color: #00BFFF;
        margin-bottom: 0.5rem;
    }
    .askneo-sub {
        text-align: center;
        font-size: 1.2rem;
        color: #cccccc;
        margin-bottom: 2rem;
    }
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }
    .mic-button button {
        height: 3.2rem !important;
        width: 3.2rem !important;
        font-size: 1.4rem;
        border-radius: 10px;
        margin-top: 0.2rem;
        margin-left: 0.1rem;
        border: 2px solid #00BFFF;
        background-color: #111;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ===== LOGO =====
st.markdown(
    """
    <div class="logo-container">
        <img src="data:image/png;base64,{}" width="160">
    </div>
    """.format(base64.b64encode(open("assets/askneo_logo.png", "rb").read()).decode()),
    unsafe_allow_html=True
)

# ===== TITLE =====
st.markdown('<div class="askneo-title">MediSage - Smart Health Assistant 🤖</div>', unsafe_allow_html=True)
st.markdown('<div class="askneo-sub">Ask me anything about diseases, symptoms, or precautions.</div>', unsafe_allow_html=True)

# ===== VOICE FUNCTION =====
def transcribe_audio():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.toast("🎙️ Listening... Speak clearly", icon="🎤")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
    except sr.WaitTimeoutError:
        st.warning("⏰ Listening timed out. Please try again.")
        return ""
    except Exception as e:
        st.error(f"Microphone error: {str(e)}")
        return ""

    try:
        text = recognizer.recognize_google(audio)
        st.toast(f"🗣️ You said: {text}")
        return text
    except sr.UnknownValueError:
        st.warning("⚠️ Couldn't understand audio.")
    except sr.RequestError:
        st.error("🔌 Speech Recognition service unavailable.")
    return ""

# ===== STATE INIT =====
if "temp_input" not in st.session_state:
    st.session_state.temp_input = ""
if "auto_run_query" not in st.session_state:
    st.session_state.auto_run_query = False

# ===== INPUT FIELD & MIC BUTTON =====
st.markdown("### 💬 Your Question")
col1, col2 = st.columns([10, 1])
with col1:
    user_input = st.text_input("Type or speak...", value=st.session_state.temp_input, key="input_box")
with col2:
    with st.container():
        st.markdown('<div class="mic-button">', unsafe_allow_html=True)
        mic = st.button("🎙️", key="mic_btn")
        st.markdown('</div>', unsafe_allow_html=True)

        if mic:
            spoken_text = transcribe_audio()
            if spoken_text:
                st.session_state.temp_input = spoken_text
                st.session_state.auto_run_query = True
                st.rerun()

# ===== AUTO RUN QUERY =====
run_query = False
if st.session_state.auto_run_query or user_input.strip():
    run_query = True
    st.session_state.auto_run_query = False

# ===== QUERY PIPELINE =====
if run_query and user_input:
    with st.spinner("🧠 Thinking..."):
        try:
            semantic_result = query_chroma(user_input)
            disease = semantic_result.split(":")[0].strip().title()
            graph_result = query_neo4j(disease)
            prompt = build_prompt(user_input, semantic_result, graph_result)
            response = llm.invoke(prompt)

            st.markdown("## 🤖 MediSage Says:")
            st.success(response.content)

            with st.expander("🔍 View Reasoning Context"):
                st.markdown(f"**Semantic Result:** {semantic_result}")
                st.markdown(f"**Detected Disease:** {disease}")
                st.markdown("**Graph Result (Precautions):**")
                st.json(graph_result)

            with open("logs.txt", "a", encoding="utf-8") as f:
                f.write(f"\n---\nTimestamp: {datetime.now()}\nUser: {user_input}\nDisease: {disease}\nGraph: {graph_result}\nResponse: {response.content}\n")
        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")