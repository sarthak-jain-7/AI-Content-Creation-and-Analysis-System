"""
========================================================================================
AI Prompt Chatbot System
========================================================================================
College: Walchand Institute of Technology, Solapur
Department: Information Technology
Course: Program Elective – V (Prompt Engineering)

Features:
- Pure, modern, full-featured Conversational AI Chatbot
- Responds accurately in real-time to any user prompt
- Customizable System Prompts / Personas & Custom Instructions
- Adjustable Parameters (Temperature, Top-P, Max Tokens, Model selection)
- Powered by high-speed Groq Cloud (openai/gpt-oss-120b, qwen/qwen3.8-27b) & OpenAI
- Chat history management, preset prompts, message export & token counters
========================================================================================
"""

import os
import time
import json
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ======================================================================================
# PAGE CONFIGURATION & STYLING
# ======================================================================================
st.set_page_config(
    page_title="AI Prompt Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern chat UI
st.markdown("""
<style>
    /* Header styling */
    .chat-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1.2rem 1.6rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .chat-header h1 {
        color: white;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
        padding: 0;
    }
    .chat-header p {
        color: #dbeafe;
        font-size: 0.9rem;
        margin: 0.3rem 0 0 0;
    }
    .badge-sub {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        margin-top: 5px;
        font-weight: 500;
    }
    
    /* Persona tag */
    .persona-badge {
        display: inline-block;
        background-color: #e0e7ff;
        color: #3730a3;
        padding: 3px 12px;
        border-radius: 15px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    /* Chat message container styling */
    .stChatMessage {
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================================================
# SIDEBAR CONFIGURATION & CONTROLS
# ======================================================================================
env_api_key = os.getenv("OPENAI_API_KEY", "")

with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/chat.png", width=85)
    st.title("🤖 Chatbot Controls")
    
    st.markdown("### 🔑 API Configuration")
    user_api_key = st.text_input(
        "API Key (Groq or OpenAI):",
        value=env_api_key,
        type="password",
        help="Supports Groq (gsk_...) and OpenAI (sk-...) keys."
    )
    
    effective_api_key = user_api_key.strip() if user_api_key else env_api_key.strip()
    is_groq = effective_api_key.startswith("gsk_")
    
    if is_groq:
        available_models = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "allam-2-7b"
        ]
        provider_name = "Groq High-Speed API"
    else:
        available_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        provider_name = "OpenAI API"
        
    selected_model = st.selectbox(
        "AI Model:",
        options=available_models,
        index=0,
        help="Choose the underlying LLM model."
    )
    
    if effective_api_key:
        st.success(f"🟢 {provider_name} Connected")
    else:
        st.warning("🟡 No API Key Configured")
        
    st.divider()
    
    st.markdown("### 🎭 System Persona / Role")
    persona_options = {
        "🌟 General AI Assistant": "You are a highly accurate, knowledgeable, and helpful AI assistant.",
        "🎓 Prompt Engineering Mentor": "You are an expert Prompt Engineering professor at WIT Solapur. Provide comprehensive, structured, and insightful answers following best prompt engineering principles.",
        "🐍 Python & CS Tutor": "You are an expert Python developer and Computer Science tutor. Provide clear code snippets, step-by-step explanations, and best practices.",
        "✍️ Creative Content Writer": "You are a creative writer who generates vivid, engaging, and imaginative stories, posts, and poems.",
        "⚡ Strict & Concise Problem Solver": "You are a concise problem solver. Give direct, factual, and actionable answers with zero unnecessary fluff.",
        "🛠️ Custom Persona": "Custom"
    }
    
    selected_persona_name = st.selectbox(
        "Choose Chat Persona:",
        options=list(persona_options.keys()),
        index=0
    )
    
    if selected_persona_name == "🛠️ Custom Persona":
        custom_system_prompt = st.text_area(
            "Custom System Instruction:",
            value="You are an expert AI tailored for university-level computer science.",
            height=90
        )
        active_system_prompt = custom_system_prompt
    else:
        active_system_prompt = persona_options[selected_persona_name]
        st.caption(f"_{active_system_prompt}_")
        
    st.divider()
    
    st.markdown("### ⚙️ Generation Hyperparameters")
    temperature = st.slider("Temperature (Creativity):", 0.0, 1.5, 0.7, 0.1, help="Higher = More creative; Lower = More focused and deterministic.")
    top_p = st.slider("Top-P (Nucleus Sampling):", 0.1, 1.0, 1.0, 0.05, help="Controls token diversity.")
    max_tokens = st.slider("Max Output Tokens:", 100, 2048, 800, 50)
    
    st.divider()
    if st.button("🗑️ Clear Chat History", type="secondary", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
        
    st.divider()
    st.markdown("""
    <div style="font-size: 0.8rem; color: #64748b;">
        <strong>Walchand Institute of Technology, Solapur</strong><br>
        Department of Information Technology<br>
        Program Elective – V (Prompt Engineering)
    </div>
    """, unsafe_allow_html=True)


# ======================================================================================
# UNIVERSAL LLM COMPLETION FUNCTION
# ======================================================================================
def generate_chat_response(messages: list, api_key: str, model: str, temp: float, top_p_val: float, max_toks: int):
    """
    Calls Groq (if key begins with gsk_) or OpenAI with error handling.
    """
    if not api_key:
        return "⚠️ API Key is missing. Please enter your API key in the sidebar or add it to `.env`."
        
    try:
        from openai import OpenAI
        
        if api_key.startswith("gsk_"):
            client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        else:
            client = OpenAI(api_key=api_key)
            
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temp,
            top_p=top_p_val,
            max_tokens=max_toks,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"


# ======================================================================================
# MAIN CHAT INTERFACE
# ======================================================================================
st.markdown("""
<div class="chat-header">
    <h1>💬 AI Prompt Chatbot</h1>
    <p>Ask anything, test any prompt, or instruct the AI to generate accurate real-time answers</p>
    <span class="badge-sub">Walchand Institute of Technology, Solapur • Department of IT</span>
</div>
""", unsafe_allow_html=True)

# Active status bar
st.markdown(f'<span class="persona-badge">🎭 Active Persona: {selected_persona_name}</span> &nbsp; <span class="persona-badge">⚡ Model: {selected_model}</span> &nbsp; <span class="persona-badge">🌡️ Temp: {temperature}</span>', unsafe_allow_html=True)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": f"Hello! 👋 I am your AI Chatbot powered by **{selected_model}**.\n\nYou can give me **any prompt, question, coding task, or topic**, and I will generate an accurate answer for you. How can I help you today?"}
    ]

# Display all messages in history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Quick Prompt Suggestions
st.markdown("##### 💡 Suggested Prompts:")
col_q1, col_q2, col_q3, col_q4 = st.columns(4)
clicked_prompt = None

with col_q1:
    if st.button("🚀 Prompt Engineering Guide", use_container_width=True):
        clicked_prompt = "What is Prompt Engineering? Explain Role, Context, Constraints, and Output Format with clear examples."
with col_q2:
    if st.button("🐍 Python Binary Search", use_container_width=True):
        clicked_prompt = "Write a complete Python program for Binary Search with comments, test cases, and time complexity analysis."
with col_q3:
    if st.button("🏥 AI in Healthcare", use_container_width=True):
        clicked_prompt = "Explain 5 transformative applications of AI in Healthcare with real-world case studies."
with col_q4:
    if st.button("🔬 Temp 0.2 vs 0.9 Explained", use_container_width=True):
        clicked_prompt = "Explain how Temperature (0.2 vs 0.9) and Top-P affect LLM token sampling mathematically and practically."

# Chat Input Box
user_prompt = st.chat_input("Type any prompt or question here...")

# Resolve input (either from button or text input)
effective_input = clicked_prompt if clicked_prompt else user_prompt

if effective_input:
    # 1. Append & render User message
    st.session_state.chat_history.append({"role": "user", "content": effective_input})
    with st.chat_message("user"):
        st.markdown(effective_input)

    # 2. Build payload with system prompt + chat history
    llm_payload = [{"role": "system", "content": active_system_prompt}]
    
    # Include up to last 10 messages for conversation context
    for past_msg in st.session_state.chat_history[-10:]:
        llm_payload.append({"role": past_msg["role"], "content": past_msg["content"]})

    # 3. Call LLM & render Assistant response
    with st.chat_message("assistant"):
        with st.spinner(f"Generating answer using {selected_model}..."):
            start_time = time.time()
            bot_reply = generate_chat_response(
                messages=llm_payload,
                api_key=effective_api_key,
                model=selected_model,
                temp=temperature,
                top_p_val=top_p,
                max_toks=max_tokens
            )
            latency = round(time.time() - start_time, 2)
            
            st.markdown(bot_reply)
            st.caption(f"⏱️ Response generated in {latency}s | Model: `{selected_model}`")

    # 4. Save Assistant message to history
    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
