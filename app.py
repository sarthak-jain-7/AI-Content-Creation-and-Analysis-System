"""
========================================================================================
AI-Powered Content Creation, Analysis and Interactive Chatbot System
========================================================================================
College: Walchand Institute of Technology, Solapur
Department: Information Technology
Course: Program Elective – V (Prompt Engineering)
Assignment: Assignment No. 8 – Integrated Coding Assignment

Features Implemented:
1. 💬 AI Prompt Chatbot (Live conversational AI answering any prompt accurately)
2. 📐 Prompt Design (Role, Context, Constraints, Output Format)
3. ✍️ Content Generation (Social Media Post, Short Story, Poem)
4. 🎙️ Podcast Planning (Title, Description, Guest Type, 8 Interview Questions)
5. 🔍 Text Analysis (Sentiment Analysis + Keyword Extraction with Live AI)
6. 🧪 Parameter Experimentation (Side-by-side Temperature & Top-P comparison)
========================================================================================
"""

import os
import re
import json
import time
from collections import Counter
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ======================================================================================
# PAGE CONFIGURATION & STYLING
# ======================================================================================
st.set_page_config(
    page_title="AI-Powered Content Creation & Prompt Chatbot System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main header styling */
    .college-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1.2rem 1.6rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .college-header h1 {
        color: white;
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0;
        padding: 0;
    }
    .college-header p {
        color: #dbeafe;
        font-size: 0.92rem;
        margin: 0.3rem 0 0 0;
    }
    .badge-sub {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-top: 4px;
        font-weight: 500;
    }
    
    /* Card design */
    .result-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }
    .prompt-component-box {
        background-color: #ffffff;
        border-left: 4px solid #2563eb;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .experiment-card {
        border-radius: 10px;
        padding: 1rem;
        background: #ffffff;
        border: 1px solid #cbd5e1;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        min-height: 280px;
    }
    .tag-badge {
        display: inline-block;
        background-color: #e0e7ff;
        color: #3730a3;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 2px 4px 4px 0;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================================================
# API SETUP & DETECTIONS (Supports Groq & OpenAI)
# ======================================================================================
env_api_key = os.getenv("OPENAI_API_KEY", "")

with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/brain.png", width=90)
    st.title("⚙️ AI Engine Config")
    
    st.markdown("### 🔑 API Key & Provider")
    user_api_key = st.text_input(
        "API Key (Groq or OpenAI):",
        value=env_api_key,
        type="password",
        help="Supports both Groq (gsk_...) and OpenAI (sk-...) keys."
    )
    
    effective_api_key = user_api_key.strip() if user_api_key else env_api_key.strip()
    
    # Detect provider based on key prefix
    is_groq = effective_api_key.startswith("gsk_")
    provider_name = "Groq High-Speed Cloud" if is_groq else "OpenAI Cloud"
    
    if is_groq:
        available_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "allam-2-7b"]
        default_model = "openai/gpt-oss-120b"
    else:
        available_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        default_model = "gpt-4o-mini"
        
    selected_model = st.selectbox(
        "Active LLM Model:",
        options=available_models,
        index=0,
        help="Model used to generate responses."
    )
    
    if effective_api_key:
        st.success(f"🟢 {provider_name} Connected!")
        st.caption(f"🚀 Model: `{selected_model}`")
    else:
        st.warning("🟡 No API Key (Local Fallback Mode Active)")
        
    st.divider()
    
    st.markdown("### 🏫 Academic Details")
    st.markdown("""
    **College:** Walchand Institute of Technology, Solapur  
    **Dept:** Information Technology  
    **Course:** PE - V (Prompt Engineering)  
    **Assignment:** No. 8 (Integrated Coding)  
    """)


# ======================================================================================
# UNIVERSAL LLM CLIENT CALLER
# ======================================================================================
def call_llm(messages: list, api_key: str, model: str = "openai/gpt-oss-120b", temperature: float = 0.7, top_p: float = 1.0, max_tokens: int = 700):
    """
    Calls Groq (if key starts with gsk_) or OpenAI with robust error handling.
    """
    if not api_key:
        return None, "API Key is missing. Please provide a valid API key."
    
    try:
        from openai import OpenAI
        
        if api_key.startswith("gsk_"):
            # Groq API endpoint
            client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        else:
            # Standard OpenAI endpoint
            client = OpenAI(api_key=api_key)
            
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content, None
    except Exception as e:
        error_msg = str(e)
        return None, f"API Error: {error_msg}"


# ======================================================================================
# LOCAL FALLBACK ENGINES
# ======================================================================================
def local_fallback_sentiment_analysis(text: str):
    positive_words = {"good", "great", "excellent", "useful", "helpful", "positive", "beneficial", "easier", "fast", "efficient", "valuable", "effective", "better", "best", "innovative", "accurate", "reliable", "progress", "success", "quick", "improve"}
    negative_words = {"bad", "harmful", "useless", "negative", "detrimental", "difficult", "slow", "inefficient", "risk", "danger", "error", "flaw", "fail", "threat", "vulnerable", "inaccurate", "unreliable", "bias", "problem", "issue", "avoid"}
    
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    total = pos_count + neg_count
    
    if total == 0:
        score = 0.0
        sentiment = "Neutral"
    else:
        score = (pos_count - neg_count) / max(total, 1)
        if score > 0.15:
            sentiment = "Positive"
        elif score < -0.15:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
    return {
        "sentiment": sentiment,
        "score": round(score, 2),
        "reasoning": f"Identified {pos_count} positive and {neg_count} negative keywords.",
        "mode": "Local Fallback Lexicon"
    }


def local_fallback_keyword_extraction(text: str, top_n: int = 8):
    stopwords = {"the", "and", "is", "in", "to", "of", "it", "a", "an", "that", "this", "for", "on", "with", "as", "by", "at", "from", "be", "are", "was", "were", "or", "not", "but", "what", "all", "we", "when", "your", "can", "said", "there", "use", "which", "do", "how", "if", "will", "up", "about", "out", "then", "them", "these", "so", "some", "would", "make", "like", "into", "has", "more", "no", "way", "could", "than", "now", "should", "makes", "avoid"}
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered_words = [w for w in words if w not in stopwords]
    counts = Counter(filtered_words)
    return [w for w, _ in counts.most_common(top_n)]


# ======================================================================================
# UI HEADER & TOPIC SELECTION
# ======================================================================================
st.markdown("""
<div class="college-header">
    <h1>🤖 AI-Powered Content Creation, Analysis & Prompt Chatbot</h1>
    <p>Integrated Coding Assignment – Prompt Engineering | Department of Information Technology</p>
    <span class="badge-sub">Walchand Institute of Technology, Solapur • Program Elective – V</span>
</div>
""", unsafe_allow_html=True)

col_topic, col_quick = st.columns([2.5, 1.5])

with col_topic:
    user_topic = st.text_input(
        "Enter Active Topic (for Content, Podcast & Analysis modules):",
        value="AI in Healthcare",
        help="Topic used across generation and analysis tabs."
    )

with col_quick:
    preset_topic = st.selectbox(
        "Or Select a Preset Topic:",
        options=["(Custom Input)", "AI in Healthcare", "Artificial Intelligence", "Machine Learning", "Cybersecurity", "Cloud Computing", "Blockchain"],
        index=0
    )
    if preset_topic != "(Custom Input)":
        user_topic = preset_topic

active_topic = user_topic.strip() if user_topic.strip() else "AI in Healthcare"


# ======================================================================================
# MAIN APPLICATION TABS
# ======================================================================================
tab_chat, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 AI Prompt Chatbot",
    "1. 📐 Prompt Design",
    "2. ✍️ Content Generation",
    "3. 🎙️ Podcast Planning",
    "4. 🔍 Text Analysis",
    "5. 🧪 Parameter Experiment"
])


# ======================================================================================
# TAB: AI PROMPT CHATBOT (ASK ANYTHING / ACCURATE ANSWERS TO ANY PROMPT)
# ======================================================================================
with tab_chat:
    st.header("💬 Interactive AI Prompt Chatbot")
    st.markdown("Give **any prompt, question, code task, or instruction**, and the AI will respond instantly and accurately.")
    
    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": f"Hello! I am your AI Prompt Assistant powered by **{selected_model}**. You can ask me anything or test any prompt. What would you like help with today?"}
        ]
        
    col_chat_cfg1, col_chat_cfg2 = st.columns([2.5, 1.5])
    with col_chat_cfg1:
        persona = st.selectbox(
            "Select Chatbot Persona / System Role:",
            options=[
                "Helpful & Accurate AI Assistant",
                "Expert Prompt Engineering Mentor",
                "Python & Computer Science Tutor",
                "Creative Content Writer",
                "Strict Concise Problem Solver"
            ]
        )
    with col_chat_cfg2:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_messages = [
                {"role": "assistant", "content": "Chat history cleared. Send any prompt below!"}
            ]
            st.rerun()

    # Map persona to system prompt
    persona_prompts = {
        "Helpful & Accurate AI Assistant": "You are a highly accurate, knowledgeable, and helpful AI assistant.",
        "Expert Prompt Engineering Mentor": "You are an expert Prompt Engineering professor at WIT Solapur. Provide clear, educational answers demonstrating structured prompt design.",
        "Python & Computer Science Tutor": "You are an expert Python and Computer Science tutor. Give precise explanations, code snippets, and best practices.",
        "Creative Content Writer": "You are a creative writer who crafts engaging, vivid, and well-structured content.",
        "Strict Concise Problem Solver": "You are a concise problem solver. Give direct, fact-based answers with zero fluff."
    }

    # Render previous messages
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Quick prompt suggestion buttons
    st.caption("💡 Quick Prompts:")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    suggested_prompt = None
    with q_col1:
        if st.button("💡 Explain Zero-Shot Prompting"):
            suggested_prompt = "Explain Zero-Shot vs Few-Shot Prompting with simple examples."
    with q_col2:
        if st.button("🐍 Python Binary Search Code"):
            suggested_prompt = "Write a clean Python function for Binary Search with comments and time complexity."
    with q_col3:
        if st.button("🏥 AI in Healthcare Summary"):
            suggested_prompt = f"Summarize the latest breakthroughs of {active_topic} in 3 concise bullet points."
    with q_col4:
        if st.button("🎯 Craft a Perfect Prompt"):
            suggested_prompt = "Give me a step-by-step formula for crafting a perfect prompt with Role, Context, Constraints, and Output Format."

    # Chat Input
    user_prompt_input = st.chat_input("Enter any prompt or question here...")
    
    final_input = suggested_prompt if suggested_prompt else user_prompt_input

    if final_input:
        # Display user message
        st.session_state.chat_messages.append({"role": "user", "content": final_input})
        with st.chat_message("user"):
            st.markdown(final_input)

        # Generate LLM response
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking with {selected_model}..."):
                system_instruction = persona_prompts.get(persona, "You are a helpful AI assistant.")
                llm_messages = [{"role": "system", "content": system_instruction}]
                
                # Append last 6 turns for context
                for m in st.session_state.chat_messages[-6:]:
                    llm_messages.append({"role": m["role"], "content": m["content"]})
                
                start_t = time.time()
                reply, err = call_llm(llm_messages, effective_api_key, model=selected_model, temperature=0.7)
                elapsed = round(time.time() - start_t, 2)
                
                if err:
                    st.error(f"⚠️ {err}")
                    reply = f"Error communicating with AI model: {err}"
                else:
                    st.markdown(reply)
                    st.caption(f"⚡ Generated in {elapsed}s using `{selected_model}`")
                    
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})


# ======================================================================================
# TAB 1: PROMPT DESIGN (FEATURE 2)
# ======================================================================================
with tab1:
    st.header("1. Structured Prompt Design")
    st.markdown("Demonstrating the **4 Pillars of Prompt Engineering**: **Role**, **Context**, **Constraints**, and **Output Format**.")
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.markdown("#### 🛠️ Component Breakdown")
        st.markdown('<div class="prompt-component-box"><strong>🎭 ROLE:</strong><br>You are an expert educational content writer.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="prompt-component-box"><strong>🌐 CONTEXT:</strong><br>Create useful and student-friendly content about <em>{active_topic}</em>.</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="prompt-component-box">
            <strong>⚙️ CONSTRAINTS:</strong>
            <ul style="margin-bottom:0; padding-left:20px;">
                <li>Use simple English</li>
                <li>Suitable for college students</li>
                <li>Avoid unsupported claims</li>
                <li>Keep the response concise</li>
                <li>Make the content educational and relevant</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="prompt-component-box">
            <strong>📋 OUTPUT FORMAT:</strong>
            <ol style="margin-bottom:0; padding-left:20px;">
                <li>Title</li>
                <li>Main content</li>
                <li>Key takeaway</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    structured_system_prompt = "You are an expert educational content writer."
    structured_user_prompt = f"""Create useful and student-friendly educational content about '{active_topic}'.

Constraints:
- Use simple English suitable for college students
- Avoid unsupported claims and maintain factual accuracy
- Keep the response concise and engaging (120-180 words)
- Make the content educational and directly relevant

Output Format:
1. Title
2. Main content
3. Key takeaway"""

    with col_p2:
        st.markdown("#### 📜 Assembled Structured Prompt")
        with st.expander("👁️ View Generated Structured Prompt", expanded=True):
            st.markdown(f"**System / Role Message:**\n```text\n{structured_system_prompt}\n```")
            st.markdown(f"**User Prompt Template:**\n```text\n{structured_user_prompt}\n```")

    st.markdown("---")
    if st.button("🚀 Execute Structured Prompt with Live LLM", key="btn_test_prompt", type="primary"):
        with st.spinner(f"Executing prompt for '{active_topic}' with {selected_model}..."):
            messages = [
                {"role": "system", "content": structured_system_prompt},
                {"role": "user", "content": structured_user_prompt}
            ]
            output, error = call_llm(messages, effective_api_key, model=selected_model)
            if error:
                st.error(f"⚠️ {error}")
            else:
                st.success(f"✅ Live Output Generated via `{selected_model}`:")
                st.markdown(f'<div class="result-card">{output}</div>', unsafe_allow_html=True)


# ======================================================================================
# TAB 2: CONTENT GENERATION (FEATURE 3)
# ======================================================================================
with tab2:
    st.header("2. AI Content Generation")
    
    col_cg1, col_cg2 = st.columns([1.5, 2.5])
    
    with col_cg1:
        content_type = st.selectbox(
            "Select Content Type:",
            options=["Social Media Post", "Short Story", "Poem"]
        )
        st.info(f"📌 **Target Topic:** `{active_topic}`")
        
        if content_type == "Social Media Post":
            specific_constraint = "- Length: 150–200 words\n- Include relevant emojis and student-friendly hashtags"
        elif content_type == "Short Story":
            specific_constraint = "- Length: 200–250 words\n- Include a student protagonist solving a challenge with clear moral"
        else:
            specific_constraint = "- Structure: 3-4 rhyming stanzas highlighting core theme"

        gen_button = st.button("✨ Generate Content", key="btn_generate_content", type="primary")

    cg_role = "You are a professional AI content writer."
    cg_full_prompt = f"""Create a high-quality {content_type.lower()} about '{active_topic}' for college students.

Constraints:
- Simple English suitable for college students
- Educational, accurate, and avoid unsupported claims
{specific_constraint}

Output Format:
1. Title
2. Main Content
3. Key Takeaway"""

    with col_cg2:
        with st.expander("🔍 View Structured Prompt Used"):
            st.code(f"ROLE: {cg_role}\n\nPROMPT:\n{cg_full_prompt}", language="markdown")

        if gen_button:
            with st.spinner(f"Generating {content_type} with {selected_model}..."):
                start_t = time.time()
                messages = [
                    {"role": "system", "content": cg_role},
                    {"role": "user", "content": cg_full_prompt}
                ]
                output, error = call_llm(messages, effective_api_key, model=selected_model)
                elapsed = round(time.time() - start_t, 2)
                
                if error:
                    st.error(f"⚠️ {error}")
                else:
                    st.success(f"✅ {content_type} Generated in {elapsed}s")
                    st.markdown(output)


# ======================================================================================
# TAB 3: PODCAST PLANNING (FEATURE 4)
# ======================================================================================
with tab3:
    st.header("3. AI Podcast Planning")
    
    col_pod1, col_pod2 = st.columns([1.2, 2.8])
    with col_pod1:
        st.text_input("Active Topic:", value=active_topic, disabled=True)
        target_aud = st.selectbox(
            "Target Audience:",
            options=["Engineering & IT Students", "General Tech Enthusiasts", "Beginners & Freshers"]
        )
        podcast_btn = st.button("🎙️ Generate Podcast Plan", key="btn_podcast_plan", type="primary")

    podcast_role = "You are an expert tech podcast producer and interviewer."
    podcast_prompt = f"""Create a comprehensive, professional podcast plan about the topic '{active_topic}'.
Target Audience: {target_aud}

Constraints:
- Simple, clear, and professional English
- Exactly 8 numbered, insightful interview questions

Output Format:
Podcast Title:
[Engaging title]

Description:
[2-3 sentence overview]

Guest Type:
[Ideal expert background]

Interview Questions:
1. [Question 1]
2. [Question 2]
3. [Question 3]
4. [Question 4]
5. [Question 5]
6. [Question 6]
7. [Question 7]
8. [Question 8]"""

    with col_pod2:
        with st.expander("🔍 View Structured Prompt Used"):
            st.code(f"ROLE: {podcast_role}\n\nPROMPT:\n{podcast_prompt}", language="markdown")

        if podcast_btn:
            with st.spinner(f"Generating podcast plan with {selected_model}..."):
                messages = [
                    {"role": "system", "content": podcast_role},
                    {"role": "user", "content": podcast_prompt}
                ]
                output, error = call_llm(messages, effective_api_key, model=selected_model)
                if error:
                    st.error(f"⚠️ {error}")
                else:
                    st.success("✅ Podcast Plan Created Successfully!")
                    st.markdown(f'<div class="result-card">{output}</div>', unsafe_allow_html=True)


# ======================================================================================
# TAB 4: TEXT ANALYSIS (FEATURE 5)
# ======================================================================================
with tab4:
    st.header("4. AI & NLP Text Analysis")
    
    default_sample = "Artificial Intelligence is a useful and helpful technology. It makes learning easier and provides students with quick feedback. However, users should verify AI-generated information and avoid depending completely on AI."
    
    user_text = st.text_area("Enter text to analyze:", value=default_sample, height=130)
    analyze_btn = st.button("⚡ Analyze Text", key="btn_analyze_text", type="primary")

    if analyze_btn:
        if not user_text.strip():
            st.error("Please enter text to analyze.")
        else:
            with st.spinner("Performing Sentiment Analysis & Keyword Extraction..."):
                analysis_results = None
                
                if effective_api_key:
                    ai_prompt = f"""Analyze the text below.
Text: \"\"\"{user_text}\"\"\"

Output STRICT JSON only:
{{
    "sentiment": "Positive" | "Negative" | "Neutral",
    "sentiment_score": float between -1.0 and 1.0,
    "reasoning": "short explanation",
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5", "keyword6", "keyword7", "keyword8"]
}}"""
                    messages = [
                        {"role": "system", "content": "You are an NLP engine. Output JSON only."},
                        {"role": "user", "content": ai_prompt}
                    ]
                    output, error = call_llm(messages, effective_api_key, model=selected_model, temperature=0.0)
                    
                    if not error and output:
                        try:
                            cleaned = re.sub(r'^```json\s*|^```\s*|```$', '', output.strip(), flags=re.MULTILINE)
                            parsed = json.loads(cleaned)
                            analysis_results = {
                                "sentiment": parsed.get("sentiment", "Neutral"),
                                "score": parsed.get("sentiment_score", 0.0),
                                "reasoning": parsed.get("reasoning", "Analyzed via LLM."),
                                "keywords": parsed.get("keywords", []),
                                "mode": f"Live AI Mode ({selected_model})"
                            }
                        except Exception:
                            analysis_results = None
                            
                if analysis_results is None:
                    local_sent = local_fallback_sentiment_analysis(user_text)
                    local_kw = local_fallback_keyword_extraction(user_text)
                    analysis_results = {
                        "sentiment": local_sent["sentiment"],
                        "score": local_sent["score"],
                        "reasoning": local_sent["reasoning"],
                        "keywords": local_kw,
                        "mode": "Local Fallback Lexicon"
                    }

                st.markdown(f"### 📊 Analysis Results `[{analysis_results['mode']}]`")
                col_r1, col_r2 = st.columns([1.5, 2.5])
                
                with col_r1:
                    sent = analysis_results["sentiment"]
                    if sent.lower() == "positive":
                        st.success("😊 **Sentiment:** Positive")
                    elif sent.lower() == "negative":
                        st.error("😞 **Sentiment:** Negative")
                    else:
                        st.info("😐 **Sentiment:** Neutral")
                    st.metric("Sentiment Score:", f"{analysis_results['score']:+.2f}")
                    st.caption(f"**Reasoning:** {analysis_results['reasoning']}")
                    
                with col_r2:
                    st.markdown("#### Top Extracted Keywords")
                    tags = "".join([f'<span class="tag-badge">🔑 {k}</span>' for k in analysis_results.get("keywords", [])])
                    st.markdown(tags, unsafe_allow_html=True)


# ======================================================================================
# TAB 5: PARAMETER EXPERIMENTATION (FEATURE 6)
# ======================================================================================
with tab5:
    st.header("5. LLM Parameter Experimentation")
    base_experiment_prompt = "Explain how Artificial Intelligence can help students in education in 80–100 words using simple English."
    
    st.info(f"**Benchmark Prompt:** *\"{base_experiment_prompt}\"*")
    
    run_benchmark = st.button("🔬 Compare Temperature 0.2 vs 0.9", key="btn_run_benchmark", type="primary")

    if run_benchmark:
        col_ea, col_eb = st.columns(2)
        
        with col_ea:
            st.markdown("#### 🔵 Experiment A: Low Temperature (0.2)")
            st.caption("Parameters: `Temperature = 0.2`, `Top-P = 1.0`")
            with st.spinner("Generating Low Temp output..."):
                msgs = [{"role": "user", "content": base_experiment_prompt}]
                out_a, _ = call_llm(msgs, effective_api_key, model=selected_model, temperature=0.2, top_p=1.0)
                st.markdown(f'<div class="experiment-card">{out_a}</div>', unsafe_allow_html=True)
                st.info("🔹 **Behavior:** Focused, predictable, factual, and deterministic.")

        with col_eb:
            st.markdown("#### 🟠 Experiment B: High Temperature (0.9)")
            st.caption("Parameters: `Temperature = 0.9`, `Top-P = 1.0`")
            with st.spinner("Generating High Temp output..."):
                msgs = [{"role": "user", "content": base_experiment_prompt}]
                out_b, _ = call_llm(msgs, effective_api_key, model=selected_model, temperature=0.9, top_p=1.0)
                st.markdown(f'<div class="experiment-card">{out_b}</div>', unsafe_allow_html=True)
                st.info("🔸 **Behavior:** Creative, diverse vocabulary, expressive phrasing.")

        st.markdown("---")
        st.markdown("### 📝 Parameter Theory")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Temperature (0.2):** Sharpens probability distribution ($T < 1.0$), forcing model to choose highest-probability tokens.")
        with c2:
            st.markdown("**Temperature (0.9):** Flattens probability distribution, introducing lexical variety and creative exploratory tokens.")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem;">
    <strong>Walchand Institute of Technology, Solapur</strong> • Department of Information Technology<br>
    Program Elective – V (Prompt Engineering) | Assignment No. 8
</div>
""", unsafe_allow_html=True)
