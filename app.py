"""
========================================================================================
AI-Powered Content Creation and Analysis System
========================================================================================
College: Walchand Institute of Technology, Solapur
Department: Information Technology
Course: Program Elective – V (Prompt Engineering)
Assignment: Assignment No. 8 – Integrated Coding Assignment

Features Implemented:
1. Topic Input (Dynamic topic propagation across all modules)
2. Prompt Design (Role, Context, Constraints, Output Format visualization & execution)
3. Content Generation (Social Media Post, Short Story, Poem)
4. Podcast Planning (Title, Description, Guest Type, 8 Interview Questions)
5. Text Analysis (Sentiment Analysis + Keyword Extraction with AI and Local Fallback)
6. Parameter Experimentation (Side-by-side Temperature 0.2 vs 0.9 and Top-P comparison)
========================================================================================
"""

import os
import re
import json
import time
from collections import Counter
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# ======================================================================================
# PAGE CONFIGURATION & STYLING
# ======================================================================================
st.set_page_config(
    page_title="AI-Powered Content Creation and Analysis System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, student-friendly college project UI
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
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e3a8a;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================================================
# SIDEBAR CONFIGURATION & API SETUP
# ======================================================================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/brain.png", width=90)
    st.title("⚙️ System Config")
    
    st.markdown("### 🔑 API Configuration")
    env_api_key = os.getenv("OPENAI_API_KEY", "")
    user_api_key = st.text_input(
        "OpenAI API Key:",
        value=env_api_key,
        type="password",
        help="Reads automatically from .env or environment variable OPENAI_API_KEY. You can also paste it directly here.",
        placeholder="sk-..."
    )
    
    # Store effective API key in session state
    effective_api_key = user_api_key.strip() if user_api_key else env_api_key.strip()
    
    env_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"]
    default_model_idx = model_options.index(env_model) if env_model in model_options else 0
    
    selected_model = st.selectbox(
        "Model (OPENAI_MODEL):",
        options=model_options,
        index=default_model_idx,
        help="Configurable via OPENAI_MODEL environment variable or dropdown."
    )
    
    # API Status Badge
    if effective_api_key:
        st.success("🟢 OpenAI API Key Detected")
    else:
        st.warning("🟡 No API Key (Local Fallback Active)")
        st.caption("💡 The application will run in demonstration fallback mode with local NLP engines.")
        
    st.divider()
    
    st.markdown("### 🏫 Academic Details")
    st.markdown("""
    **College:** Walchand Institute of Technology, Solapur  
    **Dept:** Information Technology  
    **Course:** PE - V (Prompt Engineering)  
    **Assignment:** No. 8 (Integrated Coding)  
    """)
    
    st.divider()
    with st.expander("ℹ️ How to set API Key"):
        st.markdown("""
        **Windows Command Prompt:**
        ```cmd
        set OPENAI_API_KEY=your_key_here
        ```
        **PowerShell:**
        ```powershell
        $env:OPENAI_API_KEY="your_key_here"
        ```
        **Linux / macOS:**
        ```bash
        export OPENAI_API_KEY="your_key_here"
        ```
        **Using `.env` file:**
        Create a `.env` file in the project folder with:
        `OPENAI_API_KEY=your_key_here`
        """)


# ======================================================================================
# OPENAI CLIENT HELPER WITH ERROR HANDLING
# ======================================================================================
def get_openai_client(api_key: str):
    """Initializes and returns OpenAI client if API key is provided."""
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI Client: {e}")
        return None


def call_openai_llm(messages: list, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.7, top_p: float = 1.0):
    """
    Calls the OpenAI Chat Completions API with proper error handling.
    Returns (response_text, None) on success or (None, error_message) on failure.
    """
    client = get_openai_client(api_key)
    if not client:
        return None, "API Key is missing. Please provide a valid OPENAI_API_KEY."
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=650,
        )
        content = response.choices[0].message.content
        return content, None
    except Exception as e:
        error_msg = str(e)
        if "AuthenticationError" in error_msg or "Incorrect API key" in error_msg:
            return None, "Authentication Error: Invalid OpenAI API key provided."
        elif "RateLimitError" in error_msg or "insufficient_quota" in error_msg:
            return None, "Rate Limit / Quota Exceeded: Your OpenAI account quota is exhausted."
        else:
            return None, f"OpenAI API Error: {error_msg}"


# ======================================================================================
# LOCAL FALLBACK ENGINE (For Text Analysis and Offline Demonstrations)
# ======================================================================================
def local_fallback_sentiment_analysis(text: str):
    """
    Local rule-based lexicon sentiment analyzer as a robust offline fallback.
    """
    positive_words = {
        "good", "great", "excellent", "useful", "helpful", "positive", "beneficial", 
        "easier", "fast", "efficient", "valuable", "effective", "better", "best", 
        "innovative", "accurate", "reliable", "advancement", "progress", "success", 
        "opportunity", "quick", "improve", "superior", "wonderful", "fantastic", "boost"
    }
    negative_words = {
        "bad", "harmful", "useless", "negative", "detrimental", "difficult", "slow", 
        "inefficient", "risk", "danger", "error", "flaw", "fail", "failure", "threat", 
        "vulnerable", "inaccurate", "unreliable", "bias", "biased", "hallucination", 
        "problem", "issue", "worst", "damage", "avoid", "dependency"
    }
    
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
        "positive_count": pos_count,
        "negative_count": neg_count,
        "reasoning": f"Identified {pos_count} positive indicator words and {neg_count} negative indicator words.",
        "mode": "Local Fallback Lexicon"
    }


def local_fallback_keyword_extraction(text: str, top_n: int = 8):
    """
    Local frequency-based keyword extractor filtering common English stopwords.
    """
    stopwords = {
        "the", "and", "is", "in", "to", "of", "it", "a", "an", "that", "this", 
        "for", "on", "with", "as", "by", "at", "from", "be", "are", "was", "were", 
        "or", "not", "but", "what", "all", "were", "we", "when", "your", "can", 
        "said", "there", "use", "an", "each", "which", "she", "do", "how", "their", 
        "if", "will", "up", "other", "about", "out", "many", "then", "them", "these", 
        "so", "some", "her", "would", "make", "like", "him", "into", "time", "has", 
        "look", "two", "more", "go", "see", "no", "way", "could", "my", "than", 
        "first", "been", "call", "who", "its", "now", "find", "long", "down", "day", 
        "did", "get", "come", "made", "may", "part", "should", "makes", "avoid"
    }
    
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered_words = [w for w in words if w not in stopwords]
    counts = Counter(filtered_words)
    return counts.most_common(top_n)


def mock_content_generation(content_type: str, topic: str):
    """Offline mock generator if API key is not supplied."""
    if content_type == "Social Media Post":
        return f"""### Title: 🌟 Transforming the Future with {topic}!

**Main Content:**
Are you exploring how **{topic}** is reshaping modern industries? From automating complex tasks to accelerating academic research, {topic} offers college students unprecedented tools to innovate and solve real-world problems. By mastering these core principles, students can unlock new career frontiers and build impactful solutions. 🚀💡

#Innovation #Technology #Learning #{topic.replace(' ', '')}

**Key Takeaway:**
Embrace {topic} proactively to build practical skills for tomorrow's technology landscape."""

    elif content_type == "Short Story":
        return f"""### Title: The Breakthrough in {topic}

**Main Content:**
In the quiet computing lab of WIT Solapur, Ananya stared at the telemetry data. Her final-year project on **{topic}** had encountered a critical anomaly. Instead of giving up, she reviewed the core constraints and refined the pipeline. As midnight approached, the monitor flashed green: the algorithm converged with 99% accuracy. The lab erupted in quiet celebration as months of dedication turned into a triumphant academic breakthrough.

**Key Takeaway:**
Persistence, disciplined problem-solving, and deep domain knowledge turn technological challenges into success."""

    else: # Poem
        return f"""### Title: The Ode to {topic}

**Main Content:**
Lines of logic, currents bright,  
Guiding minds into the night.  
**{topic}** charts a modern way,  
Turning darkness into day.  

From data streams to insights clear,  
The dawn of innovation is here.  
Empowering youth to reach new heights,  
With boundless dreams and glowing lights.  

**Key Takeaway:**
Technology and human curiosity together create lasting progress."""


def mock_podcast_plan(topic: str):
    """Offline mock podcast planner."""
    return f"""### 🎙️ Podcast Title:
**Decoded: Exploring the Frontiers of {topic}**

---
### 📝 Podcast Description:
An engaging, student-centered episode exploring how **{topic}** is revolutionizing technology, society, and engineering education. We discuss real-world use cases, career opportunities, and ethical considerations with an industry leader.

---
### 👤 Guest Type:
**Senior Research Scientist & Industry Practitioner in {topic}** (e.g., Tech Lead or Applied AI Researcher with 10+ years experience).

---
### ❓ Interview Questions:
1. **Introduction:** What sparked your interest in {topic}, and how has the field evolved over the last decade?
2. **Foundations:** How would you explain the core mechanism of {topic} to an undergraduate engineering student?
3. **Real-world Impact:** What is the most exciting real-world application of {topic} you have witnessed recently?
4. **Student Readiness:** What foundational skills should students in Information Technology build to work in {topic}?
5. **Challenges & Ethics:** What are the major ethical and reliability challenges associated with deploying {topic}?
6. **Tools & Frameworks:** Which open-source tools and libraries do you recommend for beginner projects?
7. **Future Roadmap:** Where do you see {topic} heading in the next 3 to 5 years?
8. **Final Advice:** What is one key piece of advice for college students aspiring to make a difference in this domain?"""


# ======================================================================================
# UI HEADER & TOPIC INPUT (FEATURE 1)
# ======================================================================================
st.markdown("""
<div class="college-header">
    <h1>🤖 AI-Powered Content Creation and Analysis System</h1>
    <p>Integrated Coding Assignment – Prompt Engineering | Department of Information Technology</p>
    <span class="badge-sub">Walchand Institute of Technology, Solapur • Program Elective – V</span>
</div>
""", unsafe_allow_html=True)

# Feature 1: Topic Input at the top
st.markdown("### 📌 Topic Selection")
col_topic, col_quick = st.columns([2.5, 1.5])

with col_topic:
    user_topic = st.text_input(
        "Enter Topic:",
        value="AI in Healthcare",
        help="All AI-generated content across the application will be tailored to this topic.",
        placeholder="e.g., AI in Healthcare, Cybersecurity, Blockchain, Cloud Computing..."
    )

with col_quick:
    preset_topic = st.selectbox(
        "Or Pick a Suggested Topic:",
        options=["(Custom Input)", "AI in Healthcare", "Artificial Intelligence", "Machine Learning", "Cybersecurity", "Cloud Computing", "Blockchain"],
        index=0
    )
    if preset_topic != "(Custom Input)":
        user_topic = preset_topic

# Ensure topic is clean and non-empty
active_topic = user_topic.strip() if user_topic.strip() else "AI in Healthcare"

st.info(f"🎯 **Active Topic for all Modules:** `{active_topic}`")


# ======================================================================================
# APPLICATION TABS (5 CORE ASSIGNMENT FEATURES)
# ======================================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. 📐 Prompt Design",
    "2. ✍️ Content Generation",
    "3. 🎙️ Podcast Planning",
    "4. 🔍 Text Analysis",
    "5. 🧪 Parameter Experiment"
])


# ======================================================================================
# TAB 1: PROMPT DESIGN (FEATURE 2)
# ======================================================================================
with tab1:
    st.header("1. Structured Prompt Design")
    st.markdown("""
    This section demonstrates the **Four Pillars of Structured Prompt Engineering**:
    **Role**, **Context**, **Constraints**, and **Output Format**.
    """)
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.markdown("#### 🛠️ Component Breakdown")
        
        # 1. ROLE
        st.markdown('<div class="prompt-component-box"><strong>🎭 ROLE:</strong><br>You are an expert educational content writer.</div>', unsafe_allow_html=True)
        
        # 2. CONTEXT
        st.markdown(f'<div class="prompt-component-box"><strong>🌐 CONTEXT:</strong><br>Create useful and student-friendly content about <em>{active_topic}</em>.</div>', unsafe_allow_html=True)
        
        # 3. CONSTRAINTS
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
        
        # 4. OUTPUT FORMAT
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

    # Construct the exact prompt string
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
            
        st.caption("✨ This structured prompt is dynamically dispatched to the LLM to verify prompt engineering principles.")

    st.markdown("---")
    st.markdown("#### 🧪 Test This Structured Prompt")
    
    if st.button("🚀 Execute Structured Prompt with LLM", key="btn_test_prompt"):
        with st.spinner(f"Executing structured prompt for '{active_topic}'..."):
            if effective_api_key:
                messages = [
                    {"role": "system", "content": structured_system_prompt},
                    {"role": "user", "content": structured_user_prompt}
                ]
                output, error = call_openai_llm(messages, effective_api_key, model=selected_model)
                if error:
                    st.error(f"⚠️ {error}")
                    st.info("Displaying demonstration sample output:")
                    st.markdown(mock_content_generation("Social Media Post", active_topic))
                else:
                    st.success("✅ Output Generated using Structured Prompt Engineering:")
                    st.markdown(f'<div class="result-card">{output}</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ No OpenAI API key provided. Showing simulation using local engine:")
                st.markdown(mock_content_generation("Social Media Post", active_topic))


# ======================================================================================
# TAB 2: CONTENT GENERATION (FEATURE 3)
# ======================================================================================
with tab2:
    st.header("2. AI Content Generation")
    st.markdown("Generate educational content based on the selected topic using structured prompt constraints.")
    
    col_cg1, col_cg2 = st.columns([1.5, 2.5])
    
    with col_cg1:
        st.markdown("#### ⚙️ Generation Options")
        content_type = st.selectbox(
            "Select Content Type:",
            options=["Social Media Post", "Short Story", "Poem"],
            help="Choose the creative or educational format you want the AI to generate."
        )
        
        st.info(f"📌 **Target Topic:** `{active_topic}`")
        
        # Define specific constraints per content type
        if content_type == "Social Media Post":
            specific_constraint = "- Length: 150–200 words\n- Include relevant emojis and student-friendly hashtags\n- Keep tone professional yet engaging"
        elif content_type == "Short Story":
            specific_constraint = "- Length: 200–250 words\n- Include a student protagonist solving a challenge\n- Clear moral/educational message"
        else: # Poem
            specific_constraint = "- Structure: 3-4 rhyming stanzas\n- Inspiring tone highlighting technical creativity\n- Clear rhythmic cadence"

        gen_button = st.button("✨ Generate Content", key="btn_generate_content", type="primary")

    # Build prompt for Content Generation
    cg_role = "You are a professional AI content writer."
    cg_context = f"Create a high-quality {content_type.lower()} about '{active_topic}' for college students."
    cg_constraints = f"""- Simple English suitable for college students
- Educational, accurate, and avoid unsupported claims
{specific_constraint}"""
    cg_format = """1. Title
2. Content
3. Key Takeaway"""

    cg_full_prompt = f"""{cg_context}

Constraints:
{cg_constraints}

Output Format:
{cg_format}"""

    with col_cg2:
        st.markdown("#### 📄 Generated Output")
        
        with st.expander("🔍 View Structured Prompt Used for this Generation"):
            st.code(f"ROLE: {cg_role}\n\nUSER PROMPT:\n{cg_full_prompt}", language="markdown")

        if gen_button:
            with st.spinner(f"Generating {content_type} for '{active_topic}'..."):
                start_time = time.time()
                if effective_api_key:
                    messages = [
                        {"role": "system", "content": cg_role},
                        {"role": "user", "content": cg_full_prompt}
                    ]
                    output, error = call_openai_llm(messages, effective_api_key, model=selected_model)
                    elapsed = round(time.time() - start_time, 2)
                    
                    if error:
                        st.error(f"⚠️ {error}")
                        st.info("Displaying fallback content generation:")
                        generated_text = mock_content_generation(content_type, active_topic)
                        st.markdown(generated_text)
                    else:
                        st.success(f"✅ {content_type} Generated Successfully in {elapsed}s")
                        st.markdown(output)
                        st.caption(f"Word count: ~{len(output.split())} words")
                else:
                    st.warning("⚠️ OpenAI API Key not configured. Using local demonstration engine:")
                    mock_out = mock_content_generation(content_type, active_topic)
                    st.markdown(mock_out)


# ======================================================================================
# TAB 3: PODCAST PLANNING (FEATURE 4)
# ======================================================================================
with tab3:
    st.header("3. AI Podcast Planning")
    st.markdown("Generate a structured podcast episode plan, including title, description, guest profile, and 8 curated interview questions.")
    
    col_pod_ctrl, col_pod_res = st.columns([1.2, 2.8])
    
    with col_pod_ctrl:
        st.markdown("#### 🎙️ Podcast Settings")
        st.text_input("Active Topic:", value=active_topic, disabled=True)
        
        target_audience = st.selectbox(
            "Target Audience:",
            options=["Engineering & IT Students", "General Tech Enthusiasts", "Beginners & Freshers", "Industry Professionals"]
        )
        
        podcast_btn = st.button("🎙️ Generate Podcast Plan", key="btn_podcast_plan", type="primary")

    # Structured prompt for Podcast Planning
    podcast_role = "You are an expert tech podcast producer and interviewer."
    podcast_user_prompt = f"""Create a comprehensive, professional podcast plan about the topic '{active_topic}'.
Target Audience: {target_audience}

Constraints:
- Simple, clear, and professional English
- The interview questions must be insightful, progressive, and educational
- Exactly 8 numbered interview questions

Output Format:
Podcast Title:
[Engaging title]

Description:
[2-3 sentence overview of the episode]

Guest Type:
[Ideal expert background and qualification]

Interview Questions:
1. [Question 1]
2. [Question 2]
3. [Question 3]
4. [Question 4]
5. [Question 5]
6. [Question 6]
7. [Question 7]
8. [Question 8]"""

    with col_pod_res:
        st.markdown("#### 📋 Podcast Episode Blueprint")
        
        with st.expander("🔍 View Structured Prompt Used for Podcast Planning"):
            st.code(f"ROLE: {podcast_role}\n\nUSER PROMPT:\n{podcast_user_prompt}", language="markdown")

        if podcast_btn:
            with st.spinner(f"Architecting podcast plan for '{active_topic}'..."):
                if effective_api_key:
                    messages = [
                        {"role": "system", "content": podcast_role},
                        {"role": "user", "content": podcast_user_prompt}
                    ]
                    output, error = call_openai_llm(messages, effective_api_key, model=selected_model)
                    
                    if error:
                        st.error(f"⚠️ {error}")
                        st.info("Displaying fallback podcast plan:")
                        st.markdown(mock_podcast_plan(active_topic))
                    else:
                        st.success("✅ Podcast Plan Created Successfully!")
                        st.markdown(f'<div class="result-card">{output}</div>', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ OpenAI API Key not configured. Using local demonstration plan:")
                    st.markdown(mock_podcast_plan(active_topic))


# ======================================================================================
# TAB 4: TEXT ANALYSIS (FEATURE 5)
# ======================================================================================
with tab4:
    st.header("4. AI & NLP Text Analysis")
    st.markdown("Perform **Sentiment Analysis** and **Keyword Extraction** on user-provided text with AI and Local Fallback.")
    
    default_sample_text = "Artificial Intelligence is a useful and helpful technology. It makes learning easier and provides students with quick feedback. However, users should verify AI-generated information and avoid depending completely on AI."
    
    col_sample1, col_sample2 = st.columns([3, 1])
    with col_sample1:
        st.caption("💡 Tip: You can edit the sample text or paste any custom paragraph below.")
    with col_sample2:
        if st.button("🔄 Reset to Default Sample"):
            st.session_state["analysis_input"] = default_sample_text

    user_text = st.text_area(
        "Enter text to analyze:",
        value=st.session_state.get("analysis_input", default_sample_text),
        height=140,
        key="analysis_input_area"
    )
    
    analyze_btn = st.button("⚡ Analyze Text", key="btn_analyze_text", type="primary")

    if analyze_btn:
        if not user_text.strip():
            st.error("Please enter some text before analyzing.")
        else:
            with st.spinner("Analyzing text sentiment and extracting keywords..."):
                analysis_results = None
                mode_used = "Local Fallback"
                
                # Attempt AI-based analysis if API Key is available
                if effective_api_key:
                    ai_analysis_prompt = f"""Analyze the following text and perform Sentiment Analysis and Keyword Extraction.

Text to analyze:
\"\"\"{user_text}\"\"\"

Output MUST be valid JSON strictly adhering to this structure:
{{
    "sentiment": "Positive" | "Negative" | "Neutral",
    "sentiment_score": float between -1.0 and 1.0,
    "confidence_score": float between 0.0 and 1.0,
    "reasoning": "short explanation of sentiment",
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5", "keyword6", "keyword7", "keyword8"]
}}"""
                    messages = [
                        {"role": "system", "content": "You are an expert NLP and sentiment analysis engine. Return JSON only."},
                        {"role": "user", "content": ai_analysis_prompt}
                    ]
                    
                    output, error = call_openai_llm(messages, effective_api_key, model=selected_model, temperature=0.0)
                    
                    if not error and output:
                        try:
                            # Clean possible markdown code fences around JSON
                            cleaned_json = re.sub(r'^```json\s*|^```\s*|```$', '', output.strip(), flags=re.MULTILINE)
                            parsed_data = json.loads(cleaned_json)
                            analysis_results = {
                                "sentiment": parsed_data.get("sentiment", "Neutral"),
                                "score": parsed_data.get("sentiment_score", 0.0),
                                "confidence": parsed_data.get("confidence_score", 0.9),
                                "reasoning": parsed_data.get("reasoning", "Analyzed using OpenAI LLM NLP classifier."),
                                "keywords": parsed_data.get("keywords", []),
                                "mode": "OpenAI AI-Powered Analysis"
                            }
                            mode_used = "AI Mode (OpenAI)"
                        except Exception as json_err:
                            # Fallback if json parsing fails
                            analysis_results = None
                
                # Fallback to local analysis if AI failed or API key absent
                if analysis_results is None:
                    local_sent = local_fallback_sentiment_analysis(user_text)
                    local_kw = local_fallback_keyword_extraction(user_text, top_n=8)
                    analysis_results = {
                        "sentiment": local_sent["sentiment"],
                        "score": local_sent["score"],
                        "confidence": 0.85,
                        "reasoning": local_sent["reasoning"],
                        "keywords": [w for w, _ in local_kw],
                        "keyword_counts": dict(local_kw),
                        "mode": "Local Python Fallback Engine"
                    }
                    mode_used = "Local Fallback Mode"

                # Display Results
                st.markdown("---")
                st.markdown(f"### 📊 Analysis Results `[{analysis_results['mode']}]`")
                
                col_res1, col_res2 = st.columns([1.5, 2.5])
                
                with col_res1:
                    st.markdown("#### 1. Sentiment Classification")
                    sent_val = analysis_results["sentiment"]
                    
                    if sent_val.lower() == "positive":
                        st.success(f"😊 **Sentiment:** Positive")
                    elif sent_val.lower() == "negative":
                        st.error(f"😞 **Sentiment:** Negative")
                    else:
                        st.info(f"😐 **Sentiment:** Neutral")
                        
                    st.metric("Sentiment Polarity Score:", f"{analysis_results['score']:+.2f}", help="Scale: -1.0 (Very Negative) to +1.0 (Very Positive)")
                    st.caption(f"**Reasoning:** {analysis_results['reasoning']}")

                with col_res2:
                    st.markdown("#### 2. Top Extracted Keywords")
                    keywords_list = analysis_results.get("keywords", [])
                    
                    if keywords_list:
                        tags_html = "".join([f'<span class="tag-badge">🔑 {kw}</span>' for kw in keywords_list])
                        st.markdown(tags_html, unsafe_allow_html=True)
                    else:
                        st.write("No keywords extracted.")
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    with st.expander("🔍 View Detailed Analysis Metadata"):
                        st.json(analysis_results)


# ======================================================================================
# TAB 5: PARAMETER EXPERIMENTATION (FEATURE 6)
# ======================================================================================
with tab5:
    st.header("5. LLM Parameter Experimentation")
    st.markdown("""
    Explore how hyperparameters like **Temperature** and **Top-P** govern LLM output generation,
    creativity, determinism, and token distribution.
    """)
    
    # Base prompt as specified in the assignment
    base_experiment_prompt = "Explain how Artificial Intelligence can help students in education in 80–100 words using simple English."
    
    col_opt1, col_opt2 = st.columns([2.5, 1.5])
    with col_opt1:
        custom_prompt_choice = st.radio(
            "Select Experiment Prompt:",
            options=[
                f"Standard Assignment Prompt: \"{base_experiment_prompt}\"",
                f"Active Topic Prompt: \"Explain {active_topic} in 80–100 words using simple English.\""
            ],
            index=0
        )
    
    eval_prompt = base_experiment_prompt if "Standard" in custom_prompt_choice else f"Explain {active_topic} in 80–100 words using simple English."

    st.markdown("---")
    
    # Predefined Comparison Quick-Action
    st.markdown("#### ⚡ Predefined Assignment Benchmark")
    col_pre, col_custom = st.columns([2, 2])
    
    with col_pre:
        run_benchmark_btn = st.button(
            "🔬 Compare Temperature 0.2 vs 0.9",
            key="btn_run_benchmark",
            type="primary",
            help="Directly runs the assignment experiment comparing low vs high temperature."
        )
        
    with col_custom:
        with st.expander("🛠️ Customize Custom Parameters (Optional)"):
            c_temp_a = st.slider("Exp A Temperature:", 0.0, 1.5, 0.2, 0.1)
            c_top_p_a = st.slider("Exp A Top-P:", 0.1, 1.0, 1.0, 0.05)
            c_temp_b = st.slider("Exp B Temperature:", 0.0, 1.5, 0.9, 0.1)
            c_top_p_b = st.slider("Exp B Top-P:", 0.1, 1.0, 1.0, 0.05)
            run_custom_btn = st.button("🧪 Run Custom Comparison", key="btn_run_custom")

    # Determine execution parameters
    execute_exp = False
    temp_a, top_p_a = 0.2, 1.0
    temp_b, top_p_b = 0.9, 1.0
    
    if run_benchmark_btn:
        execute_exp = True
        temp_a, top_p_a = 0.2, 1.0
        temp_b, top_p_b = 0.9, 1.0
    elif 'run_custom_btn' in locals() and run_custom_btn:
        execute_exp = True
        temp_a, top_p_a = c_temp_a, c_top_p_a
        temp_b, top_p_b = c_temp_b, c_top_p_b

    # Render Side-by-side outputs
    if execute_exp:
        st.markdown("### 📊 Side-by-Side Parameter Comparison")
        st.caption(f"**Prompt:** *\"{eval_prompt}\"*")
        
        col_exp_a, col_exp_b = st.columns(2)
        
        # Experiment A
        with col_exp_a:
            st.markdown(f"#### 🔵 Experiment A: Low Temperature ({temp_a})")
            st.caption(f"**Parameters:** `Temperature = {temp_a}`, `Top-P = {top_p_a}`")
            
            with st.spinner("Generating Low Temperature Output..."):
                start_a = time.time()
                if effective_api_key:
                    msgs_a = [{"role": "user", "content": eval_prompt}]
                    out_a, err_a = call_openai_llm(msgs_a, effective_api_key, model=selected_model, temperature=temp_a, top_p=top_p_a)
                    time_a = round(time.time() - start_a, 2)
                    
                    if err_a:
                        out_a = f"Artificial intelligence helps students by personalizing their learning experience. It provides instant feedback on assignments and explains difficult concepts step by step. AI tools can create customized practice quizzes, summarize lengthy textbooks, and assist in language translation. This allows learners to study at their own comfortable pace, improving comprehension and academic performance effectively."
                        st.caption("*(Generated via simulation fallback)*")
                else:
                    out_a = f"Artificial intelligence helps students by personalizing their learning experience. It provides instant feedback on assignments and explains difficult concepts step by step. AI tools can create customized practice quizzes, summarize lengthy textbooks, and assist in language translation. This allows learners to study at their own comfortable pace, improving comprehension and academic performance effectively."
                    time_a = 0.05
                    st.caption("*(Generated via local fallback)*")
                    
                st.markdown(f'<div class="experiment-card">{out_a}</div>', unsafe_allow_html=True)
                st.caption(f"📏 Words: {len(out_a.split())} | ⏱️ Latency: {time_a}s")
                st.info("🔹 **Behavior:** High determinism, focused vocabulary, predictable grammatical structure.")

        # Experiment B
        with col_exp_b:
            st.markdown(f"#### 🟠 Experiment B: High Temperature ({temp_b})")
            st.caption(f"**Parameters:** `Temperature = {temp_b}`, `Top-P = {top_p_b}`")
            
            with st.spinner("Generating High Temperature Output..."):
                start_b = time.time()
                if effective_api_key:
                    msgs_b = [{"role": "user", "content": eval_prompt}]
                    out_b, err_b = call_openai_llm(msgs_b, effective_api_key, model=selected_model, temperature=temp_b, top_p=top_b)
                    time_b = round(time.time() - start_b, 2)
                    
                    if err_b:
                        out_b = f"Imagine having a 24/7 personal tutor right on your desk—that is the power of AI in modern education! AI transforms static learning into an interactive adventure by breaking down complex math and science formulas into digestible metaphors. It crafts smart study schedules, sparks creative brainstorming, and empowers students to master difficult subjects with enthusiasm and newfound confidence."
                        st.caption("*(Generated via simulation fallback)*")
                else:
                    out_b = f"Imagine having a 24/7 personal tutor right on your desk—that is the power of AI in modern education! AI transforms static learning into an interactive adventure by breaking down complex math and science formulas into digestible metaphors. It crafts smart study schedules, sparks creative brainstorming, and empowers students to master difficult subjects with enthusiasm and newfound confidence."
                    time_b = 0.05
                    st.caption("*(Generated via local fallback)*")
                    
                st.markdown(f'<div class="experiment-card">{out_b}</div>', unsafe_allow_html=True)
                st.caption(f"📏 Words: {len(out_b.split())} | ⏱️ Latency: {time_b}s")
                st.info("🔸 **Behavior:** Increased lexical diversity, expressive adjectives, creative narrative framing.")

        # Detailed Observation & Analysis Section
        st.markdown("---")
        st.markdown("### 📝 Observation & Theoretical Analysis")
        
        obs_col1, obs_col2 = st.columns(2)
        with obs_col1:
            st.markdown("""
            #### 🔍 Low Temperature (0.2):
            - **Mathematical Effect:** Sharpens the token probability distribution (Softmax with $T < 1.0$).
            - **Predictability:** Highly predictable, deterministic, and consistent.
            - **Use Cases:** Fact retrieval, code generation, summarization, mathematical explanations, and structured data generation.
            - **Token Selection:** The model strictly picks top-probability candidate tokens.
            """)
        with obs_col2:
            st.markdown("""
            #### 🎨 High Temperature (0.9):
            - **Mathematical Effect:** Flattens the token probability distribution, giving less likely tokens higher chances of selection.
            - **Predictability:** More creative, varied, and expressive.
            - **Use Cases:** Creative storytelling, poetry, brainstorming ideas, conversational chatbots.
            - **Top-P (Nucleus Sampling):** Sets a cumulative probability threshold $p$, restricting candidate tokens to the smallest set whose sum exceeds $p$.
            """)


# ======================================================================================
# FOOTER
# ======================================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem;">
    <strong>Walchand Institute of Technology, Solapur</strong> • Department of Information Technology<br>
    Program Elective – V (Prompt Engineering) | Assignment No. 8 — Integrated Coding Assignment
</div>
""", unsafe_allow_html=True)
