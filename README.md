# 🤖 AI-Powered Content Creation and Analysis System

**Walchand Institute of Technology, Solapur**  
**Department of Information Technology**  
**Course:** Program Elective – V (Prompt Engineering)  
**Assignment No. 8:** Integrated Coding Assignment  

---

## 📌 Project Overview

The **AI-Powered Content Creation and Analysis System** is a full-stack Python and Streamlit web application designed to demonstrate the core principles of modern **Prompt Engineering**, **LLM orchestration**, **NLP text analysis**, and **hyperparameter tuning (Temperature & Top-P)**.

The application allows students, educators, and researchers to enter any topic and seamlessly generate structured content, plan podcasts, perform multi-engine sentiment and keyword text analysis, and empirically evaluate the effect of sampling parameters on Large Language Models.

---

## 🎯 Features

1. **Dynamic Topic Selection**:
   - Universal topic input (`Enter Topic`) at the top of the interface (Default: `AI in Healthcare`).
   - Quick-select presets for common engineering domains (e.g., *Cybersecurity, Cloud Computing, Blockchain, Machine Learning*).
   - Topic propagates across all application tabs.

2. **Structured Prompt Design (Feature 2)**:
   - Visibly breaks down prompt engineering into the **4 Fundamental Pillars**:
     - 🎭 **ROLE**: `"You are an expert educational content writer."`
     - 🌐 **CONTEXT**: `"Create useful and student-friendly content about {topic}."`
     - ⚙️ **CONSTRAINTS**: Simple English, factual accuracy, college-friendly tone, conciseness.
     - 📋 **OUTPUT FORMAT**: Title, Main Content, Key Takeaway.
   - Interactive preview of the exact assembled prompt.
   - Direct execution button to test the structured prompt against the LLM.

3. **AI Content Generation (Feature 3)**:
   - Dynamic dropdown to select content format:
     - 📱 **Social Media Post** (150–200 words, hooks, hashtags, takeaway)
     - 📖 **Short Story** (Narrative scenario, student challenge, moral/learning)
     - 🎨 **Poem** (Structured rhyming stanzas capturing core themes)
   - Real-time LLM generation using prompt engineering principles.
   - Expandable "View Structured Prompt Used" for verification.

4. **AI Podcast Planning (Feature 4)**:
   - Generates complete podcast episode blueprints tailored to the topic:
     1. **Podcast Title**
     2. **Podcast Description**
     3. **Guest Type Profile**
     4. **8 Curated Interview Questions**
   - Clean card-based visual layout.

5. **Text Analysis Engine (Feature 5)**:
   - Sentiment Analysis (*Positive / Negative / Neutral* with numerical polarity score and confidence).
   - Top 5–10 Keyword Extraction.
   - **Dual-Engine Architecture**:
     - 🟢 **AI-Powered Engine (OpenAI API)** for contextual semantic analysis and structured JSON parsing.
     - 🟡 **Local Fallback Engine (Rule-based lexicon & Stopword NLP)** ensuring the app works smoothly offline or without an API key.

6. **Parameter Experimentation (Feature 6)**:
   - Visual side-by-side comparison of LLM outputs using the exact same prompt:
     - **Experiment A**: Low Temperature (`0.2`), Top-P (`1.0`)
     - **Experiment B**: High Temperature (`0.9`), Top-P (`1.0`)
   - One-click benchmark comparison button: `"🔬 Compare Temperature 0.2 vs 0.9"`.
   - Interactive sliders for custom Temperature (`0.0 - 1.5`) and Top-P (`0.1 - 1.0`).
   - Theoretical explanations of token probability distributions and nucleus sampling.

---

## 🗂️ Project Structure

```text
ai_content_analysis_app/
│
├── app.py                  # Main Streamlit application with all 5 features & UI
├── requirements.txt        # Python package dependencies
├── README.md               # Detailed project & assignment documentation
├── .env.example            # Environment variable template
└── screenshots/
    └── README.txt          # Guide for saving submission screenshots
```

---

## 🛠️ Technologies Used

- **Language:** Python 3.10+ (Tested on Python 3.13)
- **Web Framework:** Streamlit
- **LLM Provider:** OpenAI API (`gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`)
- **Environment Management:** `python-dotenv`
- **NLP / Text Processing:** Python Regular Expressions, `collections.Counter`, and custom Rule-based Lexicon

---

## 📋 Assignment Mapping Table

| Assignment Requirement | Application Implementation | Location in App |
| :--- | :--- | :--- |
| **Topic Input** | Top-level interactive text input with default `AI in Healthcare` and quick presets | Header section |
| **Prompt Design** | Displays Role, Context, Constraints, Output Format + Assembled prompt + LLM Execution | **Tab 1:** `Prompt Design` |
| **Content Generation** | Generates Social Media Post, Short Story, or Poem using structured constraints | **Tab 2:** `Content Generation` |
| **Podcast Planning** | Generates Title, Description, Guest Type, and 8 numbered interview questions | **Tab 3:** `Podcast Planning` |
| **Text Analysis** | Sentiment classification + polarity score + top 8 keywords (AI + Local Fallback) | **Tab 4:** `Text Analysis` |
| **Parameter Experimentation** | Side-by-side comparison of Temperature 0.2 vs 0.9 with Top-P control & theory | **Tab 5:** `Parameter Experiment` |

---

## 🚀 Installation & Setup Guide

### Step 1: Clone or Navigate to the Project Folder
```bash
cd "f:\Assignment 8 prompt"
```

### Step 2: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure OpenAI API Key

You can configure your API key in **any** of the following ways:

#### Option A: Using `.env` File (Recommended)
Copy `.env.example` to `.env` and insert your OpenAI key:
```ini
OPENAI_API_KEY=sk-proj-your-actual-api-key-here
OPENAI_MODEL=gpt-4o-mini
```

#### Option B: Windows Command Prompt (CMD)
```cmd
set OPENAI_API_KEY=your_key_here
```

#### Option C: Windows PowerShell
```powershell
$env:OPENAI_API_KEY="your_key_here"
```

#### Option D: Linux / macOS Terminal
```bash
export OPENAI_API_KEY="your_key_here"
```

#### Option E: In-App Sidebar
You can also paste the API key directly into the sidebar text field inside the running Streamlit web application.

> **Note:** If no API key is provided, the application automatically switches to **Local Demonstration Fallback Mode** so all UI features and text analysis algorithms can still be evaluated offline.

---

## 🏃 How to Run the Application

Execute the following command in your terminal:

```bash
streamlit run app.py
```

The application will automatically open in your default browser at:
`http://localhost:8501`

---

## 🔬 Parameter Experimentation Explained

### 1. Temperature ($T$)
Temperature controls the randomness/entropy of the model's token sampling:
$$\text{Softmax}(z_i, T) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

- **Low Temperature ($T = 0.2$):** Sharpens the probability distribution. The model is conservative, focused, deterministic, and ideal for facts, code, and structured data.
- **High Temperature ($T = 0.9$):** Flattens the probability distribution, giving less probable words a higher chance to be selected. The output becomes more creative, colorful, and diverse.

### 2. Top-P (Nucleus Sampling)
- Restricts the token pool to the smallest subset whose cumulative probability exceeds the threshold $p$.
- Setting $\text{Top-P} = 1.0$ considers all candidate tokens, while $\text{Top-P} = 0.5$ limits choices strictly to the most probable 50% mass.

---

## 🧪 Testing Checklist

- [x] Application launches cleanly without errors (`streamlit run app.py`).
- [x] Topic Input dynamically updates across all modules.
- [x] Prompt Design tab visibly shows Role, Context, Constraints, and Output Format.
- [x] Content Generation produces Social Media Posts, Stories, and Poems.
- [x] Podcast Planning generates title, description, guest profile, and 8 questions.
- [x] Text Analysis outputs Sentiment label, polarity score, and keyword tags.
- [x] Local Fallback ensures uninterrupted operation when API key is missing.
- [x] Parameter Experiment compares Temperature 0.2 vs 0.9 side-by-side on the exact same prompt.
- [x] No unhandled Python traceback exceptions or exposed secrets.
