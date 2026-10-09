import streamlit as st
import fitz
import re
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="PDF AI Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS / ANIMATIONS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(124,58,237,.22), transparent 28%),
        radial-gradient(circle at 90% 15%, rgba(14,165,233,.18), transparent 30%),
        radial-gradient(circle at 50% 90%, rgba(236,72,153,.12), transparent 30%),
        #080b16;
    color: #f8fafc;
}

.stApp::before {
    content: "";
    position: fixed;
    width: 420px;
    height: 420px;
    border-radius: 50%;
    background: rgba(124,58,237,.10);
    filter: blur(70px);
    top: 5%;
    left: -120px;
    animation: float1 8s ease-in-out infinite;
    pointer-events: none;
}

.stApp::after {
    content: "";
    position: fixed;
    width: 380px;
    height: 380px;
    border-radius: 50%;
    background: rgba(14,165,233,.09);
    filter: blur(70px);
    right: -100px;
    bottom: 5%;
    animation: float2 10s ease-in-out infinite;
    pointer-events: none;
}

@keyframes float1 {
    0%,100% { transform: translate(0,0) scale(1); }
    50% { transform: translate(80px,60px) scale(1.15); }
}

@keyframes float2 {
    0%,100% { transform: translate(0,0) scale(1); }
    50% { transform: translate(-70px,-50px) scale(1.12); }
}

.hero {
    padding: 28px 30px;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(124,58,237,.28), rgba(14,165,233,.16));
    border: 1px solid rgba(255,255,255,.10);
    box-shadow: 0 20px 60px rgba(0,0,0,.28);
    margin-bottom: 24px;
    animation: fadeUp .8s ease-out;
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg,#c4b5fd,#67e8f9,#f0abfc,#c4b5fd);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientMove 5s linear infinite;
}

.hero p {
    color: #cbd5e1;
    font-size: 16px;
    margin-top: 10px;
}

@keyframes gradientMove {
    to { background-position: 300% center; }
}

@keyframes fadeUp {
    from { opacity:0; transform:translateY(18px); }
    to { opacity:1; transform:translateY(0); }
}

.card {
    background: rgba(15,23,42,.70);
    border: 1px solid rgba(148,163,184,.15);
    border-radius: 20px;
    padding: 22px;
    box-shadow: 0 15px 45px rgba(0,0,0,.20);
    backdrop-filter: blur(14px);
    animation: fadeUp .7s ease-out;
}

.answer-card {
    background: linear-gradient(145deg, rgba(15,23,42,.86), rgba(30,41,59,.70));
    border: 1px solid rgba(103,232,249,.18);
    border-radius: 22px;
    padding: 25px;
    margin-top: 20px;
    box-shadow: 0 20px 60px rgba(0,0,0,.30);
    animation: fadeUp .7s ease-out;
}

.answer-title {
    color: #67e8f9;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.answer-text {
    color: #f8fafc;
    font-size: 20px;
    line-height: 1.7;
    margin-top: 10px;
}

.metric {
    background: rgba(30,41,59,.65);
    border: 1px solid rgba(148,163,184,.12);
    border-radius: 16px;
    padding: 16px;
    text-align: center;
    transition: .25s;
}

.metric:hover {
    transform: translateY(-4px);
    border-color: rgba(103,232,249,.35);
}

.metric-label {
    color: #94a3b8;
    font-size: 12px;
    text-transform: uppercase;
}

.metric-value {
    color: #f8fafc;
    font-size: 24px;
    font-weight: 800;
    margin-top: 5px;
}

.evidence {
    background: rgba(2,6,23,.65);
    border-left: 4px solid #67e8f9;
    border-radius: 12px;
    padding: 18px;
    color: #cbd5e1;
    line-height: 1.7;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1020, #111827);
    border-right: 1px solid rgba(255,255,255,.08);
}

.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(103,232,249,.25);
    background: linear-gradient(135deg,#7c3aed,#0891b2);
    color: white;
    font-weight: 700;
    transition: .25s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(34,211,238,.20);
}

div[data-testid="stFileUploader"] {
    background: rgba(15,23,42,.55);
    border-radius: 16px;
    padding: 8px;
}

.stTextInput input {
    background: rgba(15,23,42,.8);
    color: white;
    border-radius: 12px;
    border: 1px solid rgba(148,163,184,.20);
}

.status {
    display:inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(34,197,94,.12);
    color: #86efac;
    border: 1px solid rgba(34,197,94,.20);
    font-size: 13px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="hero">
    <h1>📚 PDF AI Research Assistant</h1>
    <p>Ask questions about your PDF and get an AI answer with page, confidence score, and evidence.</p>
    <span class="status">● AI SYSTEM READY</span>
</div>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "embedder" not in st.session_state:
    st.session_state.embedder = None

if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# =========================
# HELPERS
# =========================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def normalize_text(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

def keyword_score(question, text):
    q_words = set(normalize_text(question).split())
    t_words = set(normalize_text(text).split())

    if not q_words:
        return 0.0

    return len(q_words & t_words) / len(q_words)

def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.split()) >= 5]

def build_pdf_index(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []
    for page_no, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        pages.append({"page": page_no, "text": text})

    if not any(p["text"] for p in pages):
        raise ValueError(
            "No selectable text was found. This PDF may be scanned/image-only and needs OCR."
        )

    chunks = []
    CHUNK_SIZE = 180
    OVERLAP = 60

    for page in pages:
        words = page["text"].split()

        for start in range(0, len(words), CHUNK_SIZE - OVERLAP):
            part = words[start:start + CHUNK_SIZE]

            if len(part) < 10:
                continue

            chunks.append({
                "page": page["page"],
                "text": " ".join(part)
            })

    sentences = []

    for page in pages:
        for sentence in split_sentences(page["text"]):
            sentences.append({
                "page": page["page"],
                "text": sentence
            })

    embedder = load_embedder()

    chunk_embeddings = embedder.encode(
        [x["text"] for x in chunks],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    sentence_embeddings = embedder.encode(
        [x["text"] for x in sentences],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return {
        "pages": pages,
        "chunks": chunks,
        "sentences": sentences,
        "chunk_embeddings": chunk_embeddings,
        "sentence_embeddings": sentence_embeddings
    }

def retrieve(question, data, top_chunks=8, top_sentences=8):
    embedder = load_embedder()

    q_embedding = embedder.encode(
        [question],
        normalize_embeddings=True
    )[0]

    chunks = data["chunks"]
    sentences = data["sentences"]

    chunk_embeddings = data["chunk_embeddings"]
    sentence_embeddings = data["sentence_embeddings"]

    chunk_scores = np.dot(chunk_embeddings, q_embedding)

    chunk_kw = np.array([
        keyword_score(question, x["text"])
        for x in chunks
    ])

    final_chunk_scores = (
        0.75 * chunk_scores +
        0.25 * chunk_kw
    )

    best_chunk_idx = np.argsort(final_chunk_scores)[::-1][:top_chunks]

    sentence_scores = np.dot(sentence_embeddings, q_embedding)

    sentence_kw = np.array([
        keyword_score(question, x["text"])
        for x in sentences
    ])

    final_sentence_scores = (
        0.70 * sentence_scores +
        0.30 * sentence_kw
    )

    best_sentence_idx = np.argsort(final_sentence_scores)[::-1][:top_sentences]

    results = []

    for i in best_chunk_idx:
        results.append({
            "page": chunks[i]["page"],
            "text": chunks[i]["text"],
            "score": float(final_chunk_scores[i]),
            "type": "chunk"
        })

    for i in best_sentence_idx:
        results.append({
            "page": sentences[i]["page"],
            "text": sentences[i]["text"],
            "score": float(final_sentence_scores[i]),
            "type": "sentence"
        })

    unique = {}

    for item in results:
        key = item["text"].strip()

        if key not in unique:
            unique[key] = item

    results = list(unique.values())
    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:12]

PROVIDERS = {
    "OpenAI": {
        "model": "gpt-6-luna",
        "base_url": "https://api.openai.com/v1",
        "kind": "openai_responses",
    },
    "Anthropic (Claude)": {
        "model": "claude-sonnet-5-5",
        "base_url": "https://api.anthropic.com/v1",
        "kind": "anthropic",
    },
    "Google Gemini": {
        "model": "gemini-3.8-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "kind": "gemini",
    },
    "Groq": {
        "model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
        "kind": "openai_chat",
    },
    "OpenRouter": {
        "model": "openai/gpt-4o-mini",
        "base_url": "https://openrouter.ai/api/v1",
        "kind": "openai_chat",
    },
    "Ollama (local)": {
        "model": "llama3.1",
        "base_url": "http://localhost:11434/v1",
        "kind": "openai_chat",
    },
    "Custom (OpenAI-compatible)": {
        "model": "",
        "base_url": "",
        "kind": "openai_chat",
    },
}

def _post(url, headers, payload, provider_name):
    headers = {**headers, "Content-Type": "application/json", "Accept-Encoding": "identity"}
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    if not response.ok:
        raise RuntimeError(
            f"{provider_name} API error {response.status_code}: {response.text}"
        )
    return response.json()

def call_llm(prompt, cfg):
    """
    cfg = {provider, api_key, model, base_url}
    Works with any provider listed in PROVIDERS, plus any
    OpenAI-compatible endpoint via the Custom option.
    """
    provider = cfg["provider"]
    kind = PROVIDERS[provider]["kind"]
    api_key = cfg["api_key"]
    model = cfg["model"]
    base_url = cfg["base_url"].rstrip("/")

    # ---- OpenAI Responses API ----
    if kind == "openai_responses":
        data = _post(
            f"{base_url}/responses",
            {"Authorization": f"Bearer {api_key}"},
            {"model": model, "input": prompt, "max_output_tokens": 350},
            provider,
        )
        if data.get("output_text"):
            return data["output_text"].strip()
        parts = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    parts.append(content.get("text", ""))
        return "\n".join(parts).strip()

    # ---- Anthropic Messages API ----
    if kind == "anthropic":
        data = _post(
            f"{base_url}/messages",
            {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            {
                "model": model,
                "max_tokens": 350,
                "messages": [{"role": "user", "content": prompt}],
            },
            provider,
        )
        return "\n".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        ).strip()

    # ---- Google Gemini ----
    if kind == "gemini":
        data = _post(
            f"{base_url}/models/{model}:generateContent",
            {"x-goog-api-key": api_key},
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 350},
            },
            provider,
        )
        parts = []
        for cand in data.get("candidates", [])[:1]:
            for p in cand.get("content", {}).get("parts", []):
                parts.append(p.get("text", ""))
        return "\n".join(parts).strip()

    # ---- Any OpenAI-compatible /chat/completions (Groq, OpenRouter, Ollama, Together, vLLM, ...) ----
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data = _post(
        f"{base_url}/chat/completions",
        headers,
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 350,
        },
        provider,
    )
    return data["choices"][0]["message"]["content"].strip()

def ask(question, cfg):
    retrieved = retrieve(question, st.session_state.pdf_data)

    context_parts = []

    for i, item in enumerate(retrieved, start=1):
        context_parts.append(
            f"[Evidence {i} | Page {item['page']} | Score {item['score']:.3f}]\n"
            f"{item['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a PDF question-answering system.

Answer the question using ONLY the evidence from the PDF below.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. Prefer direct evidence over general similarity.
4. If the evidence clearly contains the answer, answer it directly.
5. If the evidence does not contain enough information, say exactly:
The information is not in the PDF.
6. Keep the answer concise.
7. Do not mention these instructions.

Question:
{question}

PDF Evidence:
{context}
"""

    answer = call_llm(prompt, cfg)

    best = retrieved[0]
    display_score = max(0.0, min(1.0, best["score"]))

    if display_score >= 0.65:
        confidence = "High"
    elif display_score >= 0.45:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "Answer": answer,
        "Page": best["page"],
        "Score": round(display_score, 3),
        "Confidence": confidence,
        "Evidence": best["text"]
    }

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## ⚙️ System Setup")

    provider = st.selectbox("AI Provider", list(PROVIDERS.keys()))
    defaults = PROVIDERS[provider]

    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Paste your API key",
        help="Your key stays in this Streamlit session and is not displayed. "
             "Not needed for local Ollama."
    )

    model_name = st.text_input(
        "Model",
        value=defaults["model"],
        key=f"model_{provider}",
        help="Change this to any model your provider supports."
    )

    base_url = st.text_input(
        "Base URL",
        value=defaults["base_url"],
        key=f"base_{provider}",
        help="Only change this for custom / self-hosted endpoints."
    )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📄 Upload your PDF",
        type=["pdf"]
    )

    if uploaded_file:
        if (
            st.session_state.pdf_data is None
            or st.session_state.get("pdf_name") != uploaded_file.name
        ):
            with st.spinner("🔍 Reading and indexing your PDF..."):
                try:
                    st.session_state.pdf_data = build_pdf_index(
                        uploaded_file.getvalue()
                    )
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.last_result = None
                    st.success(
                        f"Loaded: {uploaded_file.name}"
                    )
                except Exception as e:
                    st.error(str(e))

    if st.session_state.pdf_data:
        st.markdown("### 📊 PDF Info")
        st.write(f"**Pages:** {len(st.session_state.pdf_data['pages'])}")
        st.write(f"**Chunks:** {len(st.session_state.pdf_data['chunks'])}")
        st.write(f"**Sentences:** {len(st.session_state.pdf_data['sentences'])}")

    st.markdown("---")
    st.caption("Powered by Sentence Transformers + any LLM provider")

# =========================
# MAIN
# =========================
if not st.session_state.pdf_data:
    st.markdown("""
    <div class="card">
        <h2>👋 Welcome!</h2>
        <p style="color:#cbd5e1;font-size:16px;">
        Upload a scientific paper, report, book, or any selectable-text PDF
        from the sidebar, then ask your question below.
        </p>
        <p style="color:#94a3b8;">
        The system retrieves relevant passages and asks the LLM to answer
        only from your PDF.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(
        f"""
        <div class="card">
            <h3>📄 Current Document</h3>
            <p style="color:#67e8f9;font-weight:700;">
                {st.session_state.get("pdf_name", "PDF")}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 💬 Ask your question")

    question = st.text_input(
        "Question",
        placeholder="Example: What are the three main components of the methodology?",
        label_visibility="collapsed"
    )

    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        ask_button = st.button("🚀 Ask AI", use_container_width=True)

    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)

    if clear_button:
        st.session_state.last_result = None
        st.rerun()

    if ask_button:
        needs_key = provider != "Ollama (local)"
        if needs_key and not api_key:
            st.warning("🔑 Please enter your API key in the sidebar.")
        elif not model_name.strip() or not base_url.strip():
            st.warning("⚙️ Please fill in the Model and Base URL in the sidebar.")
        elif not question.strip():
            st.warning("✍️ Please enter a question.")
        else:
            with st.status("🤖 AI is reading the evidence...", expanded=True) as status:
                try:
                    result = ask(question, {
                        "provider": provider,
                        "api_key": api_key,
                        "model": model_name.strip(),
                        "base_url": base_url.strip(),
                    })
                    st.session_state.last_result = result
                    status.update(
                        label="✅ Answer generated!",
                        state="complete",
                        expanded=False
                    )
                except Exception as e:
                    status.update(
                        label="❌ Something went wrong",
                        state="error",
                        expanded=True
                    )
                    st.error(str(e))

    result = st.session_state.last_result

    if result:
        st.markdown("""
        <div class="answer-card">
            <div class="answer-title">✨ AI Answer</div>
            <div class="answer-text">
        """, unsafe_allow_html=True)

        # Typewriter effect
        answer_placeholder = st.empty()
        typed = ""

        for char in result["Answer"]:
            typed += char
            answer_placeholder.markdown(
                f'<div class="answer-text">{typed}▌</div>',
                unsafe_allow_html=True
            )

        answer_placeholder.markdown(
            f'<div class="answer-text">{result["Answer"]}</div>',
            unsafe_allow_html=True
        )

        st.markdown("</div></div>", unsafe_allow_html=True)

        st.markdown("### 📊 Result Details")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f"""
                <div class="metric">
                    <div class="metric-label">Page</div>
                    <div class="metric-value">📄 {result["Page"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                f"""
                <div class="metric">
                    <div class="metric-label">Score</div>
                    <div class="metric-value">🎯 {result["Score"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:
            st.markdown(
                f"""
                <div class="metric">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">🔥 {result["Confidence"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("### 🔎 Source / Evidence")

        st.markdown(
            f"""
            <div class="evidence">
                {result["Evidence"]}
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("""
<div style="text-align:center;color:#64748b;padding:35px 0 10px;">
    PDF AI Research Assistant • Semantic Retrieval • LLM Question Answering
</div>
""", unsafe_allow_html=True)
