# app.py — Simple RAG (no Google Drive UI, no source references)

import os, io, pickle, base64
import re
import uuid
import time
import json
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image  # ok to keep; not strictly required now
from dotenv import load_dotenv
import streamlit.components.v1 as components
from sklearn.metrics.pairwise import cosine_similarity  # harmless even if unused

# LangChain / RAG deps
from langchain_core.documents import Document
from llama_cloud_services import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import openai

# ================= Setup =================
load_dotenv()
st.set_page_config(page_title="Underwriting Agent", layout="wide")

# Anti-flicker CSS (kept)
st.markdown("""
<style>
.stButton > button[disabled] { opacity: 1 !important; filter: none !important; }
.stButton > button, button, [role="button"] { transition: none !important; }
*:focus { outline: none !important; box-shadow: none !important; }
html, body, [data-testid="stAppViewContainer"] * { transition: none !important; animation: none !important; }

/* MAIN content only (not the sidebar) */
div[data-testid="stAppViewContainer"] .block-container{
  padding-top:54px !important;
  padding-bottom:160px !important;
}
div[data-testid="stAppViewContainer"] .block-container h1{ margin-top:0 !important; }

/* Chat bubbles */
div[data-testid="stAppViewContainer"] .user-bubble {background:#007bff;color:#fff;padding:8px;border-radius:8px;max-width:60%;float:right;margin:4px;}
div[data-testid="stAppViewContainer"] .assistant-bubble {background:#1e1e1e;color:#fff;padding:8px;border-radius:8px;max-width:60%;float:left;margin:4px;}
div[data-testid="stAppViewContainer"] .clearfix::after {content:"";display:table;clear:both;}
</style>
""", unsafe_allow_html=True)

openai.api_key = os.getenv("OPENAI_API_KEY")

# ============== Lightweight, on-demand page rendering (kept but unused) ==============
@st.cache_data(show_spinner=False)
def page_png_b64(pdf_bytes: bytes, page_no: int, dpi: int = 120) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_no - 1)
    pix = page.get_pixmap(dpi=dpi)
    doc.close()
    return base64.b64encode(pix.tobytes("png")).decode("ascii")

# ================= Chat UI helpers =================
def type_bubble(text: str, *, base_delay: float = 0.012, cutoff_chars: int = 2000):
    placeholder = st.empty()
    buf = []
    count = 0
    for ch in text:
        buf.append(ch); count += 1
        placeholder.markdown(
            f"<div class='assistant-bubble clearfix'>{''.join(buf)}</div>",
            unsafe_allow_html=True,
        )
        if count <= cutoff_chars:
            time.sleep(base_delay)
    return placeholder

def _reset_chat():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."},
        {"role": "assistant", "content": "What can I help you with?"}
    ]
    st.session_state.pending_input = None
    st.session_state.waiting_for_response = False
    st.session_state.last_suggestion = None

# ---------- Session state ----------
if "messages" not in st.session_state:
    _reset_chat()
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None
if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False
if "page_texts" not in st.session_state:
    st.session_state.page_texts = {}
if "last_suggestion" not in st.session_state:
    st.session_state.last_suggestion = None
if "next_msg_id" not in st.session_state:
    st.session_state.next_msg_id = 0

def _new_id():
    n = st.session_state.next_msg_id
    st.session_state.next_msg_id += 1
    return f"m{n}"

# ----------- helpers -----------
def _clean_heading(text: str) -> str:
    if not text: return text
    text = re.sub(r"^[#>\-\*\d\.\)\(]+\s*", "", text).strip()
    text = text.strip(" :–—-·•")
    return text

def extract_suggestion(text):
    m = re.search(r"did you mean\s+(.+?)\?", text, flags=re.IGNORECASE)
    if not m: return None
    val = _clean_heading(m.group(1).strip())
    if val.lower() == "suggestion": return None
    return val

def is_clarification(answer: str) -> bool:
    low = answer.lower().strip()
    patterns = [
        "sorry i didnt understand the question",
        "sorry i didn't understand the question",
        "hmm, i am not sure",
        "are you able to rephrase your question",
        "did you mean",
        "can you clarify",
        "could you clarify",
    ]
    return any(p in low for p in patterns)

def guess_suggestion(question: str, docs):
    q_terms = set(w.lower() for w in re.findall(r"[A-Za-z]{3,}", question))
    best_line, best_score = None, 0
    for d in docs or []:
        for ln in d.page_content.splitlines():
            raw = ln.strip()
            if not raw: continue
            if len(raw.split()) > 6: continue
            words = set(w.lower() for w in re.findall(r"[A-Za-z]{3,}", raw))
            score = len(q_terms & words)
            if score > best_score:
                best_score, best_line = score, raw
    return _clean_heading(best_line or "Valuation Summary")

def fix_markdown_tables(text: str) -> str:
    sep_re  = re.compile(r'^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$')
    row_re  = re.compile(r'^\s*\|.+\|\s*$')
    dash_re = re.compile(r'\s*[-–—]{2,}\s*$')
    lines, out = text.splitlines(), []
    n = len(lines)
    for i, line in enumerate(lines):
        is_row = bool(row_re.match(line))
        is_sep = bool(sep_re.match(line))
        prev_is_table = i > 0 and (row_re.match(lines[i-1]) or sep_re.match(lines[i-1]))
        if is_row and not prev_is_table:
            out.append(line.strip())
            if i + 1 >= n or not sep_re.match(lines[i+1]):
                cols = line.count("|") - 1
                out.append("|" + "|".join(["---"] * cols) + "|")
            continue
        if is_sep:
            out.append(line); continue
        if is_row:
            cells = [c.strip() for c in line.strip()[1:-1].split("|")]
            if all(dash_re.fullmatch(c) for c in cells):
                continue
            cells = ["—" if dash_re.fullmatch(c) else c for c in cells]
            out.append("| " + " | ".join(cells) + " |"); continue
        out.append(line)
    return "\n".join(out)

def sanitize_suggestion(answer: str, question: str, docs):
    low = answer.lower().strip()
    if ("sorry i didnt understand the question" not in low
        and "sorry i didn't understand the question" not in low
        and "did you mean" not in low):
        return answer
    sug = _clean_heading(guess_suggestion(question, docs))
    answer = re.sub(r"\bSUGGESTION\b", sug, answer, flags=re.IGNORECASE)
    answer = re.sub(r"\[.*?\]", sug, answer)
    m = re.search(r"did you mean\s+(.+?)\?", answer, flags=re.IGNORECASE)
    cand = _clean_heading(m.group(1).strip()) if m else ""
    if not cand or cand.lower() == "suggestion":
        return f"Sorry I didnt understand the question. Did you mean {sug}?"
    if len(cand.split()) > 6:
        cand = " ".join(cand.split()[:6])
        return f"Sorry I didnt understand the question. Did you mean {cand}?"
    return re.sub(r"(Did you mean\s+)[#>\-\*\d\.\)\(]+\s*", r"\1", answer, flags=re.IGNORECASE)

def _last_assistant_text(history: list) -> str:
    for m in reversed(history):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""

def classify_reply_intent(user_reply: str, prev_assistant: str) -> str:
    judge = PromptTemplate(
        template=(
            "You are a strict intent classifier.\n"
            "Assistant just said:\n{assistant}\n\n"
            "User replied:\n{user}\n\n"
            "Label the user's intent relative to the assistant's message as exactly one of:\n"
            "CONFIRM  (agree/accept/yes/affirm)\n"
            "DENY     (disagree/no/reject)\n"
            "NEITHER  (anything else: new question, unrelated, unclear)\n\n"
            "Reply with ONLY one token: CONFIRM or DENY or NEITHER."
        ),
        input_variables=["assistant", "user"]
    )
    out = ChatOpenAI(model="gpt-4o", temperature=0).invoke(
        judge.invoke({"assistant": prev_assistant or "", "user": user_reply or ""})
    ).content.strip().upper()
    if out not in {"CONFIRM", "DENY", "NEITHER"}:
        return "NEITHER"
    return out

def condense_query(chat_history, user_input: str, pdf_name: str) -> str:
    hist = []
    for m in chat_history[-6:]:
        speaker = "User" if m["role"] == "user" else "Assistant"
        hist.append(f"{speaker}: {m['content']}")
    hist_txt = "\n".join(hist)
    prompt = PromptTemplate(
        template=(
            'Turn the last user message into a single, self-contained search query about the PDF "{pdf_name}". '
            "Use ONLY info implied by the history; keep it short and noun-heavy. "
            "Return just the query.\n"
            "---\nHistory:\n{history}\n---\nLast user message: {question}\nStandalone query:"
        ),
        input_variables=["pdf_name", "history", "question"]
    )
    return ChatOpenAI(model="gpt-4o", temperature=0).invoke(
        prompt.invoke({"pdf_name": pdf_name, "history": hist_txt, "question": user_input})
    ).content.strip() or user_input

# ================= Builder =================
@st.cache_resource(show_spinner="📦 Processing & indexing PDF…")
def build_retriever_from_pdf(pdf_bytes: bytes, file_name: str):
    os.makedirs("uploaded", exist_ok=True)
    pdf_path = os.path.join("uploaded", file_name)
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    # Optional: get page count (not used later)
    doc = fitz.open(pdf_path)
    _ = doc.page_count
    doc.close()

    parser = LlamaParse(api_key=os.environ["LLAMA_CLOUD_API_KEY"], num_workers=2)
    result = parser.parse(pdf_path)

    pages = []
    page_texts = {}
    for pg in result.pages:
        cleaned = [l for l in pg.md.splitlines() if l.strip() and l.lower() != "null"]
        text = "\n".join(cleaned)
        if text:
            pages.append(Document(page_content=text, metadata={"page_number": pg.page}))
            page_texts[pg.page] = text

    splitter = RecursiveCharacterTextSplitter(chunk_size=3300, chunk_overlap=0)
    chunks = splitter.split_documents(pages)
    for idx, c in enumerate(chunks):
        c.metadata["chunk_id"] = idx + 1

    embedder = _embedder()
    vs = FAISS.from_documents(chunks, embedder)

    store = os.path.join("vectorstore", file_name)
    os.makedirs(store, exist_ok=True)
    vs.save_local(store, index_name="faiss")
    with open(os.path.join(store, "metadata.pkl"), "wb") as mf:
        pickle.dump([c.metadata for c in chunks], mf)

    reranker = CohereRerank(
        model="rerank-english-v3.0",
        user_agent="langchain",
        cohere_api_key=os.environ["COHERE_API_KEY"],
        top_n=20
    )
    retriever = ContextualCompressionRetriever(
        base_retriever=vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 50, "fetch_k": 100, "lambda_mult": 0.9}
        ),
        base_compressor=reranker
    )
    return retriever, page_texts

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

# One embedder instance shared across reruns & code paths
@st.cache_resource(show_spinner=False)
def _embedder():
    return CohereEmbeddings(
        model="embed-english-v3.0",
        user_agent="langchain",
        cohere_api_key=os.environ["COHERE_API_KEY"],
    )

# ================= Main UI =================
st.title("Underwriting Agent")

up = st.file_uploader("Upload a valuation report PDF", type="pdf")
if up and up.name != st.session_state.get("last_selected_upload"):
    st.session_state.last_selected_upload = up.name
    _reset_chat()

if not up:
    st.warning("Please upload a PDF to continue.")
    st.stop()

# Rebuild retriever when file changes
if st.session_state.get("last_processed_pdf") != up.name:
    pdf_bytes = up.getvalue()
    st.session_state.pdf_bytes = pdf_bytes
    st.session_state.pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    st.session_state.retriever, st.session_state.page_texts = build_retriever_from_pdf(pdf_bytes, up.name)
    _reset_chat()
    st.session_state.last_processed_pdf = up.name

# Chat input
user_q = st.chat_input("Type your question here…", key="main_chat_input")
if user_q:
    st.session_state.pending_input = user_q
    st.session_state.waiting_for_response = True
    st.session_state.messages.append({"id": _new_id(), "role": "user", "content": user_q})

# ===== Render chat history (no source refs) =====
if hasattr(st, "fragment"):
    @st.fragment
    def render_history(msgs):
        for msg in msgs:
            cls = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
            st.markdown(f"<div class='{cls} clearfix'>{msg['content']}</div>", unsafe_allow_html=True)
    render_history(st.session_state.messages)
else:
    for msg in st.session_state.messages:
        cls = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
        st.markdown(f"<div class='{cls} clearfix'>{msg['content']}</div>", unsafe_allow_html=True)

# ========================== ANSWER ==========================
if st.session_state.waiting_for_response and st.session_state.pending_input:
    block = st.empty()
    with block.container():
        thinking = st.empty()
        thinking.markdown("<div class='assistant-bubble clearfix'>Thinking…</div>", unsafe_allow_html=True)

        raw_q = st.session_state.pending_input or ""
        history_to_use = st.session_state.messages[-10:]
        pdf_display = os.path.splitext(up.name)[0]

        # Simple intent handling kept (optional)
        intent = classify_reply_intent(raw_q, _last_assistant_text(history_to_use))
        is_deny = (intent == "DENY")
        is_confirm = (intent == "CONFIRM")

        if is_deny:
            st.session_state.last_suggestion = None
            answer = "Alright, if you have any more questions or need further assistance, feel free to ask!"
        else:
            effective_q = st.session_state.last_suggestion if (is_confirm and st.session_state.last_suggestion) else raw_q
            query_for_retrieval = condense_query(history_to_use, effective_q, pdf_display)

            ctx, docs = "", []
            try:
                docs = st.session_state.retriever.get_relevant_documents(query_for_retrieval)
                ctx = "\n\n".join(d.page_content for d in docs)
            except Exception as e:
                st.warning(f"RAG retrieval error: {e}")

            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            full_input = {
                "chat_history": "\n".join(
                    f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                    for m in history_to_use
                ),
                "context":      ctx,
                "question":     effective_q,
                "pdf_name":     pdf_display,
            }
            answer = llm.invoke(
                PromptTemplate(
                    template = """
You are a financial-data extraction assistant.

**HARD RULE (unrelated questions)**
 • If the user's question is unrelated to this PDF or requires information outside the Context, reply **exactly**:
   "Sorry I can only answer question related to {pdf_name} pdf document" 
 
**Use ONLY what appears under “Context”**.

### How to answer
1. **Single value questions**: Short, clear sentence with the exact number from context.
2. **Table questions**: Return a valid GitHub-flavored table (with header separator).
3. **Valuation/weights**: If applicable, combine info across chunks and mention weights and $ values.

4. If you cannot find an answer in Context → reply exactly:
   "Sorry I didnt understand the question. Did you mean SUGGESTION?"

---
Context:
{context}
---
Question: {question}
Answer:
Conversation so far:
{chat_history}
""",
                    input_variables=["chat_history", "context", "question", "pdf_name"]
                ).invoke(full_input)
            ).content
            answer = fix_markdown_tables(answer)
            answer = sanitize_suggestion(answer, effective_q, docs)

            # store suggestion (if any)
            sug = extract_suggestion(answer)
            st.session_state.last_suggestion = _clean_heading(sug) if sug else None

        thinking.empty()
        type_bubble(answer)

        st.session_state.messages.append({"id": _new_id(), "role": "assistant", "content": answer})
        st.session_state.pending_input = None
        st.session_state.waiting_for_response = False

# Keepalive (optional, harmless)
components.html(
    """
    <script>
      setInterval(function () {
        fetch(window.location.pathname, {method:'GET', cache:'no-store'}).catch(()=>{});
      }, 60000);
    </script>
    """,
    height=0,
)
