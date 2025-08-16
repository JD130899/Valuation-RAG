import os, io, pickle, base64
import re
import uuid
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv
import streamlit.components.v1 as components
from urllib.parse import quote
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
from gdrive_utils import get_drive_service, get_all_pdfs, download_pdf

# ================= Setup =================
load_dotenv()
st.set_page_config(page_title="Underwriting Agent", layout="wide")
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------- Session state ----------
if "last_synced_file_id" not in st.session_state:
    st.session_state.last_synced_file_id = None
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."},
        {"role": "assistant", "content": "What can I help you with?"}
    ]
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None
if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False
if "page_images" not in st.session_state:
    st.session_state.page_images = {}
if "last_suggestion" not in st.session_state:
    st.session_state.last_suggestion = None  # remember last “Did you mean X?”
if "next_msg_id" not in st.session_state:
    st.session_state.next_msg_id = 0

# ---------- styles (full-width, floating pill row above chat input) ----------
st.markdown("""
<style>
.block-container { padding-bottom: 140px; }

/* same width/position as chat input, but content aligned to the RIGHT */
.qs-row{
  position: fixed;
  bottom: 110px;                 /* just above input */
  left: 81%;
  transform: translateX(-50%);  /* center the container itself */
  width: 100%;
  max-width: 720px;             /* match your chat input max width */
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: flex-end;    /* <-- right-align within the bar */
  padding: 0 12px;
  box-sizing: border-box;
  z-index: 9999;
}

/* pills (unchanged) */
.qs-pill{
  text-decoration:none!important;
  border-radius:999px;
  padding:8px 14px;
  border:1px solid rgba(255,255,255,0.2);
  background:black;
  color:white!important;
  font-size:0.9rem;
}
.qs-pill:hover{ background:#222; }

/* optional: on very small screens, left-align to avoid overflow */
@media (max-width: 760px){
  .qs-row{ justify-content:flex-start; }
}
</style>
""", unsafe_allow_html=True)

buttons = [
    "ETRAN Cheatsheet",
    "What is the valuation?",
    "Goodwill value",
]
links = "".join(
    f'<a class="qs-pill" href="?suggest={quote(lbl)}">{lbl}</a>'
    for lbl in buttons
)
st.markdown(f'<div class="qs-row">{links}</div>', unsafe_allow_html=True)

def _new_id():
    n = st.session_state.next_msg_id
    st.session_state.next_msg_id += 1
    return f"m{n}"

def file_badge_link(name: str, pdf_bytes: bytes, synced: bool = True):
    base = os.path.splitext(name)[0]
    b64 = base64.b64encode(pdf_bytes).decode("ascii")
    label = "Using synced file:" if synced else "Using file:"
    link_id = f"open-file-{uuid.uuid4().hex[:8]}"
    st.markdown(
        f'''
        <div style="background:#1f2c3a; padding:8px; border-radius:8px; color:#fff;">
          ✅ <b>{label}</b>
          <a id="{link_id}" href="#" target="_blank" rel="noopener" style="color:#93c5fd; text-decoration:none;">{base}</a>
        </div>
        ''',
        unsafe_allow_html=True
    )
    components.html(
        f'''<!doctype html><meta charset='utf-8'>
<style>html,body{{background:transparent;margin:0;height:0;overflow:hidden}}</style>
<script>(function(){{
  function b64ToUint8Array(s){{var b=atob(s),u=new Uint8Array(b.length);for(var i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u;}}
  var blob = new Blob([b64ToUint8Array("{b64}")], {{type:"application/pdf"}});
  var url  = URL.createObjectURL(blob);
  function attach(){{
    var d = window.parent && window.parent.document;
    if(!d) return setTimeout(attach,120);
    var a = d.getElementById("{link_id}");
    if(!a) return setTimeout(attach,120);
    a.setAttribute("href", url);
  }}
  attach();
  var me = window.frameElement; if(me){{me.style.display="none";me.style.height="0";me.style.border="0";}}
}})();</script>''',
        height=0,
    )

def render_reference_card(label: str, img_b64: str, pdf_b64: str, page: int, key: str):
    st.markdown(
        f"""
        <details class="ref" id="ref-{key}">
          <summary>📘 {label or "Reference"}</summary>
          <button class="overlay" id="overlay-{key}" type="button" aria-label="Close"></button>
          <div class="panel">
            <button class="close-x" id="close-{key}" type="button" aria-label="Close">×</button>
            <img src="data:image/png;base64,{img_b64}" alt="reference" loading="lazy"/>
            <div style="margin-top:8px; text-align:right;">
              <a id="open-{key}" href="#" target="_blank" rel="noopener">Open this page ↗</a>
            </div>
          </div>
        </details>
        <div class="clearfix"></div>
        """,
        unsafe_allow_html=True,
    )
    components.html(
        f"""<!doctype html><meta charset='utf-8'>
<style>html,body{{background:transparent;margin:0;height:0;overflow:hidden}}</style>
<script>(function(){{
  function b64ToUint8Array(s){{var b=atob(s),u=new Uint8Array(b.length);for(var i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u;}}
  var blob = new Blob([b64ToUint8Array('{pdf_b64}')], {{type:'application/pdf'}});
  var url  = URL.createObjectURL(blob) + '#page={page}';
  function attach(){{
    var d = window.parent && window.parent.document;
    if(!d) return setTimeout(attach,120);
    var ref = d.getElementById('ref-{key}');
    var a   = d.getElementById('open-{key}');
    var ovl = d.getElementById('overlay-{key}');
    var cls = d.getElementById('close-{key}');
    if(!ref || !a || !ovl || !cls) return setTimeout(attach,120);
    a.setAttribute('href', url);
    function closeRef(){{ ref.removeAttribute('open'); }}
    ovl.addEventListener('click', closeRef);
    cls.addEventListener('click', closeRef);
    d.addEventListener('keydown', function(e){{ if(e.key==='Escape') closeRef(); }});
  }}
  attach();
  var me = window.frameElement; if(me){{me.style.display='none';me.style.height='0';me.style.border='0';}}
}})();</script>""",
        height=0,
    )

# give IDs to any preloaded messages
for m in st.session_state.messages:
    if "id" not in m:
        m["id"] = _new_id()

# ----------- helpers (cleaning + suggestion + confirmation routing) -----------

def _clean_heading(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"^[#>\-\*\d\.\)\(]+\s*", "", text).strip()
    text = text.strip(" :–—-·•")
    return text

def extract_suggestion(text):
    m = re.search(r"did you mean\s+(.+?)\?", text, flags=re.IGNORECASE)
    if not m:
        return None
    val = _clean_heading(m.group(1).strip())
    if val.lower() == "suggestion":
        return None
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
            if not raw:
                continue
            if len(raw.split()) > 6:
                continue
            words = set(w.lower() for w in re.findall(r"[A-Za-z]{3,}", raw))
            score = len(q_terms & words)
            if score > best_score:
                best_score, best_line = score, raw
    return _clean_heading(best_line or "Valuation Summary")

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
    """
    Returns: CONFIRM | DENY | NEITHER
    """
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

    # Page images
    doc = fitz.open(pdf_path)
    page_images = {
        i + 1: Image.open(io.BytesIO(page.get_pixmap(dpi=180).tobytes("png")))
        for i, page in enumerate(doc)
    }
    doc.close()

    # Parse
    parser = LlamaParse(api_key=os.environ["LLAMA_CLOUD_API_KEY"], num_workers=4)
    result = parser.parse(pdf_path)
    pages = []
    for pg in result.pages:
        cleaned = [l for l in pg.md.splitlines() if l.strip() and l.lower() != "null"]
        text = "\n".join(cleaned)
        if text:
            pages.append(Document(page_content=text, metadata={"page_number": pg.page}))

    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=3300, chunk_overlap=0)
    chunks = splitter.split_documents(pages)
    for idx, c in enumerate(chunks):
        c.metadata["chunk_id"] = idx + 1

    # Embed + FAISS
    embedder = CohereEmbeddings(
        model="embed-english-v3.0",
        user_agent="langchain",
        cohere_api_key=st.secrets["COHERE_API_KEY"]
    )
    vs = FAISS.from_documents(chunks, embedder)

    # Persist (optional)
    store = os.path.join("vectorstore", file_name)
    os.makedirs(store, exist_ok=True)
    vs.save_local(store, index_name="faiss")
    with open(os.path.join(store, "metadata.pkl"), "wb") as mf:
        pickle.dump([c.metadata for c in chunks], mf)

    # Retriever + reranker
    reranker = CohereRerank(
        model="rerank-english-v3.0",
        user_agent="langchain",
        cohere_api_key=st.secrets["COHERE_API_KEY"],
        top_n=20
    )
    retriever = ContextualCompressionRetriever(
        base_retriever=vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 50, "fetch_k": 100, "lambda_mult": 0.9}
        ),
        base_compressor=reranker
    )
    return retriever, page_images

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

# ================= Sidebar: Google Drive loader =================
service = get_drive_service()
pdf_files = get_all_pdfs(service)
if pdf_files:
    names = [f["name"] for f in pdf_files]
    sel = st.sidebar.selectbox("📂 Select a PDF from Google Drive", names)
    chosen = next(f for f in pdf_files if f["name"] == sel)
    if st.sidebar.button("📥 Load Selected PDF"):
        fid, fname = chosen["id"], chosen["name"]
        if fid == st.session_state.last_synced_file_id:
            st.sidebar.info("✅ Already loaded.")
        else:
            path = download_pdf(service, fid, fname)
            if path:
                st.session_state.uploaded_file_from_drive = open(path, "rb").read()
                st.session_state.uploaded_file_name = fname
                st.session_state.last_synced_file_id = fid
                st.session_state.messages = [
                    {"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."},
                    {"role": "assistant", "content": "What can I help you with?"}
                ]
else:
    st.sidebar.warning("📭 No PDFs found in Drive.")

# ================= Main UI =================
st.title("Underwriting Agent")

if "uploaded_file_from_drive" in st.session_state:
    file_badge_link(
        st.session_state.uploaded_file_name,
        st.session_state.uploaded_file_from_drive,
        synced=True
    )
    up = io.BytesIO(st.session_state.uploaded_file_from_drive)
    up.name = st.session_state.uploaded_file_name
else:
    up = st.file_uploader("Upload a valuation report PDF", type="pdf")
    if up:
        file_badge_link(up.name, up.getvalue(), synced=False)

if not up:
    st.warning("Please upload or load a PDF to continue.")
    st.stop()

# Rebuild retriever when file changes
if st.session_state.get("last_processed_pdf") != up.name:
    pdf_bytes = up.getvalue()
    st.session_state.pdf_bytes = pdf_bytes
    st.session_state.retriever, st.session_state.page_images = build_retriever_from_pdf(pdf_bytes, up.name)
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."},
        {"role": "assistant", "content": "What can I help you with?"}
    ]
    st.session_state.last_processed_pdf = up.name

# ================= Styles =================
st.markdown("""
<style>
/* --- Chat bubbles --- */
.user-bubble {background:#007bff;color:#fff;padding:8px;border-radius:8px;max-width:60%;float:right;margin:4px;}
.assistant-bubble {background:#1e1e1e;color:#fff;padding:8px;border-radius:8px;max-width:60%;float:left;margin:4px;}
.clearfix::after {content:"";display:table;clear:both;}

/* --- Reference card --- */
.ref{ display:block; width:60%; max-width:900px; margin:6px 0 12px 8px; }
.ref summary{
  display:inline-flex; align-items:center; gap:8px; cursor:pointer; list-style:none; outline:none;
  background:#0f172a; color:#e2e8f0; border:1px solid #334155; border-radius:10px; padding:6px 10px;
}
.ref summary::before{ content:"▶"; font-size:12px; line-height:1; }
.ref[open] summary::before{ content:"▼"; }
.ref .panel{
  background:#0f172a; color:#e2e8f0; border:1px solid #334155; border-top:none;
  border-radius:10px; padding:10px; margin-top:0; box-shadow:0 6px 20px rgba(0,0,0,.25);
}
.ref .panel img{ width:100%; height:auto; border-radius:8px; display:block; }
.ref .overlay{ display:none; }
.ref[open] .overlay{ display:block; position:fixed; inset:0; z-index:998; background:transparent; border:0; padding:0; margin:0; }
.ref[open] > .panel{
  position: fixed; z-index: 999; top: 12vh; left: 50%; transform: translateX(-50%);
  width: min(900px, 90vw); max-height: 75vh; overflow: auto; box-shadow:0 20px 60px rgba(0,0,0,.45);
}
.ref .close-x{ position:absolute; top:6px; right:10px; border:0; background:transparent; color:#94a3b8; font-size:20px; line-height:1; cursor:pointer; }

</style>
""", unsafe_allow_html=True)

# ================= Prompt helpers =================
def format_chat_history(messages):
    lines = []
    for m in messages:
        speaker = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines)

prompt = PromptTemplate(
    template = """
You are a financial-data extraction assistant.

**IMPORTANT CONDITIONAL FOLLOW-UP**
🛎️ After you answer the user’s question (using steps 1–4), **only if** there is still **unused** relevant report content, **ask**:
“Would you like more detail on [X]?”
Otherwise, **do not** ask any follow-up.

**HARD RULE (unrelated questions)**
If the user's question is unrelated to this PDF or requires information outside the Context, reply **exactly**:
"Sorry I can only answer question related to {pdf_name} pdf document"

**Use ONLY what appears under “Context”**.

### How to answer
1. Single value → short sentence with the exact number.
2. Table questions → return the full table in GitHub-flavoured markdown.
3. Valuation methods → synthesize across chunks; show weights and $ values; prefer detailed breakdowns.
4. Theory/text → explain using context.
5. If you cannot find an answer in Context → reply exactly:
   "Sorry I didnt understand the question. Did you mean SUGGESTION?"

---
Context:
{context}
---
Question: {question}
Answer:""",
    input_variables=["context", "question", "pdf_name"]
)

base_text = prompt.template
wrapped_prompt = PromptTemplate(
    template=base_text + """
Conversation so far:
{chat_history}
""",
    input_variables=["chat_history", "context", "question", "pdf_name"]
)

# ================= Input =================
user_q = st.chat_input("Type your question here…")

etran_clicked = components.html(
    """
    <style>
      #etran-fab {
        position: fixed;
        right: 20px; /* Reset to original to test alignment */
        bottom: 60px; /* Increased to clear the chat input area */
        z-index: 10000;
        display: block;
      }
      #etran-btn {
        border-radius: 9999px;
        padding: 10px 16px;
        background: #000;
        color: #fff;
        border: none;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
      }
      #etran-btn:hover { background: #222; }
      #etran-btn:active { transform: translateY(1px); }
    </style>
    <div id="etran-fab"><button id="etran-btn" type="button">Etran Sheet</button></div>
    <script src="https://unpkg.com/@streamlit/component-lib/dist/index.js"></script>
    <script>
      function send(v) { Streamlit.setComponentValue(v); }
      document.getElementById("etran-btn").addEventListener("click", function() { send(true); });
      Streamlit.setFrameHeight(0);
    </script>
    """,
    height=0
)

# Only act on a real click
if etran_clicked is True:
    payload = "Etran Sheet"
    st.session_state.messages.append({"id": _new_id(), "role": "user", "content": payload})
    st.session_state.pending_input = payload
    st.session_state.waiting_for_response = True

if user_q:
    st.session_state.messages.append({"id": _new_id(), "role": "user", "content": user_q})
    st.session_state.pending_input = user_q
    st.session_state.waiting_for_response = True

# ================= History =================
for msg in st.session_state.messages:
    cls = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    st.markdown(f"<div class='{cls} clearfix'>{msg['content']}</div>", unsafe_allow_html=True)
    if msg.get("source_img") and msg.get("source_pdf_b64") and msg.get("source_page"):
        render_reference_card(
            label=(msg.get("source") or "Page"),
            img_b64=msg["source_img"],
            pdf_b64=msg["source_pdf_b64"],
            page=msg["source_page"],
            key=msg.get("id", "k0"),
        )

# ================= Answer =================
if st.session_state.waiting_for_response:
    block = st.empty()
    with block.container():
        thinking = st.empty()
        thinking.markdown("<div class='assistant-bubble clearfix'>Thinking…</div>", unsafe_allow_html=True)

        raw_q = st.session_state.pending_input or ""
        history_to_use = st.session_state.messages[-10:]
        pdf_display = os.path.splitext(up.name)[0]

        intent = classify_reply_intent(raw_q, _last_assistant_text(history_to_use))
        is_deny = (intent == "DENY")
        is_confirm = (intent == "CONFIRM")

        if is_deny:
            st.session_state.last_suggestion = None
            answer = "Alright, if you have any more questions or need further assistance, feel free to ask!"
            entry = {"id": _new_id(), "role": "assistant", "content": answer}
            thinking.empty()
            with block.container():
                st.markdown(f"<div class='assistant-bubble clearfix'>{answer}</div>", unsafe_allow_html=True)
            st.session_state.messages.append(entry)
            st.session_state.pending_input = None
            st.session_state.waiting_for_response = False
            st.stop()

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
            "chat_history": format_chat_history(history_to_use),
            "context":      ctx,
            "question":     effective_q,
            "pdf_name":     pdf_display,
        }
        answer = llm.invoke(wrapped_prompt.invoke(full_input)).content
        answer = sanitize_suggestion(answer, effective_q, docs)

        # Save a CLEANED last suggestion for future confirmations
        sug = extract_suggestion(answer)
        st.session_state.last_suggestion = _clean_heading(sug) if sug else None

        apology = f"Sorry I can only answer question related to {pdf_display} pdf document"
        is_unrelated = apology.lower() in answer.strip().lower()
        is_clarify  = is_clarification(answer)

        entry = {"id": _new_id(), "role": "assistant", "content": answer}
        thinking.empty()

    # Render answer
    with block.container():
        st.markdown(f"<div class='assistant-bubble clearfix'>{answer}</div>", unsafe_allow_html=True)

        # Only attach a reference when it's a real, contentful answer
        skip_reference = is_unrelated or is_clarify
        if docs and not skip_reference:
            # ---------- TOP-3 LLM PICK (reference card) ----------
            try:
                top3 = docs[:3]
                best_doc = top3[0] if top3 else None
                if len(top3) >= 3:
                    ranking_prompt = PromptTemplate(
                        template=(
                            "Given the user's question and 3 candidate context chunks, "
                            "reply with only the number (1, 2, or 3) of the chunk that best answers it.\n\n"
                            "Question:\n{question}\n\n"
                            "Chunk 1:\n{chunk1}\n\n"
                            "Chunk 2:\n{chunk2}\n\n"
                            "Chunk 3:\n{chunk3}\n\n"
                            "Best Chunk Number:"
                        ),
                        input_variables=["question", "chunk1", "chunk2", "chunk3"]
                    )
                    pick = ChatOpenAI(model="gpt-4o", temperature=0).invoke(
                        ranking_prompt.invoke({
                            "question": query_for_retrieval,
                            "chunk1": top3[0].page_content,
                            "chunk2": top3[1].page_content,
                            "chunk3": top3[2].page_content
                        })
                    ).content.strip()
                    if pick.isdigit() and 1 <= int(pick) <= 3:
                        best_doc = top3[int(pick) - 1]

                if best_doc is not None:
                    ref_page = best_doc.metadata.get("page_number")
                    img = st.session_state.page_images.get(ref_page)
                    if img:
                        entry["source"] = f"Page {ref_page}"
                        entry["source_img"] = pil_to_base64(img)
                        entry["source_pdf_b64"] = base64.b64encode(st.session_state.pdf_bytes).decode("ascii")
                        entry["source_page"] = ref_page
                        # draw the card
                        render_reference_card(
                            label=entry["source"],
                            img_b64=entry["source_img"],
                            pdf_b64=entry["source_pdf_b64"],
                            page=entry["source_page"],
                            key=entry["id"],
                        )
            except Exception as e:
                st.info(f"ℹ️ Reference selection skipped: {e}")

    # Persist
    st.session_state.messages.append(entry)
    st.session_state.pending_input = None
    st.session_state.waiting_for_response = False
