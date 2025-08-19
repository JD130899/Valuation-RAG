# app.py  (drop-in replacement)

import os, io, pickle, base64, re, uuid, time, json
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv
import streamlit.components.v1 as components

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

DRIVE_FOLDER_FROM_SECRET = os.getenv("GOOGLE_DRIVE_FOLDER", "").strip()
HARDCODED_FOLDER_LINK = "https://drive.google.com/drive/folders/1XGyBBFhhQFiG43jpYJhNzZYi7C-_l5me"
FOLDER_TO_USE = DRIVE_FOLDER_FROM_SECRET or HARDCODED_FOLDER_LINK

# ---------- small helpers ----------
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

def _company_from_filename(pdf_name: str) -> str:
    base = os.path.splitext(pdf_name)[0]
    base = re.sub(r"\s*[-–—:]?\s*Certified\s+Valuation\s+Report.*$", "", base, flags=re.I)
    return base.strip() or "the subject company"

def _reset_chat():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."},
        {"role": "assistant", "content": "What can I help you with?"}
    ]
    st.session_state.pending_input = None
    st.session_state.waiting_for_response = False
    st.session_state.last_suggestion = None

# ---------- Session state ----------
if "last_synced_file_id" not in st.session_state:
    st.session_state.last_synced_file_id = None
if "messages" not in st.session_state:
    _reset_chat()
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None
if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False
if "page_images" not in st.session_state:
    st.session_state.page_images = {}  # {page_number:int -> base64 PNG string}
if "page_texts" not in st.session_state:
    st.session_state.page_texts = {}
if "last_suggestion" not in st.session_state:
    st.session_state.last_suggestion = None
if "next_msg_id" not in st.session_state:
    st.session_state.next_msg_id = 0

# ---------- styles (chat + reference styles) ----------
st.markdown("""
<style>
.block-container{ padding-top:54px!important; padding-bottom:160px!important; }
.block-container h1 { margin-top:0!important; }

/* Chat bubbles */
.user-bubble {background:#007bff;color:#fff;padding:8px;border-radius:8px;max-width:60%;float:right;margin:4px;}
.assistant-bubble {background:#1e1e1e;color:#fff;padding:8px;border-radius:8px;max-width:60%;float:left;margin:4px;}
.clearfix::after {content:"";display:table;clear:both;}

/* Reference card */
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

def _new_id():
    n = st.session_state.next_msg_id
    st.session_state.next_msg_id += 1
    return f"m{n}"

# ==================== INTERACTIONS ====================
def queue_question(q: str):
    st.session_state.pending_input = q
    st.session_state.waiting_for_response = True
    st.session_state.messages.append({"id": _new_id(), "role": "user", "content": q})

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

def render_reference_card(label: str, img_b64: str, page: int, key: str):
    """Uses the single shared st.session_state.pdf_b64 (not per-message)."""
    pdf_b64 = st.session_state.get("pdf_b64", "")
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

# ----------- helpers -----------
def _clean_heading(text: str) -> str:
    if not text: return text
    text = re.sub(r"^[#>\-\*\d\.\)\(]+\s*", "", text).strip()
    return text.strip(" :–—-·•")

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
    try:
        judge = PromptTemplate(
            template=(
                "You are a strict intent classifier.\n"
                "Assistant just said:\n{assistant}\n\n"
                "User replied:\n{user}\n\n"
                "Label as one of: CONFIRM, DENY, NEITHER.\n"
                "Reply with ONLY that token."
            ),
            input_variables=["assistant", "user"]
        )
        out = ChatOpenAI(model="gpt-4o", temperature=0).invoke(
            judge.invoke({"assistant": prev_assistant or "", "user": user_reply or ""})
        ).content.strip().upper()
        return out if out in {"CONFIRM","DENY","NEITHER"} else "NEITHER"
    except Exception:
        return "NEITHER"

def condense_query(chat_history, user_input: str, pdf_name: str) -> str:
    hist = []
    for m in chat_history[-6:]:
        speaker = "User" if m["role"] == "user" else "Assistant"
        hist.append(f"{speaker}: {m['content']}")
    hist_txt = "\n".join(hist)
    prompt = PromptTemplate(
        template=(
            'Turn the last user message into a self-contained search query about the PDF "{pdf_name}". '
            "Use ONLY info implied by the history; keep it short and noun-heavy. "
            "Return just the query.\n"
            "---\nHistory:\n{history}\n---\nLast user message: {question}\nStandalone query:"
        ),
        input_variables=["pdf_name", "history", "question"]
    )
    try:
        return ChatOpenAI(model="gpt-4o", temperature=0).invoke(
            prompt.invoke({"pdf_name": pdf_name, "history": hist_txt, "question": user_input})
        ).content.strip() or user_input
    except Exception:
        return user_input

# ================= Helper for page selection + OCR =================
def _detect_selected_pages(doc: fitz.Document):
    want = set()
    total = len(doc)
    if total >= 3:
        want.add(3)
    income_hits, market_hits, valuation_summary_page = [], [], None
    for i in range(total):
        text = (doc[i].get_text() or "").upper()
        if "INCOME APPROACH" in text: income_hits.append(i + 1)
        if "MARKET APPROACH" in text: market_hits.append(i + 1)
        if valuation_summary_page is None and "VALUATION SUMMARY" in text:
            valuation_summary_page = i + 1
    if income_hits: want.add(income_hits[0])
    if len(market_hits) >= 2:
        second = market_hits[1]; want.add(second)
        if second + 1 <= total: want.add(second + 1)
    if valuation_summary_page: want.add(valuation_summary_page)
    return want

def _ocr_page_with_gpt4o(img_png_bytes: bytes) -> str:
    b64 = base64.b64encode(img_png_bytes).decode("utf-8")
    prompt_text = (
        "Extract all values and details precisely from the image. "
        "Return clean, readable plain text (not markdown tables). "
        "If formulas are present, write THE EQUATION using simple math symbols. "
        "Avoid extra commentary."
    )
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}],
            }],
            max_tokens=800,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""

# ================= Builder =================
@st.cache_resource(show_spinner="📦 Processing & indexing PDF…")
def build_retriever_from_pdf(pdf_bytes: bytes, file_name: str):
    """
    Heavy work: page previews (base64 strings), selective OCR, LlamaParse, embeddings, FAISS, retriever
    """
    os.makedirs("uploaded", exist_ok=True)
    pdf_path = os.path.join("uploaded", file_name)
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    # Open with PyMuPDF
    doc = fitz.open(pdf_path)

    # Build base64 PNG previews (reduced DPI to save memory)
    page_images_b64 = {}
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=120)  # 96–120 is plenty
        img_b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
        page_images_b64[i + 1] = img_b64

    # Detect which pages to override with OCR
    selected_pages = _detect_selected_pages(doc)
    ocr_text_by_page = {}
    for pnum in sorted(selected_pages):
        try:
            # Use the already-rendered preview (saves memory)
            img_b = base64.b64decode(page_images_b64[pnum])
            ocr_text = _ocr_page_with_gpt4o(img_b)
            if ocr_text:
                ocr_text_by_page[pnum] = ocr_text
        except Exception:
            pass

    doc.close()

    # Parse entire document with LlamaParse
    parser = LlamaParse(api_key=os.environ["LLAMA_CLOUD_API_KEY"], num_workers=4)
    result = parser.parse(pdf_path)

    # Build pages array with OCR overrides
    pages, page_texts = [], {}
    for pg in result.pages:
        pnum = pg.page  # 1-based
        if pnum in ocr_text_by_page:
            text = ocr_text_by_page[pnum].strip()
        else:
            cleaned = [l for l in pg.md.splitlines() if l.strip() and l.lower() != "null"]
            text = "\n".join(cleaned)
        if text:
            pages.append(Document(page_content=text, metadata={"page_number": pnum}))
            page_texts[pnum] = text

    # Split → Embed → Index
    splitter = RecursiveCharacterTextSplitter(chunk_size=3300, chunk_overlap=0)
    chunks = splitter.split_documents(pages)
    for idx, c in enumerate(chunks):
        c.metadata["chunk_id"] = idx + 1

    embedder = CohereEmbeddings(
        model="embed-english-v3.0",
        user_agent="langchain",
        cohere_api_key=os.environ["COHERE_API_KEY"]
    )
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
    return retriever, page_images_b64, page_texts

# --------- ETRAN CHEATSHEET HELPERS ----------
def etran_extract_from_page3(page3_text: str) -> dict:
    want_keys = [
        "Concluded Value",
        "Purchase Type",
        "Fixed Asset Value",
        "Other Tangible Assets Value",
        "Goodwill Value",
        "Free Cash Flow",
    ]
    prompt = PromptTemplate(
        template=(
            "From the provided report text (this is the exact content of page 3), "
            "extract the following fields. Return ONLY valid JSON (no code fences), "
            "with these exact keys (even if value is unknown, use an empty string):\n"
            f"{want_keys}\n\n"
            "Text:\n{page3}\n"
        ),
        input_variables=["page3"]
    )
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        raw = llm.invoke(prompt.invoke({"page3": page3_text})).content.strip()
        data = json.loads(raw)
    except Exception:
        data = {k: "" for k in want_keys}
    for k in want_keys:
        data.setdefault(k, "")
    return data

def render_etran_table(dynamic: dict) -> str:
    static_rows = [
        ("Appraisal Firm", "Value Buddy"),
        ("Appraiser", "Tim Gbur"),
        ("Appraiser Certification", "NACVA - Certified Valuation Analyst (CVA)"),
    ]
    dyn_rows = [
        ("Concluded Value", dynamic.get("Concluded Value","")),
        ("Purchase Type", dynamic.get("Purchase Type","")),
        ("Fixed Asset Value", dynamic.get("Fixed Asset Value","")),
        ("Other Tangible Assets Value", dynamic.get("Other Tangible Assets Value","")),
        ("Goodwill Value", dynamic.get("Goodwill Value","")),
        ("Free Cash Flow", dynamic.get("Free Cash Flow","")),
    ]
    rows = static_rows + dyn_rows
    lines = ["| Field | Value |", "|---|---|"]
    for k, v in rows:
        val = (v or "").replace("\n", " ").strip()
        lines.append(f"| {k} | {val if val else '—'} |")
    return "\n".join(lines)

# ================= Sidebar: Google Drive loader =================
service = get_drive_service()

pdf_files = get_all_pdfs(service, FOLDER_TO_USE)
st.sidebar.caption(f"📁 Using Drive folder: {FOLDER_TO_USE}")

if not pdf_files:
    st.sidebar.warning("No PDFs found in the folder.")
else:
    if "selected_pdf_name" not in st.session_state:
        st.session_state.selected_pdf_name = pdf_files[0]["name"]

    names = [f["name"] for f in pdf_files]
    sel_name = st.sidebar.selectbox(
        "Select a PDF to load",
        names,
        index=names.index(st.session_state.selected_pdf_name) if st.session_state.get("selected_pdf_name") in names else 0,
        key="selected_pdf_name",
    )

    if st.sidebar.button("Load selected PDF"):
        chosen = next(f for f in pdf_files if f["name"] == sel_name)
        if chosen["id"] == st.session_state.get("last_synced_file_id"):
            st.sidebar.info("Already loaded.")
        else:
            path = download_pdf(service, chosen["id"], chosen["name"])
            if path:
                with open(path, "rb") as f:
                    st.session_state.uploaded_file_from_drive = f.read()
                st.session_state.uploaded_file_name = chosen["name"]
                st.session_state.last_synced_file_id = chosen["id"]
                _reset_chat()

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
        if up.name != st.session_state.get("last_selected_upload"):
            st.session_state.last_selected_upload = up.name
            _reset_chat()

if not up:
    st.warning("Please upload or load a PDF to continue.")
    st.stop()

# Rebuild retriever when file changes (set name early to avoid partial reruns)
if st.session_state.get("last_processed_pdf") != up.name:
    st.session_state.last_processed_pdf = up.name   # set early to avoid duplicate rebuilds
    pdf_bytes = up.getvalue()
    st.session_state.pdf_bytes = pdf_bytes
    st.session_state.pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")  # single shared copy

    (st.session_state.retriever,
     st.session_state.page_images,
     st.session_state.page_texts) = build_retriever_from_pdf(pdf_bytes, up.name)

    _reset_chat()

# ===== Bottom-right pinned quick actions (compact pill) =====
pill = st.container()
with pill:
    st.markdown("<span id='pin-bottom-right'></span>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    if c1.button("Valuation", key="qa_val"):        queue_question("Valuation")
    if c2.button("Good will", key="qa_gw"):         queue_question("Good will")
    if c3.button("Etran Cheatsheet", key="qa_etran"): queue_question("Etran Cheatsheet")

components.html("""
<script>
(function pin(){
  const d = window.parent.document;
  const mark = d.querySelector('#pin-bottom-right');
  if (!mark) return setTimeout(pin,120);

  const block = mark.closest('div[data-testid="stVerticalBlock"]');       // the block you fix-position
  if (!block) return setTimeout(pin,120);
  if (block.dataset.pinned === "1") return;
  block.dataset.pinned = "1";

  // Collapse all layout ancestors so they don't reserve space
  const elc = block.closest('div[data-testid="stElementContainer"]');     // outer element container
  const vblk = block;                                                     // vertical block itself
  const hblk = block.querySelector('div[data-testid="stHorizontalBlock"]'); // created by st.columns

  [elc, vblk, hblk].forEach(n => {
    if (!n) return;
    n.style.height = '0px';
    n.style.minHeight = '0';
    n.style.margin = '0';
    n.style.padding = '0';
  });
  if (elc) { elc.style.display = 'contents'; }  // remove its box entirely

  // Now float the pill
  Object.assign(block.style, {
    position: 'fixed', right: '0px', bottom: '100px', zIndex: '10000',
    display: 'flex', flexWrap: 'nowrap', gap: '12px', padding: '10px 118px',
    borderRadius: '9999px', background: 'transparent', border: 'none',
    boxShadow: 'none', minWidth: '350px', width: 'fit-content', whiteSpace: 'nowrap',
    pointerEvents: 'auto'
  });

  // Tidy up column wrappers & buttons
  Array.from(block.children||[]).forEach(ch => { ch.style.width='auto'; ch.style.margin='0'; });
  block.querySelectorAll('button').forEach(b => {
    b.style.padding='18px 32px'; b.style.fontSize='18px'; b.style.borderRadius='9999px';
  });
})();
</script>
""", height=0)


# Chat input
user_q = st.chat_input("Type your question here…", key="main_chat_input")
if user_q:
    queue_question(user_q)

# ========================== RENDER HISTORY ==========================
for msg in st.session_state.messages:
    cls = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    st.markdown(f"<div class='{cls} clearfix'>{msg['content']}</div>", unsafe_allow_html=True)
    if msg.get("source_img") and msg.get("source_page"):
        render_reference_card(
            label=(msg.get("source") or "Page"),
            img_b64=msg["source_img"],
            page=msg["source_page"],
            key=msg.get("id", "k0"),
        )

# ========================== ANSWER ==========================
if st.session_state.waiting_for_response and st.session_state.pending_input:
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

        # ===== SPECIAL CASES (ETRAN / VALUATION / GOODWILL) =====
        def _emit_answer_with_ref(answer_text: str, page_num: int | None):
            thinking.empty()
            with block.container():
                type_bubble(answer_text)
                entry = {"id": _new_id(), "role": "assistant", "content": answer_text}
                if page_num:
                    img_b64 = st.session_state.page_images.get(page_num)
                    if img_b64:
                        entry.update({
                            "source": f"Page {page_num}",
                            "source_img": img_b64,
                            "source_page": page_num,
                        })
                        render_reference_card(
                            label=entry["source"], img_b64=img_b64,
                            page=page_num, key=entry["id"]
                        )
                st.session_state.messages.append(entry)
            st.session_state.pending_input = None
            st.session_state.waiting_for_response = False

        rq_l = raw_q.strip().lower()
        if rq_l in {"etran cheatsheet", "etran cheat sheet", "etran"}:
            page3_text = (st.session_state.page_texts or {}).get(3, "")
            if not page3_text:
                _emit_answer_with_ref("I couldn’t find page 3 content in this PDF.", None)
            else:
                extracted = etran_extract_from_page3(page3_text)
                table_md = render_etran_table(extracted)
                _emit_answer_with_ref(f"### Etran Cheatsheet\n\n{table_md}", 3)
            st.stop()

        if rq_l in {"valuation", "fair market value", "fmv", "value"}:
            page3_text = (st.session_state.page_texts or {}).get(3, "")
            if not page3_text:
                _emit_answer_with_ref("I couldn’t find page 3 content in this PDF.", None)
            else:
                data = etran_extract_from_page3(page3_text)
                fmv = (data.get("Concluded Value") or data.get("Fair Market Value") or "").strip()
                company = _company_from_filename(up.name)
                ans = f"Valuation of {company} is {fmv}." if fmv else "I couldn’t find a clear Fair Market Value on page 3."
                _emit_answer_with_ref(ans, 3)
            st.stop()

        if rq_l in {"good will", "goodwill"}:
            page3_text = (st.session_state.page_texts or {}).get(3, "")
            if not page3_text:
                _emit_answer_with_ref("I couldn’t find page 3 content in this PDF.", None)
            else:
                data = etran_extract_from_page3(page3_text)
                goodwill = (data.get("Goodwill Value") or "").strip()
                company = _company_from_filename(up.name)
                ans = f"Goodwill value of {company} is {goodwill}." if goodwill else "I couldn’t find a clear Goodwill value on page 3."
                _emit_answer_with_ref(ans, 3)
            st.stop()

        if is_deny:
            thinking.empty()
            answer = "Alright, if you have any more questions or need further assistance, feel free to ask!"
            with block.container():
                st.markdown(f"<div class='assistant-bubble clearfix'>{answer}</div>", unsafe_allow_html=True)
                st.session_state.messages.append({"id": _new_id(), "role": "assistant", "content": answer})
            st.session_state.pending_input = None
            st.session_state.waiting_for_response = False
        else:
            effective_q = st.session_state.last_suggestion if (is_confirm and st.session_state.last_suggestion) else raw_q
            query_for_retrieval = condense_query(history_to_use, effective_q, pdf_display)

            ctx, docs, retr_err = "", [], None
            try:
                docs = st.session_state.retriever.get_relevant_documents(query_for_retrieval)
                ctx = "\n\n".join(d.page_content for d in docs)
            except Exception as e:
                retr_err = str(e)

            if retr_err:
                st.warning(f"RAG retrieval error: {retr_err}")

            try:
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

**IMPORTANT CONDITIONAL FOLLOW-UP**
After you answer the user’s question (using steps 1–4), only if there is still unused relevant report content, ask:
“Would you like more detail on [X]?”
Otherwise, do not ask any follow-up.

**HARD RULE (unrelated questions)**
If the user's question is unrelated to this PDF or requires information outside the Context, reply exactly:
"Sorry I can only answer question related to {pdf_name} pdf document"

Use ONLY what appears under “Context”.

### How to answer
1. Single value → short sentence with the exact number.
2. Table questions → return the full table in GitHub-flavoured markdown.
3. Valuation methods → synthesize across chunks; show weights and $ values.
4. Theory/text → explain using context.
5. If you cannot find an answer in Context → reply exactly:
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
            except Exception as e:
                answer = f"Sorry, I hit an error while answering: {e}"

            answer = sanitize_suggestion(answer, effective_q, docs)
            st.session_state.last_suggestion = _clean_heading(extract_suggestion(answer) or "") or None

            apology = f"Sorry I can only answer question related to {pdf_display} pdf document"
            is_unrelated = apology.lower() in answer.strip().lower()
            is_clarify  = is_clarification(answer)

            entry = {"id": _new_id(), "role": "assistant", "content": answer}
            thinking.empty()

            with block.container():
                type_bubble(answer)
                skip_reference = is_unrelated or is_clarify
                if docs and not skip_reference:
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
                            try:
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
                            except Exception:
                                pass

                        if best_doc is not None:
                            ref_page = best_doc.metadata.get("page_number")
                            img_b64 = st.session_state.page_images.get(ref_page)
                            if img_b64:
                                entry["source"] = f"Page {ref_page}"
                                entry["source_img"] = img_b64
                                entry["source_page"] = ref_page
                                render_reference_card(
                                    label=entry["source"],
                                    img_b64=img_b64,
                                    page=ref_page,
                                    key=entry["id"],
                                )
                    except Exception as e:
                        st.info(f"ℹ️ Reference selection skipped: {e}")

            st.session_state.messages.append(entry)
            st.session_state.pending_input = None
            st.session_state.waiting_for_response = False
