# app.py  (simplified: no Google Drive, no reference section)

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

from sklearn.metrics.pairwise import cosine_similarity
import openai

# ================= Setup =================
load_dotenv()
st.set_page_config(page_title="Underwriting Agent", layout="wide")
openai.api_key = os.getenv("OPENAI_API_KEY")

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
if "last_synced_file_id" not in st.session_state:  # kept for parity; unused in this simplified app
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

# ---------- styles (chat only) ----------
st.markdown("""
<style>
.block-container{ padding-top:54px!important; padding-bottom:160px!important; }
.block-container h1 { margin-top:0!important; }

/* Chat bubbles */
.user-bubble {background:#007bff;color:#fff;padding:8px;border-radius:8px;max-width:60%;float:right;margin:4px;}
.assistant-bubble {background:#1e1e1e;color:#fff;padding:8px;border-radius:8px;max-width:60%;float:left;margin:4px;}
.clearfix::after {content:"";display:table;clear:both;}
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

def file_badge_link(name: str, pdf_bytes: bytes):
    base = os.path.splitext(name)[0]
    b64 = base64.b64encode(pdf_bytes).decode("ascii")
    link_id = f"open-file-{uuid.uuid4().hex[:8]}"
    st.markdown(
        f'''
        <div style="background:#1f2c3a; padding:8px; border-radius:8px; color:#fff;">
          ✅ <b>Using file:</b>
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

# Embeddings (Cohere)
embedder = CohereEmbeddings(
    model="embed-english-v3.0",
    user_agent="langchain",
    cohere_api_key=os.environ["COHERE_API_KEY"]
)

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

# ================= Main UI =================
st.title("Underwriting Agent")

up = st.file_uploader("Upload a valuation report PDF", type="pdf")
if up:
    file_badge_link(up.name, up.getvalue())
    if up.name != st.session_state.get("last_selected_upload"):
        st.session_state.last_selected_upload = up.name
        _reset_chat()

if not up:
    st.warning("Please upload a PDF to continue.")
    st.stop()

# Rebuild retriever when file changes
if st.session_state.get("last_processed_pdf") != up.name:
    st.session_state.last_processed_pdf = up.name
    pdf_bytes = up.getvalue()
    st.session_state.pdf_bytes = pdf_bytes
    st.session_state.pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")  # kept for parity

    (st.session_state.retriever,
     st.session_state.page_images,
     st.session_state.page_texts) = build_retriever_from_pdf(pdf_bytes, up.name)

    _reset_chat()

# Chat input
user_q = st.chat_input("Type your question here…", key="main_chat_input")
if user_q:
    queue_question(user_q)

# ========================== RENDER HISTORY ==========================
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

        intent = classify_reply_intent(raw_q, _last_assistant_text(history_to_use))
        is_deny = (intent == "DENY")
        is_confirm = (intent == "CONFIRM")

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
        🛎️ After you answer the user’s question (using steps 1–4), **only if** there is still **unused** relevant report content, **ask**:  
            “Would you like more detail on [X]?”  
            Otherwise, **do not** ask any follow-up.

        **HARD RULE (unrelated questions)**
        If the user's question is unrelated to this PDF or requires information outside the Context, reply exactly:
        "Sorry I can only answer question related to {pdf_name} pdf document"

        **Use ONLY what appears under “Context”.**

        ### How to answer
        1. **Single value questions**  
        • Find the row + column that match the user's words.  
        • Return the answer in a **short, clear sentence** using the exact number from the context.  
            Example: “The Income (DCF) approach value is $1,150,000.”  
        • **Do NOT repeat the metric name or company name** unless the user asks.
        
        2. **Table questions**  
        • Return the full table **with its header row** in GitHub-flavoured markdown.
        
        3. **Valuation method / theory / reasoning questions**
            
        • If the question involves **valuation methods**, **concluded value**, or topics like **Income Approach**, **Market Approach**, or **Valuation Summary**, do the following:
            - Combine and synthesize relevant information across all chunks.
            - Pay special attention to how **weights are distributed** (e.g., “50% DCF, 25% EBITDA, 25% SDE”).
            - Avoid oversimplifying if more detailed breakdowns (like subcomponents of market approach) are available.
            - If a table gives a simplified view (e.g., "50% Market Approach"), but other parts break it down (e.g., 25% EBITDA + 25% SDE), **prefer the detailed breakdown with percent value**.   
            - When describing weights, also mention the **corresponding dollar values** used in the context (e.g., “50% DCF = $3,712,000, 25% EBITDA = $4,087,000...”)
            - **If Market approach is composed of sub-methods like EBITDA and SDE, then explicitly extract and show their individual weights and values, even if not listed together in a single table.**
            
    
        4. **Theory/textual question**  
        • Try to return an explanation **based on the context**.

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

            thinking.empty()
            with block.container():
                type_bubble(answer)

            st.session_state.messages.append({"id": _new_id(), "role": "assistant", "content": answer})
            st.session_state.pending_input = None
            st.session_state.waiting_for_response = False
