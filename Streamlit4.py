import os, io, pickle, base64
import re
import uuid

import streamlit as st
import streamlit.components.v1 as components
import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv

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

# ------------------- Setup -------------------
load_dotenv()
st.set_page_config(page_title="Underwriting Agent", layout="wide")
openai.api_key = os.getenv("OPENAI_API_KEY")

# ------------------- Session state -------------------
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
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "page_images" not in st.session_state:
    st.session_state.page_images = {}
if "next_msg_id" not in st.session_state:
    st.session_state.next_msg_id = 0
# follow-up memory (for "Would you like more detail on X?")
if "pending_followup" not in st.session_state:
    st.session_state.pending_followup = None

# yes/confirm regex + follow-up extraction
FOLLOWUP_RX = re.compile(r"would you like more detail on\s+(.+?)\?\s*$", re.IGNORECASE | re.MULTILINE)
YES_RX      = re.compile(r"^\s*(yes|y|yeah|yep|sure|ok(?:ay)?|correct|that's right|please do|go ahead)\s*[.!?]*\s*$", re.IGNORECASE)

# ------------------- Utilities -------------------
def _new_id():
    n = st.session_state.next_msg_id
    st.session_state.next_msg_id += 1
    return f"m{n}"

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def file_badge_link(name: str, pdf_bytes: bytes, synced: bool = True):
    base = os.path.splitext(name)[0]          # remove .pdf
    b64  = base64.b64encode(pdf_bytes).decode("ascii")
    label = "Using synced file:" if synced else "Using file:"
    link_id = f"open-file-{uuid.uuid4().hex[:8]}"

    st.markdown(
        f'''
        <div style="background:#1f2c3a; padding:8px; border-radius:8px; color:#fff;">
          ✅ <b>{label}</b>
          <a id="{link_id}" href="#" target="_blank" rel="noopener"
             style="color:#93c5fd; text-decoration:none;">{base}</a>
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

# give IDs to any preloaded messages (greetings)
for m in st.session_state.messages:
    if "id" not in m:
        m["id"] = _new_id()

# ------------------- Retriever builder -------------------
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

# ------------------- Styles -------------------
st.markdown("""
<style>
/* Chat bubbles */
.user-bubble,
.assistant-bubble{
  display:inline-block;            /* shrink to content width */
  box-sizing:border-box;
  max-width:60%;
  padding:10px 12px;
  margin:6px 4px;
  border-radius:10px;
  color:#fff;
  line-height:1.45;

  /* keep text inside the box */
  white-space:pre-wrap;            /* respect newlines */
  overflow-wrap:anywhere;          /* break long tokens */
  word-break:break-word;

  /* avoid unintended italics from markdown */
  font-style:normal;
  font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
}
.user-bubble{ background:#007bff; float:right; }
.assistant-bubble{ background:#1e1e1e; float:left; }

/* tidy up markdown inside bubbles */
.user-bubble p,
.assistant-bubble p{ margin:0 0 .5rem; }
.user-bubble ul,.user-bubble ol,
.assistant-bubble ul,.assistant-bubble ol{ margin:.25rem 0 .75rem; padding-left:1.25rem; }
.user-bubble li,.assistant-bubble li{ margin:.15rem 0; }
.user-bubble code,.assistant-bubble code,
.user-bubble pre,.assistant-bubble pre{ white-space:pre-wrap; word-break:break-word; }
.user-bubble em,.user-bubble i,
.assistant-bubble em,.assistant-bubble i{ font-style:normal; }

/* clearfix for floats */
.clearfix::after { content:""; display:block; clear:both; }

/* Reference chip + panel */
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
.ref[open] .overlay{
  display:block; position:fixed; inset:0; z-index:998; background:transparent; border:0; padding:0; margin:0;
}
.ref[open] > .panel{
  position: fixed; z-index: 999; top: 12vh; left: 50%; transform: translateX(-50%);
  width: min(900px, 90vw); max-height: 75vh; overflow: auto; box-shadow:0 20px 60px rgba(0,0,0,.45);
}
.ref .close-x{
  position:absolute; top:6px; right:10px; border:0; background:transparent;
  color:#94a3b8; font-size:20px; line-height:1; cursor:pointer;
}

/* Link chip */
.chip{ display:inline-flex;align-items:center;gap:.5rem;
  background:#0f172a;color:#e2e8f0;border:1px solid #334155;
  border-radius:10px;padding:.35rem .6rem;font:14px/1.2 system-ui; }
.chip a{color:#93c5fd;text-decoration:none}
.chip a:hover{text-decoration:underline}
</style>
""", unsafe_allow_html=True)


# ------------------- Prompt helpers -------------------
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
🛎️ After you answer the user’s question (using steps 1–4), only if there is still unused relevant report content, ask:  
  “Would you like more detail on [X]?”  
Otherwise, do not ask any follow-up.

Use ONLY what appears under “Context”.

### How to answer
1. **Single value questions**
   • Find the row + column that match the user's words.  
   • Return the answer in a short, clear sentence using the exact number from the context.  
   • Do NOT repeat the metric/company name unless asked.

2. **Table questions**
   • Return the full table with header row in GitHub-flavoured markdown.

3. **Valuation method / theory / reasoning questions**
   • Synthesize across chunks; pay attention to weights and dollar values.
   • If Market approach has sub-methods (EBITDA/SDE), show their individual weights/values.

4. **Theory/text-only questions**
   • Explain based on context.

5. **Unrelated questions**
   Reply exactly: "Sorry I can only answer question related to {pdf_name} pdf document"

6. **If not enough info**
   Reply: “Hmm, I am not sure. Are you able to rephrase your question?”    

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

# ------------------- Sidebar: Google Drive loader -------------------
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

# ------------------- Main UI -------------------
st.title("Underwriting Agent")

# Source selector (Drive or local upload)
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

# Build/rebuild retriever when file changes
if st.session_state.get("last_processed_pdf") != up.name:
    pdf_bytes = up.getvalue()
    st.session_state.pdf_bytes = pdf_bytes
    st.session_state.retriever, st.session_state.page_images = build_retriever_from_pdf(pdf_bytes, up.name)
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."},
        {"role": "assistant", "content": "What can I help you with?"}
    ]
    st.session_state.last_processed_pdf = up.name

# ------------------- Input -------------------
user_q = st.chat_input("Type your question here…")
if user_q:
    st.session_state.messages.append({"id": _new_id(), "role": "user", "content": user_q})
    # If last assistant asked a follow-up and the user says "yes", turn it into a real query
    if st.session_state.pending_followup and YES_RX.match(user_q or ""):
        st.session_state.pending_input = f"More detail on {st.session_state.pending_followup}"
    else:
        st.session_state.pending_input = user_q
    # clear the pending follow-up either way
    st.session_state.pending_followup = None
    st.session_state.waiting_for_response = True

# ------------------- History render -------------------
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

# ------------------- Answer (single-pass, no rerun) -------------------
if st.session_state.waiting_for_response:
    block = st.empty()
    with block.container():
        thinking = st.empty()
        thinking.markdown("<div class='assistant-bubble clearfix'>Thinking…</div>", unsafe_allow_html=True)

        q = st.session_state.pending_input or ""
        ctx, docs = "", []
        try:
            docs = st.session_state.retriever.get_relevant_documents(q)
            ctx = "\n\n".join(d.page_content for d in docs)
        except Exception as e:
            st.warning(f"RAG retrieval error: {e}")

        history_to_use = st.session_state.messages[-10:]
        pdf_display = os.path.splitext(up.name)[0]

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        full_input = {
            "chat_history": format_chat_history(history_to_use),
            "context":      ctx,
            "question":     q,
            "pdf_name":     pdf_display,
        }
        answer = llm.invoke(wrapped_prompt.invoke(full_input)).content

        # Capture follow-up topic for the next turn (if the model asked one)
        m = FOLLOWUP_RX.search(answer)
        if m:
            topic = re.sub(r"^[\-\u2022•*·]+\s*", "", m.group(1).strip())
            st.session_state.pending_followup = topic or None

        entry = {"id": _new_id(), "role": "assistant", "content": answer}

        # Simple reference: first retrieved chunk's page preview
        if docs:
            best_doc = docs[0]
            ref_page = best_doc.metadata.get("page_number")
            img = st.session_state.page_images.get(ref_page)
            if img:
                entry["source"] = f"Page {ref_page}"
                entry["source_img"] = pil_to_base64(img)
                entry["source_pdf_b64"] = base64.b64encode(st.session_state.pdf_bytes).decode("ascii")
                entry["source_page"] = ref_page

        thinking.empty()

    # Final render (no rerun)
    with block.container():
        st.markdown(f"<div class='assistant-bubble clearfix'>{answer}</div>", unsafe_allow_html=True)
        if entry.get("source_img"):
            render_reference_card(
                label=entry.get("source", f"Page {entry.get('source_page')}"),
                img_b64=entry["source_img"],
                pdf_b64=entry["source_pdf_b64"],
                page=entry["source_page"],
                key=entry.get("id", "k0"),
            )

    # Persist
    st.session_state.messages.append(entry)
    st.session_state.pending_input = None
    st.session_state.waiting_for_response = False
