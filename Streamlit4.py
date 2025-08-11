import os, io, pickle, base64
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

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
import streamlit.components.v1 as components

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
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "page_images" not in st.session_state:
    st.session_state.page_images = {}

# --- Stable IDs for messages ---
if "next_msg_id" not in st.session_state:
    st.session_state.next_msg_id = 0

if "loading_new_pdf" not in st.session_state:
    st.session_state.loading_new_pdf = False



def _new_id():
    n = st.session_state.next_msg_id
    st.session_state.next_msg_id += 1
    return f"m{n}"


# Make a ONE-PAGE PDF (base64) from a given page
def single_page_pdf_b64(pdf_bytes: bytes, page_number: int) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    one = fitz.open()
    one.insert_pdf(doc, from_page=page_number-1, to_page=page_number-1)
    b64 = base64.b64encode(one.tobytes()).decode("ascii")
    one.close(); doc.close()
    return b64

def render_reference_card(label: str, img_b64: str, page_b64: str, key: str):
    # Markup: chip + overlay + modal panel
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

    # JS: attach blob URL to link + closing (overlay, X, Esc). Hidden iframe (height=0) to run script.
    components.html(
        f"""<!doctype html><meta charset='utf-8'>
<style>html,body{{background:transparent;margin:0;height:0;overflow:hidden}}</style>
<script>(function(){{
  function b64ToUint8Array(s){{var b=atob(s),u=new Uint8Array(b.length);for(var i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u;}}
  var blob = new Blob([b64ToUint8Array('{page_b64}')], {{type:'application/pdf'}});
  var url  = URL.createObjectURL(blob);

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
    
                # <<< wipe chat immediately and mark that we’re building >>>
                st.session_state.messages = []
                st.session_state.loading_new_pdf = True
                # make sure the next block rebuilds
                st.session_state.last_processed_pdf = None
    
                st.rerun()

else:
    st.sidebar.warning("📭 No PDFs found in Drive.")

# ================= Main UI =================
st.title("Underwriting Agent")

# Source selector (Drive or local upload)
if "uploaded_file_from_drive" in st.session_state:
    st.markdown(
        f"<div style='background:#1f2c3a; padding:8px; border-radius:8px; color:#fff;'>"
        f"✅ <b>Using synced file:</b> {st.session_state.uploaded_file_name}"
        "</div>",
        unsafe_allow_html=True
    )
    up = io.BytesIO(st.session_state.uploaded_file_from_drive)
    up.name = st.session_state.uploaded_file_name
else:
    up = st.file_uploader("Upload a valuation report PDF", type="pdf")

if not up:
    st.warning("Please upload or load a PDF to continue.")
    st.stop()

# Rebuild retriever when file changes

if st.session_state.get("last_processed_pdf") != up.name:
    # First pass after a new selection: clear UI and rerun quickly
    if not st.session_state.loading_new_pdf:
        st.session_state.messages = []
        st.session_state.loading_new_pdf = True
        st.rerun()

    # Second pass: actually build
    pdf_bytes = up.getvalue()
    st.session_state.pdf_bytes = pdf_bytes
    st.session_state.retriever, st.session_state.page_images = build_retriever_from_pdf(pdf_bytes, up.name)

    # Re-seed greetings only after build completes
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."},
        {"role": "assistant", "content": "What can I help you with?"}
    ]
    st.session_state.last_processed_pdf = up.name
    st.session_state.loading_new_pdf = False

    # Optional: quick refresh so the new greetings appear instantly
    st.rerun()


# ================= Styles =================
st.markdown("""
<style>
.user-bubble {background:#007bff;color:#fff;padding:8px;border-radius:8px;max-width:60%;float:right;margin:4px;}
.assistant-bubble {background:#1e1e1e;color:#fff;padding:8px;border-radius:8px;max-width:60%;float:left;margin:4px;}
.clearfix::after {content:"";display:table;clear:both;}

/* Reference chip + panel */
.ref{ display:block; width:60%; max-width:900px; margin:6px 0 12px 8px; }

/* the chip */
.ref summary{
  display:inline-flex; align-items:center; gap:8px; cursor:pointer; list-style:none; outline:none;
  background:#0f172a; color:#e2e8f0; border:1px solid #334155; border-radius:10px; padding:6px 10px;
}
.ref summary::before{ content:"▶"; font-size:12px; line-height:1; }
.ref[open] summary::before{ content:"▼"; }
/* keep summary (chip) visible in place */
.ref[open] > summary{}

/* the lightbox panel */
.ref .panel{
  background:#0f172a; color:#e2e8f0; border:1px solid #334155; border-top:none;
  border-radius:10px; padding:10px; margin-top:0; box-shadow:0 6px 20px rgba(0,0,0,.25);
}
.ref .panel img{ width:100%; height:auto; border-radius:8px; display:block; }

/* overlay that closes the lightbox (separate from <summary>) */
.ref .overlay{ display:none; }
.ref[open] .overlay{
  display:block; position:fixed; inset:0; z-index:998;
  background:transparent; border:0; padding:0; margin:0;
}

/* float the panel as a modal when open */
.ref[open] > .panel{
  position: fixed; z-index: 999; top: 12vh; left: 50%; transform: translateX(-50%);
  width: min(900px, 90vw); max-height: 75vh; overflow: auto; box-shadow:0 20px 60px rgba(0,0,0,.45);
}

/* optional close X */
.ref .close-x{
  position:absolute; top:6px; right:10px; border:0; background:transparent;
  color:#94a3b8; font-size:20px; line-height:1; cursor:pointer;
}

/* (kept from before) small link-chip style if you use it elsewhere */
.chip{ display:inline-flex;align-items:center;gap:.5rem;
  background:#0f172a;color:#e2e8f0;border:1px solid #334155;
  border-radius:10px;padding:.35rem .6rem;font:14px/1.2 system-ui; }
.chip a{color:#93c5fd;text-decoration:none}
.chip a:hover{text-decoration:underline}
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
    template="""
You are a financial-data extraction assistant.

**Use ONLY what appears under “Context”.**
1) Single value → short sentence with the exact number.
2) Table questions → return the full table in GitHub-flavoured markdown.
3) Valuation methods → synthesize across chunks, show weights and corresponding $ values; prefer detailed breakdowns.
4) Theory/text → explain using context.

If not enough info: “Hmm, I am not sure. Are you able to rephrase your question?”
---
Context:
{context}
---
Question: {question}
Answer:""",
    input_variables=["context", "question"]
)

base_text = prompt.template
wrapped_prompt = PromptTemplate(
    template=base_text + "\nConversation so far:\n{chat_history}\n",
    input_variables=["chat_history", "context", "question"]
)

# ================= Input =================
user_q = st.chat_input("Type your question here…")
if user_q:
    st.session_state.messages.append({"id": _new_id(), "role": "user", "content": user_q})
    st.session_state.pending_input = user_q
    st.session_state.waiting_for_response = True

# ================= History =================
for msg in st.session_state.messages:
    cls = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    st.markdown(f"<div class='{cls} clearfix'>{msg['content']}</div>", unsafe_allow_html=True)

    if msg.get("source_img") and msg.get("source_b64"):
        render_reference_card(
            label=(msg.get("source") or "Page"),  # e.g., "Page 17"
            img_b64=msg["source_img"],
            page_b64=msg["source_b64"],
            key=msg.get("id", "k0"),
        )



# ================= Answer (single-pass, no rerun) =================
if st.session_state.waiting_for_response:
    block = st.empty()
    with block.container():
        with st.spinner("Thinking…"):
            q = st.session_state.pending_input or ""
            ctx, docs = "", []
            try:
                docs = st.session_state.retriever.get_relevant_documents(q)
                ctx = "\n\n".join(d.page_content for d in docs)
            except Exception as e:
                st.warning(f"RAG retrieval error: {e}")

            system_prompt = (
                "You are a helpful assistant. Use ONLY the content under 'Context' to answer. "
                "If the answer is not in the context, say you don't have enough information.\n\n"
                f"Context:\n{ctx}"
            )

            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": system_prompt}, *st.session_state.messages],
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"❌ Error: {e}"

            entry = {"id": _new_id(), "role": "assistant", "content": answer}
            ref_page, ref_img_b64 = None, None

            try:
                if docs:
                    texts = [d.page_content for d in docs]
                    embedder = CohereEmbeddings(
                        model="embed-english-v3.0",
                        user_agent="langchain",
                        cohere_api_key=st.secrets["COHERE_API_KEY"]
                    )
                    emb_answer = embedder.embed_query(answer)
                    chunk_embs = embedder.embed_documents(texts)
                    sims = cosine_similarity([emb_answer], chunk_embs)[0]
                    ranked = sorted(list(zip(docs, sims)), key=lambda x: x[1], reverse=True)
                    top3 = [d for d, _ in ranked[:3]]

                    best_doc = top3[0] if top3 else (ranked[0][0] if ranked else None)
                    if len(top3) >= 3:
                        ranking_prompt = PromptTemplate(
                            template=("Given a user question and 3 candidate context chunks, return the number (1-3) "
                                      "of the chunk that best answers it.\n\n"
                                      "Question:\n{question}\n\nChunk 1:\n{chunk1}\n\nChunk 2:\n{chunk2}\n\nChunk 3:\n{chunk3}\n\nBest Chunk Number:\n"),
                            input_variables=["question", "chunk1", "chunk2", "chunk3"]
                        )
                        pick = ChatOpenAI(model="gpt-4o", temperature=0).invoke(
                            ranking_prompt.invoke({
                                "question": q,
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
                        ref_img_b64 = pil_to_base64(img) if img else None
                        if ref_img_b64:
                            entry["source"] = f"Page {ref_page}"
                            entry["source_img"] = ref_img_b64
                            # Always open a single-page blob (reliable across Cloud/Drive)
                            entry["source_b64"] = single_page_pdf_b64(st.session_state.pdf_bytes, ref_page)

            except Exception as e:
                st.info(f"ℹ️ Reference selection skipped: {e}")

    # Final render (no rerun)
    with block.container():
        st.markdown(f"<div class='assistant-bubble clearfix'>{answer}</div>", unsafe_allow_html=True)

        # Register the blob URL for this new message (no white bars; height=0)
        if entry.get("source_img"):
            render_reference_card(
                label=entry.get("source", f"Page {ref_page}"),
                img_b64=entry["source_img"],
                page_b64=entry["source_b64"],
                key=entry.get("id", "k0"),
            )



    # Persist
    st.session_state.messages.append(entry)
    st.session_state.pending_input = None
    st.session_state.waiting_for_response = False
