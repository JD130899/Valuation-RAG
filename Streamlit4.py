import os
import io
import pickle
import base64

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

# ================= Builder =================
@st.cache_resource(show_spinner="📦 Processing & indexing PDF…")
def build_retriever_from_pdf(pdf_bytes: bytes, file_name: str):
    os.makedirs("uploaded", exist_ok=True)
    pdf_path = os.path.join("uploaded", file_name)
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    # Page images (moderate dpi for speed)
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
                # reset convo for new doc
                st.session_state.messages = [
                    {"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."},
                    {"role": "assistant", "content": "What can I help you with?"}
                ]
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
    pdf_bytes = up.getvalue()
    st.session_state.retriever, st.session_state.page_images = build_retriever_from_pdf(pdf_bytes, up.name)
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."},
        {"role": "assistant", "content": "What can I help you with?"}
    ]
    st.session_state.last_processed_pdf = up.name

st.success("✅ RAG index ready.")

# ================= Chat UI CSS =================
st.markdown("""
<style>
.user-bubble {background:#007bff;color:#fff;padding:8px;border-radius:8px;max-width:60%;float:right;margin:4px;}
.assistant-bubble {background:#1e1e1e;color:#fff;padding:8px;border-radius:8px;max-width:60%;float:left;margin:4px;}
.clearfix::after {content:"";display:table;clear:both;}
/* Reference chip + panel */
.ref{ display:block; width:60%; max-width:900px; margin:6px 0 12px 8px; }
.ref summary{
  display:inline-flex; align-items:center; gap:8px; cursor:pointer; list-style:none; outline:none;
  background:#0f172a; color:#e2e8f0; border:1px solid #334155; border-radius:10px; padding:6px 10px;
}
.ref summary::before{ content:"▶"; font-size:12px; line-height:1; }
.ref[open] summary::before{ content:"▼"; }
.ref[open] summary{ border-bottom-left-radius:0; border-bottom-right-radius:0; }
.ref .panel{
  background:#0f172a; color:#e2e8f0; border:1px solid #334155; border-top:none;
  border-radius:0 10px 10px 10px; padding:10px; margin-top:0; box-shadow:0 6px 20px rgba(0,0,0,.25);
}
.ref .panel img{ width:100%; height:auto; border-radius:8px; display:block; }
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

# ================= History =================
for msg in st.session_state.messages:
    cls = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    st.markdown(f"<div class='{cls} clearfix'>{msg['content']}</div>", unsafe_allow_html=True)

    if msg.get("source_img"):
        title = msg.get("source")
        label = f"Reference: {title}" if title else "Reference"
        st.markdown(
            f"""
            <details class="ref">
              <summary>📘 {label}</summary>
              <div class="panel">
                <img src="data:image/png;base64,{msg['source_img']}" alt="reference" loading="lazy"/>
              </div>
            </details>
            <div class="clearfix"></div>
            """,
            unsafe_allow_html=True
        )

# ================= Input =================
user_q = st.chat_input("Type your question here…")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    st.session_state.pending_input = user_q
    st.session_state.waiting_for_response = True

# ================= Answer (single-pass, no rerun) =================
if st.session_state.waiting_for_response:
    block = st.empty()
    with block.container():
        st.markdown("<div class='assistant-bubble clearfix'>🧠 <em>Thinking...</em></div>", unsafe_allow_html=True)

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

    # Reference selection
    entry = {"role": "assistant", "content": answer}
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
                    template=(
                        "Given a user question and 3 candidate context chunks, return the number (1-3) "
                        "of the chunk that best answers it.\n\n"
                        "Question:\n{question}\n\nChunk 1:\n{chunk1}\n\nChunk 2:\n{chunk2}\n\nChunk 3:\n{chunk3}\n\nBest Chunk Number:\n"
                    ),
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
                if ref_page and ref_img_b64:
                    entry["source"] = f"Page {ref_page}"
                    entry["source_img"] = ref_img_b64
    except Exception as e:
        st.info(f"ℹ️ Reference selection skipped: {e}")

    # Final render (no rerun)
    with block.container():
        st.markdown(f"<div class='assistant-bubble clearfix'>{answer}</div>", unsafe_allow_html=True)
        if ref_page and ref_img_b64:
            st.markdown(
                f"""
                <details class="ref">
                  <summary>📘 Reference: Page {ref_page}</summary>
                  <div class="panel">
                    <img src="data:image/png;base64,{ref_img_b64}" alt="reference" loading="lazy"/>
                  </div>
                </details>
                <div class="clearfix"></div>
                """,
                unsafe_allow_html=True
            )

    # Persist
    st.session_state.messages.append(entry)
    st.session_state.pending_input = None
    st.session_state.waiting_for_response = False
