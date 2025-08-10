import os
import io
import pickle
import base64
import streamlit as st
from PIL import Image
import fitz  # PyMuPDF
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity

# --- LangChain / RAG deps ---
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
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Chat QA", layout="wide")
st.title("🧠 Ask Me Anything (Chat Style)")

st.markdown("""
<style>
.user-bubble {
  background:#007bff; color:#fff; padding:10px 12px; border-radius:12px;
  max-width:60%; float:right; margin:6px 0; line-height:1.4;
  box-shadow:0 2px 8px rgba(0,0,0,.15);
}
.assistant-bubble {
  background:#1e1e1e; color:#fff; padding:10px 12px; border-radius:12px;
  max-width:60%; float:left; margin:6px 0; line-height:1.4;
  box-shadow:0 2px 8px rgba(0,0,0,.15);
}
.clearfix::after { content:""; display:table; clear:both; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Collapsed chip stays compact; opened panel is wide */
.ref{ display:block; width:60%; max-width:900px; margin:6px 0 12px 8px; }

/* summary chip */
.ref summary{
  display:inline-flex; align-items:center; gap:8px;
  background:#0f172a; color:#e2e8f0;
  border:1px solid #334155; border-radius:10px;
  padding:6px 10px; cursor:pointer; list-style:none; outline:none;
}

/* caret icon swap */
.ref summary::before{ content:"▶"; font-size:12px; line-height:1; }
.ref[open] summary::before{ content:"▼"; }

/* opened panel */
.ref[open] summary{ border-bottom-left-radius:0; border-bottom-right-radius:0; }
.ref .panel{
  background:#0f172a; color:#e2e8f0; border:1px solid #334155; border-top:none;
  border-radius:0 10px 10px 10px; padding:10px; margin-top:0;
  box-shadow:0 6px 20px rgba(0,0,0,.25);
}

/* image fills the panel nicely */
.ref .panel img{ width:100%; height:auto; border-radius:8px; display:block; }
</style>
""", unsafe_allow_html=True)




# ======== Helpers ========
def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

# ======== Minimal RAG builder (returns retriever + page images) ========
@st.cache_resource(show_spinner="📦 Processing & indexing PDF…")
def build_retriever_from_pdf(pdf_bytes: bytes, file_name: str):
    # 1) write pdf to disk
    os.makedirs("uploaded", exist_ok=True)
    pdf_path = os.path.join("uploaded", file_name)
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    # 1b) Extract page images (use lighter DPI to reduce UI churn)
    doc = fitz.open(pdf_path)
    page_images = {
        i + 1: Image.open(io.BytesIO(page.get_pixmap(dpi=180).tobytes("png")))  # 180dpi to keep it snappy
        for i, page in enumerate(doc)
    }
    doc.close()

    # 2) Parse with LlamaParse
    parser = LlamaParse(api_key=os.environ["LLAMA_CLOUD_API_KEY"], num_workers=4)
    result = parser.parse(pdf_path)
    pages = []
    for pg in result.pages:
        cleaned = [l for l in pg.md.splitlines() if l.strip() and l.lower() != "null"]
        text = "\n".join(cleaned)
        if text:
            pages.append(Document(page_content=text, metadata={"page_number": pg.page}))

    # 3) Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=3300, chunk_overlap=0)
    chunks = splitter.split_documents(pages)
    for idx, c in enumerate(chunks):
        c.metadata["chunk_id"] = idx + 1

    # 4) Embed + FAISS
    embedder = CohereEmbeddings(
        model="embed-english-v3.0",
        user_agent="langchain",
        cohere_api_key=st.secrets["COHERE_API_KEY"]
    )
    vs = FAISS.from_documents(chunks, embedder)

    # 5) Persist (optional)
    store = os.path.join("vectorstore", file_name)
    os.makedirs(store, exist_ok=True)
    vs.save_local(store, index_name="faiss")
    with open(os.path.join(store, "metadata.pkl"), "wb") as mf:
        pickle.dump([c.metadata for c in chunks], mf)

    # 6) Retriever + reranker
    reranker = CohereRerank(
        model="rerank-english-v3.0",
        user_agent="langchain",
        cohere_api_key=st.secrets["COHERE_API_KEY"],
        top_n=20
    )
    retriever = ContextualCompressionRetriever(
        base_retriever=vs.as_retriever(search_type="mmr",
                                       search_kwargs={"k": 50, "fetch_k": 100, "lambda_mult": 0.9}),
        base_compressor=reranker
    )
    return retriever, page_images

# ======== State =========
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me anything about AI, ML, DL, or GenAI 🤖"}]
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None
if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "page_images" not in st.session_state:
    st.session_state.page_images = {}

# ======== Upload PDF once to build retriever ========
uploaded = st.file_uploader("Upload a valuation report PDF (required for RAG)", type="pdf")
if uploaded and st.session_state.retriever is None:
    st.session_state.retriever, st.session_state.page_images = build_retriever_from_pdf(
        uploaded.getvalue(), uploaded.name
    )

if st.session_state.retriever is None:
    st.info("⬆️ Please upload a PDF to enable RAG answers.")
else:
    st.success("✅ RAG index ready.")

# ======== Input ========
user_input = st.chat_input("Type your question here...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.pending_input = user_input
    st.session_state.waiting_for_response = True

# ======== History (right/left bubbles) + Reference expander (no popover) ========
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



# ======== Answer (with ONLY RAG flow) — single-pass render, no rerun ========
if st.session_state.waiting_for_response:
    block = st.empty()
    with block.container():
        st.markdown("<div class='assistant-bubble clearfix'>🧠 <em>Thinking...</em></div>", unsafe_allow_html=True)

    q = st.session_state.pending_input or ""
    ctx, docs = "", []
    if st.session_state.retriever:
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

    # --- Reference selection BEFORE final render (same pass, no rerun) ---
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

            best_doc = None
            if len(top3) >= 3:
                ranking_prompt = PromptTemplate(
                    template="""
Given a user question and 3 candidate context chunks, return the number (1-3) of the chunk that best answers it.

Question:
{question}

Chunk 1:
{chunk1}

Chunk 2:
{chunk2}

Chunk 3:
{chunk3}

Best Chunk Number:
""",
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
            if best_doc is None:
                best_doc = top3[0] if top3 else (ranked[0][0] if ranked else None)

            if best_doc is not None:
                ref_page = best_doc.metadata.get("page_number")
                img = st.session_state.page_images.get(ref_page)
                ref_img_b64 = pil_to_base64(img) if img else None
                if ref_page and ref_img_b64:
                    entry["source"] = f"Page {ref_page}"
                    entry["source_img"] = ref_img_b64
    except Exception as e:
        st.info(f"ℹ️ Reference selection skipped: {e}")

    # --- Final render in the SAME placeholder (no rerun, no fade) ---
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




    # Persist to history
    st.session_state.messages.append(entry)
    st.session_state.pending_input = None
    st.session_state.waiting_for_response = False
