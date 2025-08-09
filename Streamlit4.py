import os
import io
import pickle
import base64
from io import BytesIO

import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.documents import Document
from llama_cloud_services import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from gdrive_utils import get_drive_service, get_all_pdfs, download_pdf


# ============================== Setup ==============================
load_dotenv()
st.set_page_config(page_title="Valuation RAG Chatbot", layout="wide")

# Session state
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


# ====================== Helpers ======================
def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def format_chat_history(messages):
    lines = []
    for m in messages:
        speaker = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines)


# ================== Build index & images (cached) ==================
@st.cache_resource(show_spinner="📦 Processing & indexing PDF…")
def build_index_and_images(pdf_bytes: bytes, file_name: str):
    # 1) Save PDF
    os.makedirs("uploaded", exist_ok=True)
    pdf_path = os.path.join("uploaded", file_name)
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    # 2) Page images
    doc = fitz.open(pdf_path)
    page_images = {i + 1: Image.open(io.BytesIO(page.get_pixmap(dpi=300).tobytes("png")))
                   for i, page in enumerate(doc)}
    doc.close()

    # 3) Parse with LlamaParse
    parser = LlamaParse(api_key=os.environ["LLAMA_CLOUD_API_KEY"], num_workers=4)
    result = parser.parse(pdf_path)
    pages = []
    for pg in result.pages:
        cleaned = [l for l in pg.md.splitlines() if l.strip() and l.lower() != "null"]
        text = "\n".join(cleaned)
        if text:
            pages.append(Document(page_content=text, metadata={"page_number": pg.page}))

    # 4) Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=3300, chunk_overlap=0)
    chunks = splitter.split_documents(pages)
    for idx, c in enumerate(chunks):
        c.metadata["chunk_id"] = idx + 1

    # 5) Embed & index
    embedder = CohereEmbeddings(
        model="embed-english-v3.0",
        user_agent="langchain",
        cohere_api_key=st.secrets["COHERE_API_KEY"]
    )
    vs = FAISS.from_documents(chunks, embedder)

    # 6) Persist (optional)
    store = os.path.join("vectorstore", file_name)
    os.makedirs(store, exist_ok=True)
    vs.save_local(store, index_name="faiss")
    with open(os.path.join(store, "metadata.pkl"), "wb") as mf:
        pickle.dump([c.metadata for c in chunks], mf)

    # 7) Retriever + reranker
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


# ============================ Prompt ===============================
prompt = PromptTemplate(
    template="""
You are a financial-data extraction assistant.

**IMPORTANT CONDITIONAL FOLLOW-UP**  
🛎️ After you answer the user’s question (using steps 1–4), **only if** there is still **unused** relevant report content, **ask**:  
  “Would you like more detail on [X]?”  
Otherwise, **do not** ask any follow-up.

**Use ONLY what appears under “Context”.**

### How to answer
1. **Single value questions**  
   • Find the row + column that match the user's words.  
   • Return the answer in a **short, clear sentence** using the exact number from the context.  
   • **Do NOT repeat the metric name or company name** unless the user asks.

2. **Table questions**  
   • Return the full table **with its header row** in GitHub-flavoured markdown.

3. **Valuation method / theory / reasoning questions**
   • Combine and synthesize relevant information across all chunks.  
   • Pay special attention to how **weights are distributed** (e.g., “50% DCF, 25% EBITDA, 25% SDE”).  
   • Prefer detailed breakdowns if available and mention corresponding **dollar values**.  
   • If Market approach has sub-methods (EBITDA/SDE), **explicitly extract their individual weights and values**.

4. **Theory/textual question**  
   • Try to return an explanation **based on the context**.

If you still cannot see the answer, reply: **“Hmm, I am not sure. Are you able to rephrase your question?”**

---
Context:
{context}

---
Question: {question}
Answer:""",
    input_variables=["context", "question"]
)

wrapped_prompt = PromptTemplate(
    template=prompt.template + """
Conversation so far:
{chat_history}
""",
    input_variables=["chat_history", "context", "question"]
)


# ===================== RAG + rerank (returns refs) =====================
def answer_with_rag_and_rerank(question: str, retriever, page_images):
    """
    Returns: (answer_text, refs)
      refs = list of {"label": "Page X", "b64": "<base64 png>"} (may be empty)
    """
    # 1) Retrieve
    docs = retriever.get_relevant_documents(question)
    ctx = "\n\n".join(d.page_content for d in docs)

    # 2) History (last 10)
    history_to_use = st.session_state.messages[-10:]
    chat_hist = format_chat_history(history_to_use)

    # 3) Initial answer
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    ans = llm.invoke(wrapped_prompt.invoke({
        "chat_history": chat_hist,
        "context": ctx,
        "question": question
    })).content

    # 4) Rerank vs answer text, keep top-3 refs
    refs = []
    if docs:
        texts = [d.page_content for d in docs]
        try:
            emb_query = CohereEmbeddings(
                model="embed-english-v3.0",
                user_agent="langchain",
                cohere_api_key=st.secrets["COHERE_API_KEY"]
            ).embed_query(ans)
            chunk_embs = CohereEmbeddings(
                model="embed-english-v3.0",
                user_agent="langchain",
                cohere_api_key=st.secrets["COHERE_API_KEY"]
            ).embed_documents(texts)
            sims = cosine_similarity([emb_query], chunk_embs)[0]
            ranked = sorted(list(zip(docs, sims)), key=lambda x: x[1], reverse=True)
            top3 = [d for d, _ in ranked[:3]]
        except Exception:
            top3 = [docs[0]]

        for d in top3:
            p = d.metadata.get("page_number")
            img = page_images.get(p)
            if p and img:
                refs.append({"label": f"Page {p}", "b64": pil_to_base64(img)})

    return ans, refs


# ================== Sidebar: Google Drive loader ==================
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


# ============================ Main UI =============================
st.title("Underwriting Agent")

# Load file (Drive or uploader)
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

# Reset chat if a new PDF is selected
if st.session_state.get("last_processed_pdf") != up.name:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."},
        {"role": "assistant", "content": "What can I help you with?"}
    ]
    st.session_state["last_processed_pdf"] = up.name

# Build (cached)
pdf_bytes = up.getvalue()
retriever, page_images = build_index_and_images(pdf_bytes, up.name)


# ===================== Display history (once) =====================
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # if previous assistant messages had refs, show dropdown
        if msg.get("refs"):
            with st.expander("📘 Reference (click to open)"):
                labels = [r["label"] for r in msg["refs"]]
                choice = st.selectbox("Page", labels, key=f"refsel_hist_{idx}")
                chosen = next(r for r in msg["refs"] if r["label"] == choice)
                st.image(BytesIO(base64.b64decode(chosen["b64"])),
                         use_container_width=True, caption=choice)


# ============================== Input =============================
user_q = st.chat_input("Message")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    st.session_state.pending_input = user_q
    st.session_state.waiting_for_response = True


# ====== While waiting: show spinner, compute, then replace it ======
if st.session_state.waiting_for_response:
    # 1) placeholder
    response_placeholder = st.empty()
    with response_placeholder.container():
        with st.chat_message("assistant"):
            st.markdown("🧠 *Thinking...*")

    # 2) compute
    q = st.session_state.pending_input
    try:
        answer, refs = answer_with_rag_and_rerank(q, retriever, page_images)
    except Exception as e:
        answer, refs = f"❌ Error: {e}", []

    # 3) replace placeholder (no rerun)
    with response_placeholder.container():
        with st.chat_message("assistant"):
            st.markdown(answer)
            if refs:
                with st.expander("📘 Reference (click to open)"):
                    labels = [r["label"] for r in refs]
                    choice = st.selectbox("Page", labels, key="refsel_newmsg")
                    chosen = next(r for r in refs if r["label"] == choice)
                    st.image(BytesIO(base64.b64decode(chosen["b64"])),
                             use_container_width=True, caption=choice)

    # 4) persist to history for next render
    entry = {"role": "assistant", "content": answer}
    if refs:
        entry["refs"] = refs
    st.session_state.messages.append(entry)

    # 5) reset flags
    st.session_state.pending_input = None
    st.session_state.waiting_for_response = False
