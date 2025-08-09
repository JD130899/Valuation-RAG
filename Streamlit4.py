import os, io, pickle, base64
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

# ---------------- Setup ----------------
load_dotenv()
st.set_page_config(page_title="Valuation RAG Chatbot", layout="wide")

if "last_synced_file_id" not in st.session_state:
    st.session_state.last_synced_file_id = None
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant","content":"Hi! I am here to answer any questions you may have about your valuation report."},
        {"role":"assistant","content":"What can I help you with?"}
    ]
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None
if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False

@st.cache_resource(show_spinner="📦 Processing & indexing PDF…")
def build_index_and_images(pdf_bytes: bytes, file_name: str):
    os.makedirs("uploaded", exist_ok=True)
    pdf_path = os.path.join("uploaded", file_name)
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    doc = fitz.open(pdf_path)
    page_images = {i+1: Image.open(io.BytesIO(page.get_pixmap(dpi=300).tobytes("png")))
                   for i, page in enumerate(doc)}
    doc.close()

    parser = LlamaParse(api_key=os.environ["LLAMA_CLOUD_API_KEY"], num_workers=4)
    result = parser.parse(pdf_path)
    pages = []
    for pg in result.pages:
        cleaned = [l for l in pg.md.splitlines() if l.strip() and l.lower()!="null"]
        text = "\n".join(cleaned)
        if text:
            pages.append(Document(page_content=text, metadata={"page_number":pg.page}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=3300, chunk_overlap=0)
    chunks = splitter.split_documents(pages)
    for idx, c in enumerate(chunks):
        c.metadata["chunk_id"] = idx+1

    embedder = CohereEmbeddings(
        model="embed-english-v3.0",
        user_agent="langchain",
        cohere_api_key=st.secrets["COHERE_API_KEY"]
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
        cohere_api_key=st.secrets["COHERE_API_KEY"],
        top_n=20
    )
    retriever = ContextualCompressionRetriever(
        base_retriever=vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k":50,"fetch_k":100,"lambda_mult":0.9}
        ),
        base_compressor=reranker
    )
    return retriever, page_images

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

# ---------------- Sidebar (Drive) ----------------
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
                    {"role":"assistant","content":"Hi! I am here to answer any questions you may have about your valuation report."},
                    {"role":"assistant","content":"What can I help you with?"}
                ]
else:
    st.sidebar.warning("📭 No PDFs found in Drive.")

# ---------------- Main ----------------
st.title("Underwriting Agent")

if "uploaded_file_from_drive" in st.session_state:
    st.markdown(
        f"<div style='background:#1f2c3a; padding:8px; border-radius:8px; color:#fff;'>"
        f"✅ <b>Using synced file:</b> {st.session_state.uploaded_file_name}</div>",
        unsafe_allow_html=True
    )
    up = io.BytesIO(st.session_state.uploaded_file_from_drive)
    up.name = st.session_state.uploaded_file_name
else:
    up = st.file_uploader("Upload a valuation report PDF", type="pdf")

if not up:
    st.warning("Please upload or load a PDF to continue.")
    st.stop()

if st.session_state.get("last_processed_pdf") != up.name:
    st.session_state.messages = [
        {"role":"assistant","content":"Hi! I am here to answer any questions you may have about your valuation report."},
        {"role":"assistant","content":"What can I help you with?"}
    ]
    st.session_state["last_processed_pdf"] = up.name

pdf_bytes = up.getvalue()
retriever, page_images = build_index_and_images(pdf_bytes, up.name)

# ---------------- Styles (flex, no floats) ----------------
st.markdown("""
<style>
.chat {display:flex; flex-direction:column; gap:10px;}
.bubble {padding:10px; border-radius:10px; max-width:60%;}
.user-bubble {background:#2b6cb0;color:#fff;align-self:flex-end;}
.assistant-bubble {background:#1e1e1e;color:#fff;align-self:flex-start;}
</style>
""", unsafe_allow_html=True)

def format_chat_history(messages):
    return "\n".join(("User: " if m["role"]=="user" else "Assistant: ")+m["content"] for m in messages)

prompt = PromptTemplate(
    template = """
You are a financial-data extraction assistant.
[snip identical guidance]
---
Context:
{context}
---
Question: {question}
Answer:""",
    input_variables=["context","question"]
)
wrapped_prompt = PromptTemplate(
    template=prompt.template + "\nConversation so far:\n{chat_history}\n",
    input_variables=["chat_history","context","question"]
)

# -------- Input FIRST so question shows immediately --------
user_q = st.chat_input("Message")
if user_q:
    st.session_state.messages.append({"role":"user","content":user_q})
    st.session_state.pending_input = user_q
    st.session_state.waiting_for_response = True

# -------- Chat area (single container + stable placeholder) --------
chat_area = st.container()
with chat_area:
    st.markdown("<div class='chat'>", unsafe_allow_html=True)
    for i, msg in enumerate(st.session_state.messages):
        cls = "user-bubble" if msg["role"]=="user" else "assistant-bubble"
        st.markdown(f"<div class='bubble {cls}'>{msg['content']}</div>", unsafe_allow_html=True)
        if msg.get("source_img"):
            # Stable key so widgets don't remount → avoids layout jump
            with st.expander("📘 Reference", expanded=False, key=f"ref_{i}"):
                st.image(Image.open(io.BytesIO(base64.b64decode(msg["source_img"]))),
                         caption=msg.get("source"), use_container_width=True)

    reply_placeholder = st.empty()   # reserved spot for assistant reply
    st.markdown("</div>", unsafe_allow_html=True)

# -------- Answer (no st.spinner overlay) --------
if st.session_state.waiting_for_response:
    with reply_placeholder.container():
        st.markdown("<div class='bubble assistant-bubble'>🧠 <i>Thinking…</i></div>", unsafe_allow_html=True)

    q = st.session_state.pending_input

    docs = retriever.get_relevant_documents(q)
    ctx  = "\n\n".join(d.page_content for d in docs)
    history_to_use = st.session_state.messages[-10:]

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    ans = llm.invoke(wrapped_prompt.invoke({
        "chat_history": format_chat_history(history_to_use),
        "context": ctx,
        "question": q
    })).content

    # quick rerank (keep your logic if you want, key is to keep it inside this block)
    texts = [d.page_content for d in docs] if docs else []
    page, b64 = None, None
    if texts:
        emb = CohereEmbeddings(model="embed-english-v3.0", user_agent="langchain",
                               cohere_api_key=st.secrets["COHERE_API_KEY"])
        emb_query  = emb.embed_query(ans)
        chunk_embs = emb.embed_documents(texts)
        sims   = cosine_similarity([emb_query], chunk_embs)[0]
        ranked = sorted(list(zip(docs, sims)), key=lambda x: x[1], reverse=True)
        best   = ranked[0][0]
        page   = best.metadata.get("page_number")
        img    = page_images.get(page)
        b64    = pil_to_base64(img) if img else None

    with reply_placeholder.container():
        st.markdown(f"<div class='bubble assistant-bubble'>{ans}</div>", unsafe_allow_html=True)
        if page and b64:
            with st.expander("📘 Reference", expanded=False, key=f"ref_new_{len(st.session_state.messages)}"):
                st.image(Image.open(io.BytesIO(base64.b64decode(b64))),
                         caption=f"Page {page}", use_container_width=True)

    entry = {"role":"assistant","content":ans}
    if page and b64:
        entry["source"] = f"Page {page}"
        entry["source_img"] = b64
    st.session_state.messages.append(entry)

    st.session_state.pending_input = None
    st.session_state.waiting_for_response = False
