import streamlit as st
import openai
import os
import io
import time
import pickle
import base64
from dotenv import load_dotenv

# RAG deps
import fitz  # PyMuPDF
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.documents import Document
from llama_cloud_services import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

# Optional (for LLM ranking helper, but we keep core send/receive logic the same)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from gdrive_utils import get_drive_service, get_all_pdfs, download_pdf

# ───────────────────────────── Setup ─────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = st.secrets.get("COHERE_API_KEY")

st.set_page_config(page_title="Chat QA", layout="wide")
st.title("🧠 Ask Me Anything (Chat Style)")

# ─────────────────────── Init state (chat core) ───────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about AI, ML, DL, or GenAI 🤖"}
    ]
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None
if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False

# ─────────────────────── Extra state for RAG ───────────────────────
if "last_synced_file_id" not in st.session_state:
    st.session_state.last_synced_file_id = None
if "uploaded_file_from_drive" not in st.session_state:
    st.session_state.uploaded_file_from_drive = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "last_processed_pdf" not in st.session_state:
    st.session_state.last_processed_pdf = None

# ─────────────────────── CACHING BUILDER (RAG) ───────────────────────
@st.cache_resource(show_spinner="📦 Processing & indexing PDF…")
def build_index_and_images(pdf_bytes: bytes, file_name: str):
    # 1) Save PDF
    os.makedirs("uploaded", exist_ok=True)
    pdf_path = os.path.join("uploaded", file_name)
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    # 2) Extract page images
    doc = fitz.open(pdf_path)
    page_images = {i + 1: Image.open(io.BytesIO(page.get_pixmap(dpi=300).tobytes("png"))) for i, page in enumerate(doc)}
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
        cohere_api_key=COHERE_API_KEY,
    )
    vs = FAISS.from_documents(chunks, embedder)

    # 6) Persist
    store = os.path.join("vectorstore", file_name)
    os.makedirs(store, exist_ok=True)
    vs.save_local(store, index_name="faiss")
    with open(os.path.join(store, "metadata.pkl"), "wb") as mf:
        pickle.dump([c.metadata for c in chunks], mf)

    # 7) Retriever + reranker
    reranker = CohereRerank(
        model="rerank-english-v3.0",
        user_agent="langchain",
        cohere_api_key=COHERE_API_KEY,
        top_n=20,
    )
    retriever = ContextualCompressionRetriever(
        base_retriever=vs.as_retriever(search_type="mmr", search_kwargs={"k": 50, "fetch_k": 100, "lambda_mult": 0.9}),
        base_compressor=reranker,
    )

    return retriever, page_images


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

# ─────────────────────── Sidebar: Google Drive loader ───────────────────────
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
                # Reset chat for new doc context
                st.session_state.messages = [
                    {"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."},
                    {"role": "assistant", "content": "What can I help you with?"},
                ]
else:
    st.sidebar.warning("📭 No PDFs found in Drive.")

# ─────────────────────── Main UI hint for current doc ───────────────────────
up = None
if st.session_state.get("uploaded_file_from_drive"):
    st.markdown(
        f"<div style='background:#1f2c3a; padding:8px; border-radius:8px; color:#fff;'>✅ <b>Using synced file:</b> {st.session_state.uploaded_file_name}</div>",
        unsafe_allow_html=True,
    )
    up = io.BytesIO(st.session_state.uploaded_file_from_drive)
    up.name = st.session_state.uploaded_file_name
else:
    st.info("Optionally, upload a valuation PDF to enable RAG (context-augmented answers).")
    up = st.file_uploader("Upload a valuation report PDF", type="pdf")

# If a brand-new PDF is selected, reset opening messages to valuation assistant style
if up is not None and st.session_state.get("last_processed_pdf") != up.name:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."},
        {"role": "assistant", "content": "What can I help you with?"},
    ]
    st.session_state.last_processed_pdf = up.name

# Build retriever (if a PDF is present); otherwise keep RAG disabled
retriever, page_images = None, {}
if up is not None:
    pdf_bytes = up.getvalue()
    if pdf_bytes:
        retriever, page_images = build_index_and_images(pdf_bytes, up.name)

# ─────────────────────── Chat input (keep same logic) ───────────────────────
user_input = st.chat_input("Type your question here...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.pending_input = user_input
    st.session_state.waiting_for_response = True

# ─────────────────────── Display history (enhanced: show sources if present) ───────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Optional source image (from RAG best chunk)
        if msg.get("source_img"):
            with st.popover(msg.get("source", "📘 Reference")):
                data = base64.b64decode(msg["source_img"]) if isinstance(msg["source_img"], str) else msg["source_img"]
                st.image(Image.open(io.BytesIO(data)), use_container_width=True)

# ─────────────────────── While waiting: Thinking… then answer ───────────────────────
if st.session_state.waiting_for_response:
    response_placeholder = st.empty()
    with response_placeholder.container():
        with st.chat_message("assistant"):
            st.markdown("🧠 *Thinking...*")

    # —— Build RAG-aware messages (but keep same send/receive call) ——
    messages_for_api = list(st.session_state.messages)

    # If we have a retriever and the last turn is a user question, augment with context
    best_source_entry = {}
    try:
        if retriever is not None and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            q = st.session_state.messages[-1]["content"]
            docs = retriever.get_relevant_documents(q)
            ctx = "\n\n".join(d.page_content for d in docs)

            # Prepend a system message instructing the model to only use provided context
            system_rules = (
                "You are a financial-data extraction assistant.\n\n"
                "Use ONLY what appears under 'Context'.\n\n"
                "How to answer:\n"
                "1) Single value questions: find the exact cell and return a short sentence with the exact number.\n"
                "2) Table questions: return the full table with header in GitHub-flavored markdown.\n"
                "3) Valuation method/theory: combine relevant parts, include weights and corresponding dollar values, and break down sub-methods.\n"
                "4) If the answer isn't in Context, say: 'Hmm, I am not sure. Are you able to rephrase your question?'\n\n"
                f"Context:\n{ctx}"
            )

            messages_for_api = ([{"role": "system", "content": system_rules}] + messages_for_api)

            # Lightweight 3-chunk reranking to pick a reference image
            texts = [d.page_content for d in docs]
            if texts:
                emb_query = CohereEmbeddings(
                    model="embed-english-v3.0", user_agent="langchain", cohere_api_key=COHERE_API_KEY
                ).embed_query(q)
                chunk_embs = CohereEmbeddings(
                    model="embed-english-v3.0", user_agent="langchain", cohere_api_key=COHERE_API_KEY
                ).embed_documents(texts)
                sims = cosine_similarity([emb_query], chunk_embs)[0]
                ranked = sorted(list(zip(docs, sims)), key=lambda x: x[1], reverse=True)
                top3 = [d for d, _ in ranked[:3]] if ranked else []

                # Use LLM to pick best of top3 (keeps UX from earlier app)
                if len(top3) >= 1:
                    try:
                        ranking_prompt = PromptTemplate(
                            template=(
                                "Given a user question and 3 candidate context chunks, return the number (1-3) of the chunk that best answers it.\n\n"
                                "Question:\n{question}\n\nChunk 1:\n{chunk1}\n\nChunk 2:\n{chunk2}\n\nChunk 3:\n{chunk3}\n\nBest Chunk Number:"
                            ),
                            input_variables=["question", "chunk1", "chunk2", "chunk3"],
                        )
                        llm = ChatOpenAI(model="gpt-4o", temperature=0)
                        pick = llm.invoke(
                            ranking_prompt.invoke(
                                {
                                    "question": q,
                                    "chunk1": top3[0].page_content if len(top3) > 0 else "",
                                    "chunk2": top3[1].page_content if len(top3) > 1 else "",
                                    "chunk3": top3[2].page_content if len(top3) > 2 else "",
                                }
                            )
                        ).content.strip()
                        best_doc = top3[int(pick) - 1] if pick.isdigit() and 1 <= int(pick) <= len(top3) else top3[0]
                    except Exception:
                        best_doc = top3[0]

               
    except Exception as rag_e:
        # Fail open: if RAG fails, continue with vanilla chat
        st.sidebar.warning(f"RAG pipeline issue: {rag_e}")

    # —— Call the model (unchanged call pattern) ——
    try:
        model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        response = openai.chat.completions.create(
            model=model_name,
            messages=messages_for_api,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"❌ Error: {e}"

    # Replace the spinner with actual content
    with response_placeholder.container():
        with st.chat_message("assistant"):
            st.markdown(answer)
    

    # Append the assistant message back into the same chat history structure
    new_assistant_msg = {"role": "assistant", "content": answer}
    new_assistant_msg.update(best_source_entry)
    st.session_state.messages.append(new_assistant_msg)

    # Reset pending flags (keep exact UX behavior)
    st.session_state.pending_input = None
    st.session_state.waiting_for_response = False
