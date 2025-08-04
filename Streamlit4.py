import streamlit as st
import os
import re
import fitz  # PyMuPDF
import openai
import base64
import io
import time
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from langchain_core.documents import Document
from llama_cloud_services import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.vectorstores import FAISS
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from gdrive_utils import get_drive_service, get_all_pdfs, download_pdf
import pickle

load_dotenv()
st.set_page_config(page_title="Valuation RAG Chatbot", layout="wide")
openai.api_key = os.environ["OPENAI_API_KEY"]

# === SIDEBAR: List all Drive PDFs ===
service = get_drive_service()
pdf_files = get_all_pdfs(service)
pdf_names = [file["name"] for file in pdf_files]
pdf_lookup = {file["name"]: file["id"] for file in pdf_files}

selected_pdf = st.sidebar.selectbox("📄 Select a PDF to load", pdf_names)
if st.sidebar.button("Load Selected PDF"):
    file_id = pdf_lookup[selected_pdf]
    file_name = selected_pdf
    faiss_dir = os.path.join("vectorstore", file_name)
    index_file = os.path.join(faiss_dir, "faiss.index")
    metadata_file = os.path.join(faiss_dir, "metadata.pkl")

    if os.path.exists(index_file) and os.path.exists(metadata_file):
        st.success(f"✅ Already parsed: {file_name}. Ready to ask questions.")
        with open(os.path.join("uploaded", file_name), "rb") as f:
            st.session_state["uploaded_file_from_drive"] = f.read()
        st.session_state["uploaded_file_name"] = file_name
        st.session_state.last_uploaded = file_name
        st.rerun()
    else:
        pdf_path = download_pdf(service, file_id, file_name)
        if pdf_path:
            with open(pdf_path, "rb") as f:
                st.session_state["uploaded_file_from_drive"] = f.read()
            st.session_state["uploaded_file_name"] = file_name
            st.session_state.last_uploaded = file_name
            st.rerun()

# === Existing sync from latest PDF ===
if st.sidebar.button("📡 Sync from Google Drive (Latest)"):
    from gdrive_utils import get_latest_pdf
    latest = get_latest_pdf(service)
    if latest:
        file_id = latest["id"]
        file_name = latest["name"]
        pdf_path = download_pdf(service, file_id, file_name)
        st.success(f"✅ Downloaded: {file_name}")
        with open(pdf_path, "rb") as f:
            st.session_state["uploaded_file_from_drive"] = f.read()
        st.session_state["uploaded_file_name"] = file_name
        st.session_state.last_uploaded = file_name
        st.rerun()
    else:
        st.warning("No PDF found in Google Drive folder.")

# === Streamlit UI ===
st.title("Underwriting Agent")
if "uploaded_file_from_drive" not in st.session_state:
    uploaded_file = st.file_uploader("Upload a valuation report PDF", type="pdf")
else:
    st.info(f"✅ Using synced file from Drive: {st.session_state['uploaded_file_name']}")
    uploaded_file = io.BytesIO(st.session_state["uploaded_file_from_drive"])
    uploaded_file.name = st.session_state["uploaded_file_name"]

# === Helper: Convert PIL to base64 ===
def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

# === Proceed only if file is present ===
if uploaded_file is not None:
    file_name = uploaded_file.name
    FAISS_FOLDER = os.path.join("vectorstore", file_name)
    index_file = os.path.join(FAISS_FOLDER, "faiss.index")
    metadata_file = os.path.join(FAISS_FOLDER, "metadata.pkl")

    if "retriever" in st.session_state and st.session_state.get("retriever_for") == file_name:
        pass
    else:
        with st.spinner("Processing PDF..."):
            os.makedirs("uploaded", exist_ok=True)
            PDF_PATH = os.path.join("uploaded", file_name)
            with open(PDF_PATH, "wb") as f:
                f.write(uploaded_file.getbuffer())

            EXTRACTED_FOLDER = os.path.join(os.getcwd(), "extracted")
            os.makedirs(EXTRACTED_FOLDER, exist_ok=True)
            st.session_state.last_uploaded = file_name

            doc = fitz.open(PDF_PATH)
            all_images = [Image.open(io.BytesIO(page.get_pixmap(dpi=300).tobytes("png"))) for page in doc]
            st.session_state.page_images = {i + 1: img for i, img in enumerate(all_images)}
            doc.close()

            st.session_state.initialized = True

            def parse_pdf():
                parser = LlamaParse(api_key=os.environ["LLAMA_CLOUD_API_KEY"], num_workers=4)
                result = parser.parse(PDF_PATH)
                pages = []
                for page in result.pages:
                    page_num = page.page
                    replacement_path = os.path.join(EXTRACTED_FOLDER, f"page_{page_num}.txt")
                    if os.path.exists(replacement_path):
                        with open(replacement_path) as f:
                            content = f.read().strip()
                    else:
                        content = page.md.strip()
                        cleaned = [l for l in content.splitlines() if l.strip() and l.strip().lower() != "null"]
                        content = "\n".join(cleaned)
                    if content:
                        pages.append(Document(page_content=content, metadata={"page_number": page_num}))
                return pages

            docs = parse_pdf()
            chunks = RecursiveCharacterTextSplitter(chunk_size=3300, chunk_overlap=0).split_documents(docs)
            for i, doc in enumerate(chunks):
                doc.metadata["chunk_id"] = i + 1

            os.makedirs(FAISS_FOLDER, exist_ok=True)

            embed = CohereEmbeddings(
                model="embed-english-v3.0",
                user_agent="langchain",
                cohere_api_key=st.secrets["COHERE_API_KEY"]
            )

            if os.path.exists(index_file) and os.path.exists(metadata_file):
                with open(metadata_file, "rb") as f:
                    stored_metadatas = pickle.load(f)
                vs = FAISS.load_local(FAISS_FOLDER, embed, index_name="faiss")
            else:
                texts = [doc.page_content for doc in chunks]
                metadatas = [doc.metadata for doc in chunks]
                embeddings = []
                for i, text in enumerate(texts):
                    try:
                        emb = embed.embed_query(text)
                        embeddings.append(emb)
                    except Exception as e:
                        st.error(f"Embedding failed on chunk {i}: {e}")
                        embeddings.append([0.0] * 1024)
                    time.sleep(0.5)

                vs = FAISS.from_documents(chunks, embed)
                vs.save_local(FAISS_FOLDER, index_name="faiss")
                with open(metadata_file, "wb") as f:
                    pickle.dump(metadatas, f)

            base_ret = vs.as_retriever(search_type="mmr", search_kwargs={"k": 50, "fetch_k": 100, "lambda_mult": 0.9})
            reranker = CohereRerank(
                model="rerank-english-v3.0",
                user_agent="langchain",
                cohere_api_key=st.secrets["COHERE_API_KEY"],
                top_n=20
            )

            st.session_state.retriever = ContextualCompressionRetriever(
                base_retriever=base_ret,
                base_compressor=reranker
            )
            st.session_state.reranker = reranker
            st.session_state["retriever_for"] = file_name
            # === Begin Chat Interface ===
            prompt = PromptTemplate(
                template="""
                You are a financial-data extraction assistant.
                Context:
                {context}
                ---
                Question: {question}
                Answer:""",
                input_variables=["context", "question"]
            )
            
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."},
                    {"role": "assistant", "content": "What can I help you with?"}
                ]
            
            user_question = st.chat_input("Message")
            
            # Render chat history
            for msg in st.session_state.messages:
                role_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
                st.markdown(f"<div class='{role_class} clearfix'>{msg['content']}</div>", unsafe_allow_html=True)
                if msg["role"] == "assistant" and msg.get("source_img"):
                    with st.popover(f"📘 Reference:"):
                        st.image(Image.open(io.BytesIO(base64.b64decode(msg["source_img"]))), caption=msg["source"], use_container_width=True)
            
            # Handle new question
            if user_question:
                st.session_state.messages.append({"role": "user", "content": user_question})
                st.markdown(f"<div class='user-bubble clearfix'>{user_question}</div>", unsafe_allow_html=True)
                with st.spinner("Thinking..."):
                    retrieved_docs = st.session_state.retriever.get_relevant_documents(user_question)
                    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
                    final_prompt = prompt.invoke({"context": context_text, "question": user_question})
            
                    llm = ChatOpenAI(model="gpt-4o", temperature=0)
                    response = llm.invoke(final_prompt)
            
                    # Source doc matching
                    embed = CohereEmbeddings(model="embed-english-v3.0", user_agent="langchain", cohere_api_key=st.secrets["COHERE_API_KEY"])
                    llm_embedding = embed.embed_query(response.content)
            
                    texts = [doc.page_content for doc in retrieved_docs]
                    chunk_embeddings = embed.embed_documents(texts)
                    similarities = cosine_similarity([llm_embedding], chunk_embeddings)[0]
                    ranked = sorted(
                        [(i, doc, sim) for i, (doc, sim) in enumerate(zip(retrieved_docs, similarities))],
                        key=lambda x: x[2], reverse=True
                    )
            
                    top3_docs = [doc for _, doc, _ in ranked[:3]]
                    top3_chunks = [doc.page_content for doc in top3_docs]
            
                   ranking_prompt = PromptTemplate(template="""
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

  
else:
    st.warning("📄 Please upload or select a PDF to continue.")
    st.stop()
