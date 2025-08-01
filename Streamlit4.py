import streamlit as st
import os
import io
import base64
import time
import pickle
import fitz  # PyMuPDF
import openai
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.vectorstores import FAISS
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from gdrive_utils import get_drive_service, get_all_pdfs, download_pdf

load_dotenv()
st.set_page_config(page_title="Valuation RAG Chatbot", layout="wide")

if "last_synced_file_id" not in st.session_state:
    st.session_state.last_synced_file_id = None

openai.api_key = os.environ["OPENAI_API_KEY"]

# === UI ===
st.title("Underwriting Agent")

# === Load from Drive ===
service = get_drive_service()
pdf_files = get_all_pdfs(service)

if pdf_files:
    pdf_names = [f["name"] for f in pdf_files]
    selected_name = st.sidebar.selectbox("📂 Select a PDF from Google Drive", pdf_names)
    selected_file = next(f for f in pdf_files if f["name"] == selected_name)

    if st.sidebar.button("📥 Load Selected PDF"):
        file_id = selected_file["id"]
        file_name = selected_file["name"]

        if file_id == st.session_state.get("last_synced_file_id"):
            st.sidebar.info("✅ Already loaded.")
        else:
            pdf_path = download_pdf(service, file_id, file_name)
            if pdf_path:
                with open(pdf_path, "rb") as f:
                    st.session_state["uploaded_file_from_drive"] = f.read()
                st.session_state["uploaded_file_name"] = file_name
                st.session_state["last_uploaded"] = file_name
                st.session_state["last_synced_file_id"] = file_id
                st.session_state["messages"] = [
                    {"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."},
                    {"role": "assistant", "content": "What can I help you with?"}
                ]
                st.rerun()
else:
    st.sidebar.warning("📭 No PDFs found in Google Drive.")

# === Load cached PDF ===
if "uploaded_file_from_drive" not in st.session_state:
    st.warning("📂 Please select a file from Drive.")
    st.stop()

uploaded_file = io.BytesIO(st.session_state["uploaded_file_from_drive"])
file_name = st.session_state["uploaded_file_name"]
uploaded_file.name = file_name
st.markdown(f"<div style='background-color:#1f2c3a;padding:10px;border-radius:10px;color:white;'>✅ <b>Using synced file from Drive:</b> {file_name}</div>", unsafe_allow_html=True)

# === Load cached retriever and page images ===
FAISS_FOLDER = os.path.join("vectorstore", file_name)
index_file = os.path.join(FAISS_FOLDER, "faiss.index")
metadata_file = os.path.join(FAISS_FOLDER, "metadata.pkl")
images_file = os.path.join("extracted", f"{file_name}_images.pkl")

if not (os.path.exists(index_file) and os.path.exists(metadata_file) and os.path.exists(images_file)):
    st.error("🛑 This file hasn't been preprocessed yet. Please wait for the background sync.")
    st.stop()

embed = CohereEmbeddings(model="embed-english-v3.0", user_agent="langchain", cohere_api_key=st.secrets["COHERE_API_KEY"])
vs = FAISS.load_local(FAISS_FOLDER, embed, index_name="faiss")

base_ret = vs.as_retriever(search_type="mmr", search_kwargs={"k": 50, "fetch_k": 100, "lambda_mult": 0.9})
reranker = CohereRerank(model="rerank-english-v3.0", user_agent="langchain", cohere_api_key=st.secrets["COHERE_API_KEY"], top_n=20)
retriever = ContextualCompressionRetriever(base_retriever=base_ret, base_compressor=reranker)
st.session_state.retriever = retriever
st.session_state.reranker = reranker
st.session_state["retriever_for"] = file_name

with open(images_file, "rb") as f:
    images = pickle.load(f)
st.session_state.page_images = {i + 1: img for i, img in enumerate(images)}

# === Prompt Template ===
prompt = PromptTemplate(
    template="""
    You are a financial-data extraction assistant.
    ...
    Context:
    {context}

    ---
    Question: {question}
    Answer:""",
    input_variables=["context", "question"]
)

def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

# === Chat ===
user_question = st.chat_input("Message")

for msg in st.session_state.messages:
    role_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    st.markdown(f"<div class='{role_class} clearfix'>{msg['content']}</div>", unsafe_allow_html=True)
    if msg["role"] == "assistant" and msg.get("source_img"):
        with st.popover("📘 Reference:"):
            st.image(Image.open(io.BytesIO(base64.b64decode(msg["source_img"]))), caption=msg["source"], use_container_width=True)

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    st.markdown(f"<div class='user-bubble clearfix'>{user_question}</div>", unsafe_allow_html=True)
    with st.spinner("Thinking..."):
        retrieved_docs = retriever.get_relevant_documents(user_question)
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        final_prompt = prompt.invoke({"context": context_text, "question": user_question})

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        response = llm.invoke(final_prompt)

        embed = CohereEmbeddings(model="embed-english-v3.0", user_agent="langchain", cohere_api_key=st.secrets["COHERE_API_KEY"])
        llm_embedding = embed.embed_query(response.content)

        texts = [doc.page_content for doc in retrieved_docs]
        chunk_embeddings = embed.embed_documents(texts)
        similarities = cosine_similarity([llm_embedding], chunk_embeddings)[0]
        ranked = sorted([(i, doc, sim) for i, (doc, sim) in enumerate(zip(retrieved_docs, similarities))], key=lambda x: x[2], reverse=True)

        top3_docs = [doc for _, doc, _ in ranked[:3]]
        top3_chunks = [doc.page_content for doc in top3_docs]

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

        ranking_input = ranking_prompt.invoke({
            "question": user_question,
            "chunk1": top3_chunks[0],
            "chunk2": top3_chunks[1],
            "chunk3": top3_chunks[2]
        })

        ranking_response = llm.invoke(ranking_input)
        response_text = ranking_response.content.strip()

        best_doc = top3_docs[int(response_text)-1] if response_text.isdigit() else top3_docs[0]

        if response.content.lower().startswith("the report does not specify"):
            page = raw_img = b64_img = None
        else:
            page = best_doc.metadata.get("page_number")
            raw_img = st.session_state.page_images.get(page)
            b64_img = pil_to_base64(raw_img) if raw_img else None

        entry = {"role": "assistant", "content": response.content}
        if page and b64_img:
            entry["source"] = f"Page {page}"
            entry["source_img"] = b64_img

        st.session_state.messages.append(entry)
        container = st.empty()
        typed = ""
        for char in response.content:
            typed += char
            container.markdown(f"<div class='assistant-bubble clearfix'>{typed}</div>", unsafe_allow_html=True)
            time.sleep(0.008)
        if b64_img:
            with st.popover("📘 Reference:"):
                st.image(Image.open(io.BytesIO(base64.b64decode(b64_img))), caption=f"Page {page}", use_container_width=True)

# === Style ===
st.markdown("""
<style>
html, body { font-size: 16px !important; line-height: 1.6; font-family: "Segoe UI", sans-serif; }
.user-bubble {
    background-color: #007bff; color: white; padding: 10px; border-radius: 12px; max-width: 60%;
    float: right; margin: 5px 0 10px auto; text-align: right; font-size: 18px;
}
.assistant-bubble {
    background-color: #1e1e1e; color: white; padding: 10px; border-radius: 12px; max-width: 60%;
    float: left; margin: 5px auto 10px 0; text-align: left; font-size: 18px;
}
.system-bubble {
    background-color: #1e1e1e; color: white; padding: 10px; border-radius: 12px; max-width: 60%;
    margin: 5px auto 10px auto; text-align: left; font-size: 18px;
}
.clearfix::after { content: ""; display: block; clear: both; }
</style>
""", unsafe_allow_html=True)
