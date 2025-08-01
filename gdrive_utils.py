import os
import io
import json
import time
import pickle
import streamlit as st
from PIL import Image
import fitz  # PyMuPDF

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# LangChain and parsing
from langchain_core.documents import Document
from llama_cloud_services import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS

# === CONFIG ===
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FOLDER_ID = "1VglZDFbufOxHTZ4qZ_feUw_XHaxacPxr"  # Replace with your folder ID


# === Auth ===
def get_drive_service():
    service_account_info = st.secrets["service_account"]
    creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)


# === Get All PDFs ===
def get_all_pdfs(service):
    query = f"'{FOLDER_ID}' in parents and trashed = false"
    try:
        results = service.files().list(
            q=query,
            orderBy="createdTime desc",
            pageSize=20,
            fields="files(id, name, mimeType)"
        ).execute()
        files = results.get("files", [])
        return [file for file in files if file["name"].lower().endswith(".pdf")]
    except Exception as e:
        st.error(f"❌ Error accessing Drive folder: {e}")
        return []


# === Download and trigger processing ===
def download_pdf(service, file_id, file_name):
    try:
        request = service.files().get_media(fileId=file_id)
        file_path = os.path.join("uploaded", file_name)
        os.makedirs("uploaded", exist_ok=True)
        with io.FileIO(file_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()

        # === Check if already processed ===
        faiss_path = os.path.join("vectorstore", file_name, "faiss.index")
        images_path = os.path.join("extracted", f"{file_name}_images.pkl")
        if not os.path.exists(faiss_path) or not os.path.exists(images_path):
            process_pdf(file_path, file_name)

        return file_path
    except Exception as e:
        st.error(f"❌ Failed to download PDF: {e}")
        return None


# === One-time processing logic ===
def process_pdf(file_path, file_name):
    st.info(f"⚙️ Processing new file: {file_name}")

    # === Create directories ===
    FAISS_FOLDER = os.path.join("vectorstore", file_name)
    EXTRACTED_FOLDER = os.path.join("extracted")
    os.makedirs(FAISS_FOLDER, exist_ok=True)
    os.makedirs(EXTRACTED_FOLDER, exist_ok=True)

    # === Convert pages to images ===
    doc = fitz.open(file_path)
    all_images = [Image.open(io.BytesIO(page.get_pixmap(dpi=300).tobytes("png"))) for page in doc]
    with open(os.path.join(EXTRACTED_FOLDER, f"{file_name}_images.pkl"), "wb") as f:
        pickle.dump(all_images, f)
    doc.close()

    # === Parse content using LlamaParse ===
    parser = LlamaParse(api_key=os.environ["LLAMA_CLOUD_API_KEY"], num_workers=4)
    result = parser.parse(file_path)
    pages = []
    for page in result.pages:
        content = page.md.strip()
        cleaned = [l for l in content.splitlines() if l.strip() and l.strip().lower() != "null"]
        content = "\n".join(cleaned)
        if content:
            pages.append(Document(page_content=content, metadata={"page_number": page.page}))

    # === Split into chunks ===
    splitter = RecursiveCharacterTextSplitter(chunk_size=3300, chunk_overlap=0)
    chunks = splitter.split_documents(pages)
    for i, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = i + 1

    # === Embed and store ===
    embed = CohereEmbeddings(
        model="embed-english-v3.0",
        user_agent="langchain",
        cohere_api_key=st.secrets["COHERE_API_KEY"]
    )

    texts = [doc.page_content for doc in chunks]
    embeddings = []
    for i, text in enumerate(texts):
        try:
            emb = embed.embed_query(text)
            embeddings.append(emb)
        except Exception as e:
            st.error(f"Embedding failed for chunk {i}: {e}")
            embeddings.append([0.0] * 1024)
        time.sleep(0.5)

    vs = FAISS.from_documents(chunks, embed)
    vs.save_local(FAISS_FOLDER, index_name="faiss")
    with open(os.path.join(FAISS_FOLDER, "metadata.pkl"), "wb") as f:
        pickle.dump([doc.metadata for doc in chunks], f)

    st.success(f"✅ Processing complete: {file_name}")
