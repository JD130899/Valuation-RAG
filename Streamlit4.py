import os
import io
import time
import pickle
import base64

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
import openai
from gdrive_utils import get_drive_service, get_all_pdfs, download_pdf

# ————————————— Setup —————————————————————————————————————
load_dotenv()
st.set_page_config(page_title="Valuation RAG Chatbot", layout="wide")
openai.api_key = os.environ["OPENAI_API_KEY"]

if "last_synced_file_id" not in st.session_state:
    st.session_state.last_synced_file_id = None
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant","content":"Hi! I am here to answer any questions you may have about your valuation report."},
        {"role":"assistant","content":"What can I help you with?"}
    ]

# ————————————— CACHING BUILDER —————————————————————————————————
@st.cache_resource(show_spinner="📦 Processing & indexing PDF…")
def build_index_and_images(pdf_bytes: bytes, file_name: str):
    # 1) Save PDF
    os.makedirs("uploaded", exist_ok=True)
    pdf_path = os.path.join("uploaded", file_name)
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    # 2) Extract page images
    doc = fitz.open(pdf_path)
    page_images = {
        i+1: Image.open(io.BytesIO(page.get_pixmap(dpi=300).tobytes("png")))
        for i, page in enumerate(doc)
    }
    doc.close()

    # 3) Parse with LlamaParse
    parser = LlamaParse(api_key=os.environ["LLAMA_CLOUD_API_KEY"], num_workers=4)
    result = parser.parse(pdf_path)
    pages = []
    for pg in result.pages:
        cleaned = [l for l in pg.md.splitlines() if l.strip() and l.lower()!="null"]
        text = "\n".join(cleaned)
        if text:
            pages.append(Document(page_content=text, metadata={"page_number":pg.page}))

    # 4) Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=3300, chunk_overlap=0)
    chunks = splitter.split_documents(pages)
    for idx, c in enumerate(chunks):
        c.metadata["chunk_id"] = idx+1

    # 5) Embed & index
    embedder = CohereEmbeddings(
        model="embed-english-v3.0",
        user_agent="langchain",
        cohere_api_key=st.secrets["COHERE_API_KEY"]
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
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ————————————— Sidebar: Google Drive loader —————————————————————————
service = get_drive_service()
pdf_files = get_all_pdfs(service)
if pdf_files:
    names = [f["name"] for f in pdf_files]
    sel   = st.sidebar.selectbox("📂 Select a PDF from Google Drive", names)
    chosen = next(f for f in pdf_files if f["name"]==sel)
    if st.sidebar.button("📥 Load Selected PDF"):
        fid, fname = chosen["id"], chosen["name"]
        if fid == st.session_state.last_synced_file_id:
            st.sidebar.info("✅ Already loaded.")
        else:
            path = download_pdf(service, fid, fname)
            if path:
                st.session_state.uploaded_file_from_drive = open(path,"rb").read()
                st.session_state.uploaded_file_name = fname
                st.session_state.last_synced_file_id = fid

                # 🔗 Set display name (no .pdf) and a Drive view link
                # 🔗 Store display name, pretty viewer link, and direct PDF link
                st.session_state.report_display_name = os.path.splitext(fname)[0]
                st.session_state.report_link_view = f"https://drive.google.com/file/d/{fid}/view"           # for banner
                st.session_state.report_link_pdf  = f"https://drive.google.com/uc?export=download&id={fid}" # for deep links


                # reset chat
                st.session_state.messages = [
                    {"role":"assistant","content":"Hi! I am here to answer any questions you may have about your valuation report."},
                    {"role":"assistant","content":"What can I help you with?"}
                ]
else:
    st.sidebar.warning("📭 No PDFs found in Drive.")


# ————————————— Main UI —————————————————————————————————————————
st.title("Underwriting Agent")

# Figure out the active file (Drive vs local upload)
if "uploaded_file_from_drive" in st.session_state:
    # We have a Drive-loaded file
    up = io.BytesIO(st.session_state.uploaded_file_from_drive)
    up.name = st.session_state.uploaded_file_name
else:
    up = st.file_uploader("Upload a valuation report PDF", type="pdf")

if not up:
    st.warning("Please upload or load a PDF to continue.")
    st.stop()

# Reset greeting if the PDF changed
if st.session_state.get("last_processed_pdf") != up.name:
    st.session_state.messages = [
        {"role":"assistant","content":"Hi! I am here to answer any questions you may have about your valuation report."},
        {"role":"assistant","content":"What can I help you with?"}
    ]
    st.session_state["last_processed_pdf"] = up.name

# Convert to bytes for caching
pdf_bytes = up.getvalue()

# 🔗 For locally uploaded PDFs, create display name + data URL link
if "uploaded_file_from_drive" not in st.session_state:
    st.session_state.report_display_name = os.path.splitext(up.name)[0]
    st.session_state.report_link = "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode("utf-8")

# — Top banner with hyperlink (no “.pdf”) ————————————————————————
if "report_link_view" in st.session_state and "report_display_name" in st.session_state:
    st.markdown(
        f"""
        <div style='background:#1f2c3a; padding:8px; border-radius:8px; color:#fff;'>
          ✅ <b>Using synced file:</b>
          <a href="{st.session_state.report_link_view}" target="_blank"
             style="color:#fff; text-decoration:underline;">
             {st.session_state.report_display_name}
          </a>
        </div>
        """,
        unsafe_allow_html=True
    )

# — build (or fetch from cache) ————————————————————————————————
retriever, page_images = build_index_and_images(pdf_bytes, up.name)

# ————————————— Chat bubbles styling —————————————————————————————————
st.markdown("""
<style>
.user-bubble {background:#007bff;color:#fff;padding:8px;border-radius:8px;max-width:60%;float:right;margin:4px;}
.assistant-bubble {background:#1e1e1e;color:#fff;padding:8px;border-radius:8px;max-width:60%;float:left;margin:4px;}
.clearfix::after {content:"";display:table;clear:both;}
</style>
""", unsafe_allow_html=True)

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
    🛎️ After you answer the user’s question (using steps 1–4), **only if** there is still **unused** relevant report content, **ask**:  
      “Would you like more detail on [X]?”  
   Otherwise, **do not** ask any follow-up.

**Use ONLY what appears under “Context”.**

### Special interpretation rules  
      • If the question is about **"valuation"** in general (e.g., “What is the valuation?”), answer by giving the Fair Market Value   
      • If the question is about **risk** (e.g., “How risky is the business?”), use the **risk assessment section**, and include the **risk classification** (e.g., secure, controlled, etc.).

### How to answer
1. **Single value questions**  
   • Find the row + column that match the user's words.  
   • Return the answer in a **short, clear sentence** using the exact number from the context.  
     Example: “The Income (DCF) approach value is $1,150,000.”  
   • **Do NOT repeat the metric name or company name** unless the user asks.

2. **Table questions**  
   • Return the full table **with its header row** in GitHub-flavoured markdown.
    
3. **Valuation method / theory / reasoning questions**
    • Combine and synthesize relevant information across all chunks.
    • Prefer detailed breakdowns (weights + dollar values) when available.

4. **Theory/textual question**  
   • Try to return an explanation **based on the context**.
       
If you still cannot see the answer, reply **“Hmm, I am not sure. Are you able to rephrase your question?”**
    
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
    template=base_text + """
Conversation so far:
{chat_history}

""", 
    input_variables=["chat_history", "context", "question"]
)

# — Render chat so far (add link inside popover if present) ————————————
for msg in st.session_state.messages:
    cls = "user-bubble" if msg["role"]=="user" else "assistant-bubble"
    st.markdown(f"<div class='{cls} clearfix'>{msg['content']}</div>", unsafe_allow_html=True)
    if msg.get("source_img"):
        with st.popover("📘 Reference:"):
            data = base64.b64decode(msg["source_img"])
            st.image(Image.open(io.BytesIO(data)), caption=msg["source"], use_container_width=True)
            if msg.get("source_link"):
                st.markdown(f"[Open this page in new tab]({msg['source_link']})")

# — user input ——————————————————————————————————————————————
user_q = st.chat_input("Message")
if user_q:
    st.session_state.messages.append({"role":"user","content":user_q})
    st.rerun()

# — answer when last role was user —————————————————————————————————
if st.session_state.messages and st.session_state.messages[-1]["role"]=="user":
    q = st.session_state.messages[-1]["content"]
    with st.spinner("Thinking…"):
        docs = retriever.get_relevant_documents(q)
        ctx  = "\n\n".join(d.page_content for d in docs)
        history_to_use = st.session_state.messages[-10:]

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        full_input = {
            "chat_history": format_chat_history(history_to_use),
            "context":      ctx,
            "question":     q
        }
        ans = llm.invoke(wrapped_prompt.invoke(full_input)).content
      
        # — your 3-chunk reranking logic intact ——————————————
        texts = [d.page_content for d in docs]
        emb_query = CohereEmbeddings(
            model="embed-english-v3.0", user_agent="langchain", cohere_api_key=st.secrets["COHERE_API_KEY"]
        ).embed_query(ans)
        chunk_embs = CohereEmbeddings(
            model="embed-english-v3.0", user_agent="langchain", cohere_api_key=st.secrets["COHERE_API_KEY"]
        ).embed_documents(texts)
        sims = cosine_similarity([emb_query], chunk_embs)[0]
        ranked = sorted(list(zip(docs, sims)), key=lambda x: x[1], reverse=True)
        top3 = [d for d,_ in ranked[:3]]

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
            input_variables=["question","chunk1","chunk2","chunk3"]
        )
        pick = ChatOpenAI(model="gpt-4o", temperature=0).invoke(
            ranking_prompt.invoke({
                "question": q,
                "chunk1": top3[0].page_content,
                "chunk2": top3[1].page_content,
                "chunk3": top3[2].page_content
            })
        ).content.strip()

        if pick.isdigit():
            best_doc = top3[int(pick)-1]
        else:
            best_doc = top3[0]

        page = best_doc.metadata.get("page_number")
        img = page_images.get(page)
        b64 = pil_to_base64(img) if img else None

        entry = {"role":"assistant","content":ans}
        if page and b64:
            entry["source"]     = f"Page {page}"
            entry["source_img"] = b64
            # Use direct PDF link for page jumping
            base = st.session_state.get("report_link_pdf") or st.session_state.get("report_link")
            if base:
                entry["source_link"] = f"{base}#page={page}"


        st.session_state.messages.append(entry)
        st.rerun()
