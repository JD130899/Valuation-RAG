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
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()


import pickle
from langchain.vectorstores import FAISS as FAISS_LC 
st.info(f"Using Cohere API Key: {st.secrets['COHERE_API_KEY'][:8]}...")

openai.api_key = os.environ["OPENAI_API_KEY"]

# === Streamlit UI Config ===
st.set_page_config(page_title="Valuation RAG Chatbot", layout="wide")
st.title("Underwriting Agent")

uploaded_file = st.file_uploader("Upload a valuation report PDF", type="pdf")

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

if uploaded_file is not None:
    file_name = uploaded_file.name
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != file_name:
        with st.spinner("Processing PDF..."):
            os.makedirs("uploaded", exist_ok=True)
            PDF_PATH = os.path.join("uploaded", file_name)
            with open(PDF_PATH, "wb") as f:
                f.write(uploaded_file.getbuffer())
            EXTRACTED_FOLDER = os.path.join(os.getcwd(), "extracted")
            os.makedirs(EXTRACTED_FOLDER, exist_ok=True)

            # --- Store uploaded file name ---
            st.session_state.last_uploaded = file_name

            # === Load PDF images once ===
            doc = fitz.open(PDF_PATH)
            all_images = [Image.open(io.BytesIO(page.get_pixmap(dpi=300).tobytes("png"))) for page in doc]
            st.session_state.page_images = {i + 1: img for i, img in enumerate(all_images)}
            doc.close()

            # === Initialize session ===
            st.session_state.initialized = True

           
            st.session_state.initialized = True
            with st.spinner("Preparing your valuation assistant..."):
                doc = fitz.open(PDF_PATH)
                target_headings = {
                    "income_approach": {"text": "INCOME APPROACH", "take": 1},
                    "market_approach": {"text": "MARKET APPROACH", "take": 2},
                }
                valuation_summary_text = "VALUATION SUMMARY"
                valuation_summary_page = None
                heading_pages = {key: [] for key in target_headings}
        
                for i in range(len(doc)):
                    if i < 5: continue
                    text = doc[i].get_text().upper()
                    for key, config in target_headings.items():
                        if config["text"] in text:
                            heading_pages[key].append(i)
                    if valuation_summary_page is None and valuation_summary_text in text:
                        valuation_summary_page = i
                final_selections = []
                for key, pages in heading_pages.items():
                    idx = target_headings[key]["take"] - 1
                    if idx < len(pages):
                        final_selections.append((pages[idx], key))
                        if key == "market_approach" and pages[idx] + 1 < len(doc):
                            final_selections.append((pages[idx] + 1, f"{key}_continued"))
                if valuation_summary_page is not None:
                    final_selections.append((valuation_summary_page, "valuation_summary"))
                doc.close()
        
                for idx, label in final_selections:
                    image = all_images[idx]
                    out_path = os.path.join(EXTRACTED_FOLDER, f"{label}_page_{idx+1}.png")
                    image.save(out_path)
        
                for filename in sorted(os.listdir(EXTRACTED_FOLDER)):
                    if filename.endswith(".png"):
                        image_path = os.path.join(EXTRACTED_FOLDER, filename)
                        match = re.search(r'page_(\d+)', filename)
                        if not match: continue
                        page_num = match.group(1)
                        with open(image_path, "rb") as f:
                            img_bytes = f.read()
                        b64_img = base64.b64encode(img_bytes).decode("utf-8")
                        resp = openai.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "user", "content": [
                                    {"type": "text", "text": "Extract all values from this image."},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                                ]},
                            ],
                            max_tokens=512,
                        )
                        text_out = resp.choices[0].message.content
                        with open(os.path.join(EXTRACTED_FOLDER, f"page_{page_num}.txt"), "w") as f:
                            f.write(text_out)
        
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
                FAISS_FOLDER = os.path.join("vectorstore", file_name)
                os.makedirs(FAISS_FOLDER, exist_ok=True)
                index_file = os.path.join(FAISS_FOLDER, "faiss.index")
                metadata_file = os.path.join(FAISS_FOLDER, "metadata.pkl")
                
                if os.path.exists(index_file) and os.path.exists(metadata_file):
                    with open(metadata_file, "rb") as f:
                        stored_metadatas = pickle.load(f)
                    embed = CohereEmbeddings(model="embed-english-v3.0", user_agent="langchain",cohere_api_key=st.secrets["COHERE_API_KEY"])
                    vs = FAISS_LC.load_local(FAISS_FOLDER, embed, index_name="faiss")
                    print("✅ FAISS loaded from disk.")
                else:
                    embed = CohereEmbeddings(model="embed-english-v3.0", user_agent="langchain",cohere_api_key=st.secrets["COHERE_API_KEY"])
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
                
                    text_embeddings = list(zip(texts, embeddings))
                    vs = FAISS_LC.from_embeddings(text_embeddings, metadatas)

                    vs.save_local(FAISS_FOLDER, index_name="faiss")
                    with open(metadata_file, "wb") as f:
                        pickle.dump(metadatas, f)
                    print("💾 FAISS saved to disk.")
                base_ret = vs.as_retriever(search_type="mmr", search_kwargs={"k": 50, "fetch_k": 100, "lambda_mult": 0.9})
                reranker = CohereRerank(model="rerank-english-v3.0", user_agent="langchain", cohere_api_key=st.secrets["COHERE_API_KEY"], top_n=20)
                st.session_state.retriever = ContextualCompressionRetriever(base_retriever=base_ret, base_compressor=reranker)
                st.session_state.reranker = reranker
    else:
        PDF_PATH = os.path.join("uploaded", file_name) 
else:
    st.warning("📄 Please upload a PDF to continue.")
    st.stop()

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

prompt = PromptTemplate(
        template = """
        You are a financial-data extraction assistant.
    ...
    Context:
    {context}
    
    ---
    Question: {question}
    Answer:""",
            input_variables=["context", "question"]
        )

def typewriter_output(answer):
    if answer.strip().startswith("```markdown"):
        st.markdown(answer.strip().removeprefix("```markdown").removesuffix("```" ).strip())
    else:
        container = st.empty()
        typed = ""
        for char in answer:
            typed += char
            container.markdown(f"<div class='assistant-bubble clearfix'>{typed}</div>", unsafe_allow_html=True)
            time.sleep(0.008)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."},
        {"role": "assistant", "content": "What can I help you with?"}
    ]

user_question = st.chat_input("Message")

for msg in st.session_state.messages:
    role_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    st.markdown(f"<div class='{role_class} clearfix'>{msg['content']}</div>", unsafe_allow_html=True)
    if msg["role"] == "assistant" and msg.get("source_img"):
        with st.popover(f"📘 Reference:"):
            st.image(Image.open(io.BytesIO(base64.b64decode(msg["source_img"]))), caption=msg["source"], use_container_width=True)

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    st.markdown(f"<div class='user-bubble clearfix'>{user_question}</div>", unsafe_allow_html=True)
    with st.spinner("Thinking..."):
        if "retriever" not in st.session_state:
            st.error("⚠️ Assistant not ready yet. Please wait a moment after uploading the PDF.")
            st.stop()
        
        retrieved_docs = st.session_state.retriever.invoke(user_question)

        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        final_prompt = prompt.invoke({"context": context_text, "question": user_question})
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        response = llm.invoke(final_prompt)
        #best_doc = st.session_state.reranker.compress_documents(retrieved_docs, query=response.content)[0]
        #best_doc = st.session_state.reranker.compress_documents(retrieved_docs, query=user_question)[0]
        embed = CohereEmbeddings(model="embed-english-v3.0", user_agent="langchain",cohere_api_key=st.secrets["COHERE_API_KEY"])
        llm_embedding = embed.embed_query(response.content)

        texts = [doc.page_content for doc in retrieved_docs]
        chunk_embeddings = embed.embed_documents(texts)
        similarities = cosine_similarity([llm_embedding], chunk_embeddings)[0]
        ranked = sorted(
            [(i, doc, sim) for i, (doc, sim) in enumerate(zip(retrieved_docs, similarities))],
            key=lambda x: x[2], reverse=True
        )

        # === Prepare top 3 for LLM re-ranking ===
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

        # Check if it's numeric
        if response_text.isdigit():
            best_index = int(response_text) - 1
            best_doc = top3_docs[best_index]  
        else:
            st.error(f"Ranking response is not a valid index: {response_text}")
            best_index = 0
            best_doc = top3_chunks[0]  # fallback doc



        
        page = best_doc.metadata.get("page_number") if best_doc else None
        raw_img = st.session_state.page_images.get(page)
        b64_img = pil_to_base64(raw_img) if raw_img else None
        st.session_state.messages.append({
            "role": "assistant", "content": response.content,
            "source": f"Page {page}" if page else None,
            "source_img": b64_img
        })
        typewriter_output(response.content)
        if b64_img:
            with st.popover(f"📘 Reference:"):
                st.image(Image.open(io.BytesIO(base64.b64decode(b64_img))), caption=f"Page {page}", use_container_width=True)
