import streamlit as st
import os
import re
import fitz
import openai
import base64
from pdf2image import convert_from_path
import fitz 
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

import time
import io
import base64
from PIL import Image


openai.api_key = os.environ["OPENAI_API_KEY"]

# === Streamlit UI Config ===
st.set_page_config(page_title="Valuation RAG Chatbot", layout="wide")
st.title("Underwriting Agent")

# === File uploader ===
uploaded_file = st.file_uploader("Upload a valuation report PDF", type="pdf")


# HELPER: turn PIL→base64
def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")



# Once, at startup, stash every page’s PIL.Image in session_state
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        os.makedirs("uploaded", exist_ok=True)
        PDF_PATH = os.path.join("uploaded", uploaded_file.name)
        with open(PDF_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())

        EXTRACTED_FOLDER = os.path.join(os.getcwd(), "extracted")
        os.makedirs(EXTRACTED_FOLDER, exist_ok=True)
        
        all_images = convert_from_path(PDF_PATH, dpi=300)
        st.session_state.page_images = {i + 1: img for i, img in enumerate(all_images)}
else:
    st.warning("📄 Please upload a PDF to continue.")
    st.stop()




# === CSS Styling for chat bubbles ===
st.markdown("""
<style>
html, body {
    font-size: 16px !important;
    line-height: 1.6;
    font-family: "Segoe UI", sans-serif;
}
.user-bubble {
    background-color: #007bff;
    color: white;
    padding: 10px;
    border-radius: 12px;
    max-width: 60%;
    float: right;
    margin: 5px 0 10px auto;
    text-align: right;
    font-size: 18px;
}
.assistant-bubble {
    background-color: #1e1e1e;
    color: white;
    padding: 10px;
    border-radius: 12px;
    max-width: 60%;
    float: left;
    margin: 5px auto 10px 0;
    text-align: left;
    font-size: 18px;
}
.system-bubble {
    background-color: #1e1e1e;
    color: white;
    padding: 10px;
    border-radius: 12px;
    max-width: 60%;
    margin: 5px auto 10px auto;
    text-align: left;
    font-size: 18px;
}
.clearfix::after {
    content: "";
    display: block;
    clear: both;
}

</style>
""", unsafe_allow_html=True)


# === State initialization ===
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    with st.spinner("Preparing your valuation assistant..."):

        # === Step 1: Extract pages of interest as images ===
        target_headings = {
            "income_approach": {"text": "INCOME APPROACH", "take": 1},
            "market_approach": {"text": "MARKET APPROACH", "take": 2},
        }
        valuation_summary_text = "VALUATION SUMMARY"
        valuation_summary_page = None

        doc = fitz.open(PDF_PATH)
        heading_pages = {key: [] for key in target_headings}

        for i in range(len(doc)):
            if i < 5:
                continue
            text = doc[i].get_text().upper()
            for key, config in target_headings.items():
                if config["text"] in text:
                    heading_pages[key].append(i)
            if valuation_summary_page is None and valuation_summary_text in text:
                valuation_summary_page = i
        doc.close()

        final_selections = []
        for key, pages in heading_pages.items():
            take_index = target_headings[key]["take"] - 1
            if take_index < len(pages):
                selected_page = pages[take_index]
                final_selections.append((selected_page, key))
                if key == "market_approach":
                    total_pages = len(fitz.open(PDF_PATH))
                    if selected_page + 1 < total_pages:
                        final_selections.append((selected_page + 1, f"{key}_continued"))
        if valuation_summary_page is not None:
            final_selections.append((valuation_summary_page, "valuation_summary"))


        for idx, label in final_selections:
            image = all_images[idx]
            output_path = os.path.join(EXTRACTED_FOLDER, f"{label}_page_{idx+1}.png")
            image.save(output_path)

        for filename in sorted(os.listdir(EXTRACTED_FOLDER)):
            if filename.endswith(".png"):
                image_path = os.path.join(EXTRACTED_FOLDER, filename)
                match = re.search(r'page_(\d+)', filename)
                if not match:
                    continue
                page_num = match.group(1)
                with open(image_path, "rb") as image_file:
                    image_bytes = image_file.read()
                    base64_image = base64.b64encode(image_bytes).decode("utf-8")
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Extract all values from this image."},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                            ],
                        },
                    ],
                    max_tokens=512,
                )
                output_text = response.choices[0].message.content
                with open(os.path.join(EXTRACTED_FOLDER, f"page_{page_num}.txt"), "w") as f:
                    f.write(output_text)

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
                    cleaned = [line for line in content.splitlines() if line.strip() and line.strip().lower() != "null"]
                    content = "\n".join(cleaned)
                if content:
                    pages.append(Document(page_content=content, metadata={"page_number": page_num}))
            return pages

        docs = parse_pdf()
        splitter = RecursiveCharacterTextSplitter(chunk_size=3300, chunk_overlap=0)
        split_docs = splitter.split_documents(docs)
        for i, doc in enumerate(split_docs):
            doc.metadata["chunk_id"] = i + 1

        embedding = CohereEmbeddings(model="embed-english-v3.0", user_agent="langchain")
        vectorstore = FAISS.from_documents(split_docs, embedding)
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 50, "fetch_k": 100, "lambda_mult": 0.9})
        reranker = CohereRerank(model="rerank-english-v3.0", user_agent="langchain", top_n=20)
        final_retriever = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=reranker)

        st.session_state.retriever = final_retriever
        st.session_state.reranker = reranker

# === Prompt ===
prompt = PromptTemplate(
        template = """
        You are a financial-data extraction assistant.
    
    **Use ONLY what appears under “Context”.**
    
    ### How to answer
    1. **Single value questions**  
       • Find the row + column that match the user's words.  
       • Return the answer in a **short, clear sentence** using the exact number from the context.  
         Example: “The Income (DCF) approach value is $1,150,000.”  
       • **Do NOT repeat the metric name or company name** unless the user asks.
    
    2. **Table questions**  
       • Return the full table **with its header row** in GitHub-flavoured markdown.
    
    3. **Valuation method / theory / reasoning questions**
       • If the question involves **valuation methods**, **concluded value**, or topics like **Income Approach**, **Market Approach**, or **Valuation Summary**, do the following:
         - Combine and synthesize relevant information across all chunks.
         - Pay special attention to how **weights are distributed** (e.g., “50% DCF, 25% EBITDA, 25% SDE”).
         - Avoid oversimplifying if more detailed breakdowns (like subcomponents of market approach) are available.
         - If a table gives a simplified view (e.g., "50% Market Approach"), but other parts break it down (e.g., 25% EBITDA + 25% SDE), **prefer the detailed breakdown with percent value**.   
         - When describing weights, also mention the **corresponding dollar values** used in the context (e.g., “50% DCF = $3,712,000, 25% EBITDA = $4,087,000...”)
         - **If Market approach is composed of sub-methods like EBITDA and SDE, then explicitly extract and show their individual weights and values, even if not listed together in a single table.**
    
    
    4. **Theory/textual question**  
       • Try to return an explanation **based on the context**.
    
       
    If you still cannot see the answer, reply **“I don't know.”**
    
    ---
    Context:
    {context}
    
    ---
    Question: {question}
    Answer:""",
            input_variables=["context", "question"]
        )

def typewriter_output(answer):
    if answer.strip().startswith("```markdown"):
        markdown_table = answer.strip().removeprefix("```markdown").removesuffix("```" ).strip()
        st.markdown(markdown_table)
    else:
        container = st.empty()
        typed = ""
        for char in answer:
            typed += char
            container.markdown(f"<div class='assistant-bubble clearfix'>{typed}</div>", unsafe_allow_html=True)
            time.sleep(0.008)

# === Chat memory setup ===
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."},
        {"role": "assistant", "content": "What can I help you with?"}
    ]

user_question = st.chat_input("Message")

# === Chat history display ===
for msg in st.session_state.messages:
    role_class = "user-bubble" if msg["role"]=="user" else "assistant-bubble"
    st.markdown(
        f"<div class='{role_class} clearfix'>{msg['content']}</div>",
        unsafe_allow_html=True
    )

    if msg["role"]=="assistant" and msg.get("source_img"):
        # a tiny click‑to‑expand box, no full‑width expander
       with st.popover(f"📘 Source Info (Page {msg['source']})"):
            # show only the snippet
            st.image(
                Image.open(io.BytesIO(base64.b64decode(msg["source_img"]))),
                caption=msg["source"],
                use_container_width=True
            )

# === Chat input and logic ===

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    st.markdown(f"<div class='user-bubble clearfix'>{user_question}</div>", unsafe_allow_html=True)

    with st.spinner("Thinking..."):
        retrieved_docs = st.session_state.retriever.invoke(user_question)
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        final_prompt = prompt.invoke({"context": context_text, "question": user_question})
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        response = llm.invoke(final_prompt)

        def find_best_matching_doc_reranker(answer_text, retrieved_docs, reranker):
            reranked = reranker.compress_documents(retrieved_docs, query=answer_text)
            return reranked[0] if reranked else None

        matched_doc = find_best_matching_doc_reranker(response.content, retrieved_docs, st.session_state.reranker)

        page = matched_doc.metadata.get("page_number") if matched_doc else None
        raw_img = st.session_state.page_images.get(page)
        b64_img = pil_to_base64(raw_img) if raw_img else None

        # append the assistant reply with source_img=b64_img
        st.session_state.messages.append({
            "role":       "assistant",
            "content":    response.content,
            "source":     f"Page {page}" if page else None,
            "source_img": b64_img
        })

        typewriter_output(response.content)

        if b64_img:
            with st.popover(f"📘 Source Info Page: {page}"):
                # decode base64 back into a PIL.Image
                img = Image.open(io.BytesIO(base64.b64decode(b64_img)))
                st.image(img, caption=f"Page {page}", use_container_width=True)




       
