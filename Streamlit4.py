# custom_fab_bottom_right.py
import time
import streamlit as st

# ===== NEW: imports for sidebar loader =====
from dotenv import load_dotenv
from gdrive_utils import get_drive_service, get_all_pdfs, download_pdf

load_dotenv()
st.set_page_config(page_title="Underwriting Agent (Demo)", layout="wide")

# ---------- Global store so history survives reloads ----------
@st.cache_resource
def _store():
    return {"messages": []}

store = _store()

# ---------------- Session state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = store["messages"] or [
        {"role": "assistant", "content": "Hi! Ask anything about your valuation report."}
    ]
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None
if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False
if "next_id" not in st.session_state:
    st.session_state.next_id = 0

# ===== NEW: sidebar-related state (does not affect layout) =====
st.session_state.setdefault("last_synced_file_id", None)
st.session_state.setdefault("uploaded_file_from_drive", None)
st.session_state.setdefault("uploaded_file_name", None)

def _new_id():
    st.session_state.next_id += 1
    return f"m{st.session_state.next_id}"

def _sync_store():
    store["messages"] = st.session_state.messages

def queue_question(q: str):
    if not q:
        return
    st.session_state.pending_input = q
    st.session_state.waiting_for_response = True
    st.session_state.messages.append({"id": _new_id(), "role": "user", "content": q})
    _sync_store()

def answer_pending():
    time.sleep(0.2)  # simulate LLM
    q = st.session_state.pending_input
    ans = f"(demo) You asked: {q}"
    st.session_state.messages.append({"id": _new_id(), "role": "assistant", "content": ans})
    st.session_state.pending_input = None
    st.session_state.waiting_for_response = False
    _sync_store()

# --------------- Styles (UNCHANGED) ---------------
st.markdown("""
<style>
  .block-container { padding-bottom: 140px; }

  .fab-wrap {
    position: fixed;
    right: 24px;
    bottom: 110px;
    z-index: 1000;
    display: flex; gap: 10px; align-items: center; justify-content: flex-end;
  }
  .fab-btn {
    background: #000 !important; color: #fff !important;
    border-radius: 9999px !important; padding: 10px 16px !important; border: none !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25); font-weight: 600; cursor: pointer;
  }
  .fab-btn:hover { filter: brightness(1.08); }

  /* custom chat bubbles */
  .user-bubble {
    background: #007bff; color: #fff;
    padding: 8px 12px; border-radius: 12px;
    margin: 4px 0; max-width: 60%;
    float: right; clear: both;
  }
  .assistant-bubble {
    background: #1e1e1e; color: #fff;
    padding: 8px 12px; border-radius: 12px;
    margin: 4px 0; max-width: 60%;
    float: left; clear: both;
  }
</style>
""", unsafe_allow_html=True)

# ===================== NEW: Sidebar (Google Drive loader) =====================
st.sidebar.title("Underwriting Agent")
try:
    service = get_drive_service()
    pdf_files = get_all_pdfs(service)
except Exception as e:
    pdf_files = []
    st.sidebar.warning(f"Drive not available: {e}")

if pdf_files:
    names = [f["name"] for f in pdf_files]
    sel = st.sidebar.selectbox("📂 Select a PDF from Google Drive", names, index=0 if names else None)
    chosen = next((f for f in pdf_files if f["name"] == sel), None)
    if st.sidebar.button("📥 Load Selected PDF"):
        if chosen:
            fid, fname = chosen["id"], chosen["name"]
            if fid == st.session_state.last_synced_file_id:
                st.sidebar.info("✅ Already loaded.")
            else:
                path = download_pdf(service, fid, fname)
                if path:
                    with open(path, "rb") as fh:
                        st.session_state.uploaded_file_from_drive = fh.read()
                    st.session_state.uploaded_file_name = fname
                    st.session_state.last_synced_file_id = fid
                    # keep your original one-line welcome
                    st.session_state.messages = [
                        {"role": "assistant", "content": "Hi! Ask anything about your valuation report."}
                    ]
                    st.session_state.pending_input = None
                    st.session_state.waiting_for_response = False
                    _sync_store()
else:
    st.sidebar.warning("📭 No PDFs found in Drive.")

# Optional: quick reset while testing; keeps layout the same
if st.sidebar.button("🧹 New chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask anything about your valuation report."}
    ]
    st.session_state.pending_input = None
    st.session_state.waiting_for_response = False
    _sync_store()

# ===================== Main (UNCHANGED) =====================
st.title("Underwriting Agent (Demo)")

# (Non-intrusive) show which file is active, if any
if st.session_state.uploaded_file_name:
    st.info(f"Using file: **{st.session_state.uploaded_file_name}**")

# ---------------- Render full history with custom bubbles ----------------
for m in st.session_state.messages:
    cls = "user-bubble" if m["role"] == "user" else "assistant-bubble"
    st.markdown(f"<div class='{cls}'>{m['content']}</div>", unsafe_allow_html=True)

# ---------------- Handle ?qs= (UNCHANGED) ----------------
qs = st.query_params.get("qs")
if qs:
    queue_question(qs)
    try:
        del st.query_params["qs"]
    except Exception:
        pass

# ---------------- Fixed buttons (UNCHANGED HTML FORMS) ----------------
st.markdown("""
<div class="fab-wrap">
  <form method="get" style="margin:0;">
    <input type="hidden" name="qs" value="ETRAN Cheatsheet"/>
    <button class="fab-btn" type="submit">ETRAN Cheatsheet</button>
  </form>
  <form method="get" style="margin:0;">
    <input type="hidden" name="qs" value="What is the valuation?"/>
    <button class="fab-btn" type="submit">Valuation</button>
  </form>
  <form method="get" style="margin:0;">
    <input type="hidden" name="qs" value="Goodwill value"/>
    <button class="fab-btn" type="submit">Goodwill value</button>
  </form>
</div>
""", unsafe_allow_html=True)

# ---------------- Chat input (UNCHANGED) ----------------
user_q = st.chat_input("Type your question here…")
if user_q:
    queue_question(user_q)

# ---------------- Answer queued question (UNCHANGED) ----------------
if st.session_state.waiting_for_response and st.session_state.pending_input:
    st.markdown("<div class='assistant-bubble'>Thinking…</div>", unsafe_allow_html=True)
    answer_pending()
    st.rerun()
