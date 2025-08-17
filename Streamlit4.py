# custom_fab_bottom_right.py
import os, io, base64, uuid, time
import streamlit as st
from dotenv import load_dotenv
import streamlit.components.v1 as components

# If you use these elsewhere later, they’re safe to import now:
# import fitz  # PyMuPDF
# from PIL import Image

# ---- Your Drive helpers (already in your repo) ----
from gdrive_utils import get_drive_service, get_all_pdfs, download_pdf

# ================= Setup =================
load_dotenv()
st.set_page_config(page_title="Underwriting Agent", layout="wide")

# ---------- Global store so history survives reloads ----------
@st.cache_resource
def _store():
    return {"messages": []}

store = _store()

# ---------------- Session state ----------------
def _new_id():
    st.session_state.next_id += 1
    return f"m{st.session_state.next_id}"

def _sync_store():
    store["messages"] = st.session_state.messages

def _reset_chat():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."},
        {"role": "assistant", "content": "What can I help you with?"}
    ]
    st.session_state.pending_input = None
    st.session_state.waiting_for_response = False

if "messages" not in st.session_state:
    # prefer saved store if present; otherwise use welcome pair
    st.session_state.messages = store["messages"] or [
        {"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."},
        {"role": "assistant", "content": "What can I help you with?"}
    ]
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None
if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False
if "next_id" not in st.session_state:
    st.session_state.next_id = 0

# Track Drive selection/file so we can reset chat when a NEW file is loaded
if "last_synced_file_id" not in st.session_state:
    st.session_state.last_synced_file_id = None
if "uploaded_file_from_drive" not in st.session_state:
    st.session_state.uploaded_file_from_drive = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "last_selected_upload" not in st.session_state:
    st.session_state.last_selected_upload = None

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

# ---------- Small helper: “open file” badge ----------
def file_badge_link(name: str, pdf_bytes: bytes, *, synced: bool = True):
    base = os.path.splitext(name)[0]
    b64 = base64.b64encode(pdf_bytes).decode("ascii")
    label = "Using synced file:" if synced else "Using file:"
    link_id = f"open-file-{uuid.uuid4().hex[:8]}"
    st.markdown(
        f'''
        <div style="background:#1f2c3a; padding:8px; border-radius:8px; color:#fff; margin:6px 0 12px;">
          ✅ <b>{label}</b>
          <a id="{link_id}" href="#" target="_blank" rel="noopener" style="color:#93c5fd; text-decoration:none;">{base}</a>
        </div>
        ''',
        unsafe_allow_html=True
    )
    components.html(
        f'''<!doctype html><meta charset='utf-8'>
<style>html,body{{background:transparent;margin:0;height:0;overflow:hidden}}</style>
<script>(function(){{
  function b64ToUint8Array(s){{var b=atob(s),u=new Uint8Array(b.length);for(var i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u;}}
  var blob = new Blob([b64ToUint8Array("{b64}")], {{type:"application/pdf"}});
  var url  = URL.createObjectURL(blob);
  function attach(){{
    var d = window.parent && window.parent.document;
    if(!d) return setTimeout(attach,120);
    var a = d.getElementById("{link_id}");
    if(!a) return setTimeout(attach,120);
    a.setAttribute("href", url);
  }}
  attach();
  var me = window.frameElement; if(me){{me.style.display="none";me.style.height="0";me.style.border="0";}}
}})();</script>''',
        height=0,
    )

# --------------- Styles ---------------
st.markdown("""
<style>
  .block-container { padding-bottom: 140px; }

  .fab-wrap {
    position: fixed;
    right: 24px;
    bottom: 110px;   /* sits above st.chat_input */
    z-index: 1000;
    display: flex; gap: 10px; align-items: center; justify-content: flex-end;
  }
  .fab-btn {
    background: #000 !important; color: #fff !important;
    border-radius: 9999px !important; padding: 10px 16px !important; border: none !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25); font-weight: 600; cursor: pointer;
  }
  .fab-btn:hover { filter: brightness(1.08); }

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
  .clearfix::after { content: ""; display: table; clear: both; }
</style>
""", unsafe_allow_html=True)

# ================= Sidebar: Google Drive loader =================
st.sidebar.header("📂 Files")
try:
    service = get_drive_service()
    pdf_files = get_all_pdfs(service)
    if pdf_files:
        names = [f["name"] for f in pdf_files]
        sel = st.sidebar.selectbox("Select a PDF from Google Drive", names, key="gd_sel")
        chosen = next(f for f in pdf_files if f["name"] == sel)
        if st.sidebar.button("📥 Load Selected PDF", key="gd_load"):
            fid, fname = chosen["id"], chosen["name"]
            if fid == st.session_state.last_synced_file_id:
                st.sidebar.info("✅ Already loaded.")
            else:
                path = download_pdf(service, fid, fname)
                if path:
                    st.session_state.uploaded_file_from_drive = open(path, "rb").read()
                    st.session_state.uploaded_file_name = fname
                    st.session_state.last_synced_file_id = fid
                    _reset_chat()
    else:
        st.sidebar.info("No PDFs found in Drive.")
except Exception as e:
    st.sidebar.warning(f"Drive not configured: {e}")

# ================= Main UI =================
st.title("Underwriting Agent")

# Show a badge if we have a loaded/synced file; else allow upload
if st.session_state.uploaded_file_from_drive and st.session_state.uploaded_file_name:
    file_badge_link(
        st.session_state.uploaded_file_name,
        st.session_state.uploaded_file_from_drive,
        synced=True
    )
else:
    up = st.file_uploader("Upload a valuation report PDF", type="pdf", key="local_uploader")
    if up:
        # Only reset when user actually changes the file
        if up.name != st.session_state.last_selected_upload:
            st.session_state.last_selected_upload = up.name
            # Stash like Drive path for consistent badge behavior
            st.session_state.uploaded_file_from_drive = up.getvalue()
            st.session_state.uploaded_file_name = up.name
            st.session_state.last_synced_file_id = None
            _reset_chat()
    if st.session_state.uploaded_file_from_drive and st.session_state.uploaded_file_name:
        file_badge_link(
            st.session_state.uploaded_file_name,
            st.session_state.uploaded_file_from_drive,
            synced=False
        )

# ---------------- Render full history with custom bubbles ----------------
for m in st.session_state.messages:
    cls = "user-bubble" if m["role"] == "user" else "assistant-bubble"
    st.markdown(f"<div class='{cls} clearfix'>{m['content']}</div>", unsafe_allow_html=True)

# ---------------- Handle ?qs= ----------------
qs = st.query_params.get("qs")
if qs:
    queue_question(qs)
    try:
        del st.query_params["qs"]
    except Exception:
        pass

# ---------------- Fixed buttons (bottom-right) ----------------
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

# ---------------- Chat input ----------------
user_q = st.chat_input("Type your question here…")
if user_q:
    queue_question(user_q)

# ---------------- Answer queued question ----------------
if st.session_state.waiting_for_response and st.session_state.pending_input:
    # Show Thinking... as assistant bubble
    st.markdown("<div class='assistant-bubble'>Thinking…</div>", unsafe_allow_html=True)
    answer_pending()
    st.rerun()
