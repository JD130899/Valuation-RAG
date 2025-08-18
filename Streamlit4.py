# app.py
import os
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Pinned Buttons Chat", layout="wide")

# (optional) tighten page padding a bit
st.markdown("""
<style>
  .user-bubble{ background:#2563eb; color:#fff; padding:8px 14px; border-radius:14px; margin:4px 0; max-width:70%; align-self:flex-end; }
  .assistant-bubble{ background:#1f2937; color:#fff; padding:8px 14px; border-radius:14px; margin:4px 0; max-width:70%; align-self:flex-start; }
  .chat-row{ display:flex; width:100%; }
   /* small top padding so the H1 isn't cut off */
  .block-container{
    padding-top: 14px !important;   /* was 0 */
    padding-bottom: 160px !important;
  }
  /* keep the heading from adding extra gap */
  .block-container h1 { margin-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "quick_action" not in st.session_state:
    st.session_state.quick_action = None

# ---------------- model call ----------------
def call_llm(prompt: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"(LLM error) {e}"

# ---------------- title ----------------
st.title("Chat with Pinned Bottom-Right Buttons")

# ---------------- pinned bottom-right actions (compact pill) ----------------
pill = st.container()
with pill:
    # sentinel used by the JS to find & pin the block
    st.markdown("<span id='pin-bottom-right'></span>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Valuation", key="qa_val"):
            st.session_state.quick_action = "Valuation"
    with c2:
        if st.button("Good will", key="qa_gw"):
            st.session_state.quick_action = "Good will"

# Pin the container AND collapse its original wrapper so no gap is left at the top
components.html("""
<script>
(function pin(){
  const d = window.parent.document;
  const mark = d.querySelector('#pin-bottom-right');
  if(!mark) return setTimeout(pin,120);

  const block = mark.closest('div[data-testid="stVerticalBlock"]');
  if(!block) return setTimeout(pin,120);
  if(block.dataset.pinned==="1") return;
  block.dataset.pinned="1";

  // Collapse the host container so it doesn't occupy space up top
  const host = block.closest('div[data-testid="stElementContainer"]');
  if (host) {
    host.style.height = '0px';
    host.style.minHeight = '0';
    host.style.padding = '0';
    host.style.margin = '0';
    host.style.display = 'contents';
  }

  // Float the pill
  Object.assign(block.style, {
    position:'fixed', right:'18px', bottom:'88px', zIndex:'10000',
    display:'inline-flex', gap:'8px', padding:'6px 8px',
    borderRadius:'9999px', background:'rgba(17,24,39,.96)',
    border:'1px solid rgba(255,255,255,.12)', boxShadow:'0 8px 28px rgba(0,0,0,.35)',
    width:'fit-content', maxWidth:'none'
  });

  // Remove column stretch + compact buttons
  Array.from(block.children||[]).forEach(ch => { ch.style.width='auto'; ch.style.margin='0'; });
  block.querySelectorAll('button').forEach(b => { b.style.padding='6px 12px'; b.style.borderRadius='9999px'; });
})();
</script>
""", height=0)

# ---------------- render history ----------------
for m in st.session_state.messages:
    if m["role"] == "user":
        st.markdown(
            f"<div class='chat-row' style='justify-content:flex-end'><div class='user-bubble'>{m['content']}</div></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='chat-row' style='justify-content:flex-start'><div class='assistant-bubble'>{m['content']}</div></div>",
            unsafe_allow_html=True
        )

# ---------------- collect input ----------------
typed = st.chat_input("Type your question…")
user_text = st.session_state.quick_action or typed
st.session_state.quick_action = None

# ---------------- handle a new turn ----------------
if user_text:
    # append user
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.markdown(
        f"<div class='chat-row' style='justify-content:flex-end'><div class='user-bubble'>{user_text}</div></div>",
        unsafe_allow_html=True
    )

    # assistant placeholder
    thinking_slot = st.empty()
    with thinking_slot.container():
        st.markdown(
            "<div class='chat-row' style='justify-content:flex-start'><div class='assistant-bubble'>🧠 Thinking…</div></div>",
            unsafe_allow_html=True
        )

    # call LLM
    reply = call_llm(user_text)

    # replace thinking with final reply
    thinking_slot.markdown(
        f"<div class='chat-row' style='justify-content:flex-start'><div class='assistant-bubble'>{reply}</div></div>",
        unsafe_allow_html=True
    )

    # persist assistant reply
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # rerun so full history shows properly
    st.rerun()
