import os
import streamlit as st
import logic

st.set_page_config(page_title="StockSense", page_icon="📦")


@st.cache_data
def get_df():
    return logic.load_data()


def get_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


df = get_df()
api_key = get_key()
ss = st.session_state
ss.setdefault("messages", [{"role": "assistant", "content": logic.INTRO}])
ss.setdefault("debug", {})

# ---------- sidebar ----------
with st.sidebar:
    st.header("About")
    st.caption("🤖 AI assistant · read-only · synthetic demo data only")
    st.caption(f"Stock snapshot: {logic.SNAPSHOT}")
    st.subheader("Try asking")
    st.markdown(
        "- Do we have BrightSmile Toothpaste 200g in Pune?\n"
        "- What about Delhi?\n- Which warehouse has QuickNoodle 70g?\n"
        "- What's running low in Mumbai?\n- Reserve 50 units for me")
    st.subheader("Demo controls")
    ai_phrase = st.toggle("AI tone rewrite (numbers verified)", value=False)
    outage = st.toggle("Simulate AI outage", value=False)
    show_debug = st.toggle("Show debug panel", value=True)
    if st.button("Reset conversation"):
        for k in list(ss.keys()):
            del ss[k]
        st.rerun()
    st.subheader("Stock manager queue")
    st.caption("Prototype: requests are listed here; no real notification is sent.")
    for e in ss.get("escalations", []):
        st.write(f"🕒 {e['time']} - {e['request']}")

st.title("📦 StockSense")
st.caption("Stock enquiries for Delhi, Mumbai and Pune warehouses. I'm an AI assistant and can make mistakes - "
           "for reservations or anything important, confirm with the stock manager.")
if not api_key:
    st.warning("No API key found - running in keyword-fallback mode.")

for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask about stock..."):
    ss.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    reply, dbg = logic.handle(prompt, ss, df, api_key, ai_phrase, outage)
    ss.messages.append({"role": "assistant", "content": reply})
    ss.debug = dbg
    with st.chat_message("assistant"):
        st.markdown(reply)

if show_debug and ss.debug:
    with st.expander("🔍 Debug panel: how the last answer was produced"):
        st.json(ss.debug)
