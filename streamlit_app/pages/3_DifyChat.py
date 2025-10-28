# pages/3_DifyChat.py
import streamlit as st
from pathlib import Path
from streamlit.components.v1 import html

st.set_page_config(page_title="Dify Chat", page_icon="💬", layout="wide")

# Estilos globales (Inter + blanco/azules)
with open("streamlit_app/static/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Cabecera simple
st.title("💬 SkinBridge Assistant")
st.write(
    "Chat with SkinBridge Assistant. The microphone is allowed from the iframe if your browser permits it."
)

# ---- IFRAME DIFY (tu snippet) ----
iframe = iframe = """
<div style="background-color:#ffffff; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,0.1); padding:10px;">
    <iframe
        src="https://udify.app/chatbot/iT4a9IIkVncAS0Ak"
        style="width:100%; height:680px; background-color:#ffffff; border:none; border-radius:10px;"
        frameborder="0"
        allow="microphone">
    </iframe>
</div>
"""

# Render del iframe en Streamlit
# Ajusta 'height' si quieres más/menos espacio vertical
html(iframe, height=600)

# Pie de página opcional
st.markdown("<p style='color:#555;'>If you don't see the chat, open this link in another tab or check blockers.</p>", unsafe_allow_html=True)
