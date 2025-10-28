import streamlit as st
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Explicación de la CNN", page_icon="🧠", layout="wide")

# Estilos (blanco + azules + Inter)
with open("streamlit_app/static/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🧠 Explicación de la Red Convolucional (CNN)")
st.write("""
Esta sección resume de forma didáctica cómo **nuestra CNN** procesa las imágenes dermatoscópicas.
La red aprende **patrones de textura, bordes y color** relevantes para distinguir entre lesiones **benignas** y **malignas**.
""")

with st.expander("¿Qué es una CNN? (resumen para clínicos)"):
    st.markdown("""
- **Convoluciones**: filtros que resaltan patrones (bordes, texturas, pigmentación).
- **Pooling**: reduce el tamaño manteniendo la información esencial (robustez al ruido).
- **Capas densas**: combinan las pistas extraídas para tomar la decisión final.
- **Regularización** (BatchNorm/Dropout): estabiliza el aprendizaje y reduce sobreajuste.
- **Salida**: probabilidad de **Maligno** (o vector de clases).
""")

st.subheader("Arquitectura de la red")
img_path = Path("streamlit_app/static/cnn_architecture.png")  # <-- coloca aquí tu diagrama
if img_path.exists():
    st.image(str(img_path), caption="Diagrama de la arquitectura CNN", use_column_width=True)
else:
    st.warning("No se encontró `streamlit_app/static/cnn_architecture.png`. Añade tu diagrama para verlo aquí.")

with st.expander("¿Cómo se relaciona con el preprocesamiento?"):
    st.write("""
- **Máscara de lesión**: la composición *lesion-preserved* protege la **textura** intra-lesional.
- **Hair removal**: evita falsos bordes o patrones espurios por pelo.
- **Color space** (`RGB`, `RGB+H`, `RGB+S`): añade información cromática relevante sin distorsionar.
""")
