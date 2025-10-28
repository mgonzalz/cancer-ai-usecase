import sys
from pathlib import Path
import streamlit as st
from PIL import Image

APP_DIR  = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
APP_SRC  = APP_DIR / "src"

for p in (ROOT_DIR, APP_SRC):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from preproc_streamlit import process_uploaded_image
from infer_preproc import predict_proba_malignant

# --- Config de página ---
st.set_page_config(
    page_title="SkinBridge Inference + Preprocessing",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estilos ---
with open("streamlit_app/static/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Header ---
st.markdown(Path("streamlit_app/templates/header.html").read_text(), unsafe_allow_html=True)
st.markdown("""
<div style="background-color:#f7f9fc; padding:20px; border-radius:10px; margin-bottom:20px;">
<h3 style="color:#0b5394; margin-top:-5px;">🧠 ¿Cómo funciona SkinBridge?</h3>
<p style="font-size:1.05em; color:#1a1a1a;">
<b>SkinBridge</b> utiliza una <b>red neuronal convolucional (CNN)</b> diseñada para analizar imágenes dermatoscópicas y 
asistir en la detección temprana de lesiones cutáneas potencialmente malignas.
Antes de que la red procese las imágenes, se aplica un <b>pipeline de preprocesamiento</b> automático que:
</p>
<ul style="color:#1a1a1a; font-size:1.05em;">
    <li>Corrige artefactos de <b>viñeteado</b> (bordes oscuros de la lente).</li>
    <li><b>Elimina pelos</b> que puedan ocultar partes del lunar.</li>
    <li><b>Segmenta la lesión</b> para aislarla del fondo.</li>
    <li>Compone una imagen final que conserva la lesión intacta y limpia el entorno.</li>
</ul>
<p style="font-size:1.05em; color:#1a1a1a;">
El resultado se muestra como un <b>collage de 6 paneles</b> para que el profesional pueda visualizar
cada etapa del proceso antes de la inferencia del modelo.
Posteriormente, la CNN utilizará la imagen procesada para su clasificación (Benigno/Maligno).
</p>
</div>
""", unsafe_allow_html=True)
st.subheader("📤 Sube tu imagen dermatoscópica")
uploaded_file = st.file_uploader("Selecciona una imagen...", type=["jpg", "jpeg", "png"])

# --- Umbral de decisión ---
th = st.slider("Umbral de decisión para 'Malignant' (0–1)", 0.00, 1.00, 0.50, 0.01,
               help="Si p(Malignant) ≥ umbral ➜ predicción Malignant; en caso contrario, Benign.")

def classify_health_profile(p_mal: float, th: float) -> tuple[str, str, str, tuple[float,float,float,float]]:
    """
    Perfiles adaptativos al umbral:
      D — Green Safe:        p < th
      C — Ivory Zone:        th ≤ p < th+0.25
      B — Amber Watch:       th+0.25 ≤ p < th+0.40
      A — Crimson Alert:     p ≥ th+0.40
    (cortes se recortan a [0,1])
    Devuelve: (nombre, descripción, color_hex, (cD,cC,cB,cA) con límites efectivos)
    """
    cD = max(0.0, min(1.0, th))
    cC = max(0.0, min(1.0, th + 0.25))
    cB = max(0.0, min(1.0, th + 0.40))
    cA = 1.0  # techo

    if p_mal >= cB:   # A
        return ("🟥 Crimson Alert", "Alta probabilidad de malignidad", "#b30000", (cD, cC, cB, cA))
    elif p_mal >= cC: # B
        return ("🟧 Amber Watch", "Probabilidad moderada de malignidad", "#ff9900", (cD, cC, cB, cA))
    elif p_mal >= cD: # C
        return ("⬜ Ivory Zone", "Caso ambiguo, requiere valoración clínica", "#e6e6e6", (cD, cC, cB, cA))
    else:             # D
        return ("🟩 Green Safe", "Probablemente benigno", "#00b050", (cD, cC, cB, cA))


# --- Lógica principal ---
if uploaded_file is not None:
    try:
        # Procesamos primero para obtener la imagen final (sin mostrar collage aún)
        st.info("Procesando la imagen y generando predicción…")
        collage_img = process_uploaded_image(uploaded_file, profile="rgb_base", split="test")

        # Recuperamos la imagen final del pipeline
        data = st.session_state.get("preproc", {})
        final_pil: Image.Image = data.get("final_pil", None)

        if final_pil is None:
            st.error("❌ No se generó la imagen final. Revisa el pipeline.")
        else:
            # --- INFERENCIA DEL MODELO ---
            p_mal = predict_proba_malignant(final_pil)
            p_ben = 1.0 - p_mal
            pred_label = "🩸 Malignant" if p_mal >= th else "🩺 Benign"
            # Mostramos primero los resultados
            st.subheader("🧠 Resultados del modelo")
            st.markdown(f"<h2 style='text-align:center; color:#0b5394;'>{pred_label}</h2>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.metric("Prob. Malignant", f"{p_mal:.3f}")
            with c2:
                st.metric("Prob. Benign", f"{p_ben:.3f}")
            with c3:
                st.metric("Umbral actual", f"{th:.2f}")

            level_name, level_desc, level_color, (cD, cC, cB, cA) = classify_health_profile(p_mal, th)

            # Recuadro coloreado
            st.markdown(f"""
            <div style="
                background-color:{level_color};
                color:{'white' if (level_name.startswith('🟥') or level_name.startswith('🟩')) else 'black'};
                border-radius:10px; padding:16px; text-align:center; margin-top:15px;
                font-size:1.2em; font-weight:bold;">
                {level_name} — {level_desc}
            </div>
            """, unsafe_allow_html=True)

            # Rango adaptativo actualmente usado (opcional)
            st.caption(
                f"Perfiles (adaptados al umbral {th:.2f}): "
                f"🟩 D: p < {cD:.2f} · "
                f"⬜ C: [{cD:.2f}, {cC:.2f}) · "
                f"🟧 B: [{cC:.2f}, {cB:.2f}) · "
                f"🟥 A: ≥ {cB:.2f}"
            )

            st.progress(min(max(p_mal, 0.0), 1.0), text="Probabilidad de Malignant")
            st.caption("La red utiliza la **imagen preprocesada lesion-preserved** (224×224, RGB).")

            st.divider()

            # --- COLLAJE DE PREPROCESAMIENTO ---
            st.subheader("🧩 Etapas del preprocesamiento")
            if collage_img is not None:
                st.image(collage_img, caption="Collage de 6 paneles del pipeline", use_column_width=True)
            else:
                st.warning("⚠️ No se pudo generar el collage visual.")
    except Exception as e:
        st.error(f"❌ Error durante el procesamiento o la inferencia: {e}")

# --- Footer ---
st.markdown(Path("streamlit_app/templates/footer.html").read_text(), unsafe_allow_html=True)
