# pages/2_GradCAM.py
import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import matplotlib.cm as cm
from streamlit_app.src.infer_preproc import load_model

st.set_page_config(page_title="Grad-CAM — Interpretación Visual", page_icon="🔥", layout="wide")

# --- Estilos globales ---
with open("streamlit_app/static/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🔥 Grad-CAM — Interpretación Visual del Modelo")

# --- Verificamos que haya imagen en sesión ---
if "preproc" not in st.session_state:
    st.warning("Primero sube una imagen en la página principal para generar el preprocesamiento.")
    st.stop()

data = st.session_state["preproc"]
final_pil: Image.Image = data["final_pil"]
IMG_SIZE = (224, 224)

model = load_model()

# --- Capas convolucionales disponibles ---
conv_layers = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
if not conv_layers:
    st.error("❌ El modelo no contiene capas Conv2D. No se puede generar Grad-CAM.")
    st.stop()

selected_layer = st.selectbox("🧩 Selecciona la capa convolucional a analizar:", conv_layers, index=max(len(conv_layers)-2, 0))

with st.expander("Explicación de las convoluciones"):
# --- Explicación de las capas convolucionales ---
    st.markdown("""
    <hr style='margin-top:30px; margin-bottom:20px;'>

    ### 🧩 **¿Qué son las convoluciones y cómo interpretarlas en tu modelo?**

    Las capas **convolucionales (Conv2D)** son el núcleo de una CNN.  
    Cada una aprende a detectar **patrones visuales de distinta complejidad** dentro de la imagen dermatoscópica:

    ---

    #### 1️⃣ Primera capa — *Detección básica de bordes y color*
    <div style='background-color:#eaf3fc; padding:10px; border-radius:10px;'>
    <ul>
    <li><b>Qué ve:</b> líneas, bordes, contrastes simples, transiciones de color.</li>
    <li><b>Por qué es útil:</b> delimita la lesión del fondo y reconoce los contornos.</li>
    <li><b>Grad-CAM:</b> suele resaltar el borde del lunar o zonas externas uniformes.</li>
    </ul>
    </div>

    ---

    #### 2️⃣ Segunda capa — *Texturas y patrones intermedios*
    <div style='background-color:#fff5e6; padding:10px; border-radius:10px;'>
    <ul>
    <li><b>Qué ve:</b> texturas, moteados, irregularidades de pigmento o relieve.</li>
    <li><b>Por qué es útil:</b> capta la <b>heterogeneidad cromática</b>, clave para valorar malignidad.</li>
    <li><b>Grad-CAM:</b> muestra calor en áreas internas del lunar, especialmente pigmentadas.</li>
    </ul>
    </div>

    ---

    #### 3️⃣ Tercera capa — *Patrones globales de alto nivel*
    <div style='background-color:#eafbea; padding:10px; border-radius:10px;'>
    <ul>
    <li><b>Qué ve:</b> formas abstractas combinadas, asimetría, bordes irregulares.</li>
    <li><b>Por qué es útil:</b> integra toda la información anterior para decidir la clase final.</li>
    <li><b>Grad-CAM:</b> concentra la atención en la <b>zona diagnóstica principal</b> de la lesión.</li>
    </ul>
    </div>

    ---

    ### 🩺 **¿Cuál capa conviene inspeccionar?**
    - Si deseas ver **cómo empieza a "mirar" la red**, selecciona la **primera capa**.  
    - Para explorar **la textura y color internos**, observa la **segunda capa**.  
    - Para interpretar **la decisión final del modelo**, usa la **última capa convolucional**.

    > 💡 En la práctica clínica, la **última capa conv** suele ser la más informativa,  
    > pues indica dónde el modelo ha concentrado su atención al clasificar la lesión.
    """, unsafe_allow_html=True)

# --- Función Grad-CAM ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Devuelve un heatmap normalizado (0-1)"""
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) == 0:
        return np.zeros_like(heatmap)
    return heatmap / np.max(heatmap)

# --- Preparar entrada ---
img_array = np.expand_dims(np.asarray(final_pil.resize(IMG_SIZE), dtype=np.float32) / 255.0, axis=0)
preds = model.predict(img_array, verbose=0)
pred_label = int(np.argmax(preds))
p_mal = float(preds[0, -1]) if preds.shape[1] == 2 else float(preds[0, 0])

# --- Generar heatmap ---
heatmap = make_gradcam_heatmap(img_array, model, selected_layer, pred_index=pred_label)

# --- Convertir heatmap a mapa de color (JET azul→rojo) ---
heatmap = np.uint8(255 * heatmap)
jet = cm.get_cmap("jet")
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]

jet_heatmap = (jet_heatmap * 255).astype(np.uint8)
jet_heatmap = Image.fromarray(jet_heatmap).resize(final_pil.size)
jet_heatmap = np.asarray(jet_heatmap)

# --- Superponer sobre la imagen original ---
base = np.asarray(final_pil.convert("RGB"), dtype=np.uint8)
overlay = cv2.addWeighted(base, 0.6, jet_heatmap, 0.4, 0)

overlay_pil = Image.fromarray(overlay)

# --- Layout visual ---
st.subheader("🩺 Interpretación visual")
col1, col2 = st.columns(2)
with col1:
    st.image(final_pil, caption="Imagen preprocesada (entrada del modelo)", use_column_width=True)
with col2:
    st.image(overlay_pil, caption=f"Grad-CAM ({selected_layer}) — mapa térmico azul→rojo", use_column_width=True)

st.markdown(f"""
**Predicción del modelo:**  
- Probabilidad de *Malignant*: **{p_mal:.3f}**  
- Capa inspeccionada: `{selected_layer}`
""")

with st.expander("ℹ️ Interpretación clínica del Grad-CAM"):
    st.write("""
    El mapa **Grad-CAM** colorea las zonas más relevantes para la decisión del modelo:
    - **Rojo intenso / Amarillo** → áreas con **alta activación** (más influyentes).
    - **Verde / Azul** → regiones **menos relevantes**.
    - Una concentración roja sobre la lesión indica que la red se centra en la región tumoral.
    - Si el calor aparece fuera de la lesión, podría indicar **foco erróneo o artefacto** (pelo, borde, etc.).
    """)

st.divider()
st.markdown(Path("streamlit_app/templates/footer.html").read_text(), unsafe_allow_html=True)
