import streamlit as st
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from streamlit_app.src.features import (
    asymmetry_index, border_irregularity, color_metrics,
    diameter_pixels, hair_coverage
)

st.set_page_config(page_title="Features ABCD", page_icon="📊", layout="wide")

# Cargar estilos
with open("streamlit_app/static/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📊 Informe ABCD de la lesión")
st.write("""
Este informe calcula métricas clásicas **ABCD** utilizadas en dermatoscopia:
**Asimetría (A)**, **Borde (B)**, **Color (C)** y **Diámetro (D)**, y además reporta la **cobertura de pelo** que podría dificultar la evaluación.
Si no has subido una imagen en la página principal, no se mostrarán resultados.
""")

# Verificamos que hay artefactos en sesión
if "preproc" not in st.session_state:
    st.warning("Primero sube una imagen en la página principal para generar el preprocesamiento.")
    st.stop()

data = st.session_state["preproc"]
orig: Image.Image = data["orig_pil"]
vfix: Image.Image = data["vfix_pil"]
final_img: Image.Image = data["final_pil"]
lesion_mask: np.ndarray = data["lesion_mask"]
hair_mask: np.ndarray = data["hair_mask"]
img_size = data.get("img_size", (orig.width, orig.height))

# Visual arriba: original + overlays
c1, c2, c3 = st.columns(3)
with c1:
    st.caption("Original")
    st.image(orig, use_column_width=True)
with c2:
    st.caption("Máscara de lesión (overlay)")
    lesion_overlay = ImageOps.colorize(Image.fromarray(lesion_mask).convert("L"),
                                       black="black", white="red").convert("RGB")
    st.image(Image.blend(vfix, lesion_overlay, alpha=0.45), use_column_width=True)
with c3:
    st.caption("Máscara de pelo (overlay)")
    hair_overlay = ImageOps.colorize(Image.fromarray(hair_mask).convert("L"),
                                     black="black", white="yellow").convert("RGB")
    st.image(Image.blend(vfix, hair_overlay, alpha=0.45), use_column_width=True)

st.divider()

# Cálculo de métricas
img_rgb = np.asarray(vfix, dtype=np.uint8)
m_u8 = (lesion_mask > 0).astype(np.uint8) * 255
h_u8 = (hair_mask > 0).astype(np.uint8) * 255

A = asymmetry_index(m_u8)                      # 0..1 (más alto = más asimétrico)
B = border_irregularity(m_u8)                  # 0..1 (más alto = borde más irregular)
C = color_metrics(img_rgb, m_u8, k=4)          # HSV stats + clusters significativos
D_px = diameter_pixels(m_u8)                   # en píxeles (no hay escala en mm)
Hair_cov_pct = hair_coverage(h_u8, m_u8)       # % de lesión cubierta por pelo

# Presentación amigable para médico
st.subheader("Resumen clínico (ABCD)")
colA, colB, colC, colD, colH = st.columns([1,1,1,1,1])
colA.metric("A — Asimetría", f"{A:.2f}", help="0=simétrica, 1=asimetría alta")
colB.metric("B — Borde (irregularidad)", f"{B:.2f}", help="0=regular, 1=muy irregular")
colC.metric("C — Colores (clusters ≥5%)", f"{C['clusters_sig']}", help="Nº de grupos de color dentro de la lesión")
colD.metric("D — Diámetro (px)", f"{D_px:.0f}px", help="Diámetro máximo en píxeles (no hay escala en mm)")
colH.metric("Cobertura de pelo", f"{Hair_cov_pct:.1f}%", help="% de la lesión afectada por pelo")

with st.expander("Detalles de color (HSV en la lesión)"):
    st.write(f"- H mean/std: **{C['h_mean']} / {C['h_std']}**")
    st.write(f"- S mean/std: **{C['s_mean']} / {C['s_std']}**")
    st.write(f"- V mean/std: **{C['v_mean']} / {C['v_std']}**")

st.info("""
**Interpretación clínica orientativa**  
- **A (Asimetría)**: valores altos sugieren asimetría en uno o dos ejes.  
- **B (Borde)**: mayor irregularidad puede asociarse a lesiones atípicas.  
- **C (Color)**: múltiples tonalidades (≥3–4 grupos) pueden aumentar la sospecha.  
- **D (Diámetro)**: el valor está en **píxeles** por falta de escala; si se conoce la referencia en mm/píxel, puede convertirse.  
- **Pelo**: una alta cobertura puede enmascarar bordes/texturas; considerar limpieza adicional o repetición de la captura.
""")

# Botón de exportar JSON (pequeño informe)
import json, io
report = {
    "A_asymmetry_0_1": round(A, 3),
    "B_border_irregularity_0_1": round(B, 3),
    "C_color": {
        "clusters_sig": int(C["clusters_sig"]),
        "h_mean": C["h_mean"], "h_std": C["h_std"],
        "s_mean": C["s_mean"], "s_std": C["s_std"],
        "v_mean": C["v_mean"], "v_std": C["v_std"],
    },
    "D_diameter_pixels": D_px,
    "hair_coverage_percent": round(Hair_cov_pct, 2),
    "image_size": img_size
}
buf = io.BytesIO()
buf.write(json.dumps(report, indent=2).encode("utf-8"))
st.download_button("💾 Descargar informe (JSON)", data=buf, file_name="features_abcd_report.json", mime="application/json")
