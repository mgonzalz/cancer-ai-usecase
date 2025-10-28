import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from streamlit_app.src.preproc_utils import six_panel_collage, color_overlay

# --- Rutas robustas ---
APP_DIR  = Path(__file__).resolve().parents[1]    # .../streamlit_app
ROOT_DIR = APP_DIR.parent                         # raíz del repo
TOP_SRC  = ROOT_DIR / "src"                       # <raíz>/src (tu librería)
APP_SRC  = APP_DIR / "src"                        # streamlit_app/src (por si copias preproc ahí)

for p in (ROOT_DIR, TOP_SRC, APP_SRC):
    if str(p) not in sys.path:
        sys.path.append(str(p))

# --- Importa (3 intentos) ---
try:
    # preferido: <raíz>/src/preproc/...
    from src.preproc.vignette import detect_and_mask_vignette
    from src.preproc.segment import segment_lesion
    from src.preproc.hair import remove_hair, compose_lesion_preserved
    from src.preproc.aug import geo_light
    from src.preproc.colors import build_tensor as build_tensor_colors
except ModuleNotFoundError:
    try:
        # alternativa: <raíz>/src como path directo (sin prefijo src)
        from preproc.vignette import detect_and_mask_vignette
        from preproc.segment import segment_lesion
        from preproc.hair import remove_hair, compose_lesion_preserved
        from preproc.aug import geo_light
        from preproc.colors import build_tensor as build_tensor_colors
    except ModuleNotFoundError:
        # última opción: streamlit_app/src/preproc/...
        from streamlit_app.src.preproc.vignette import detect_and_mask_vignette
        from streamlit_app.src.preproc.segment import segment_lesion
        from streamlit_app.src.preproc.hair import remove_hair, compose_lesion_preserved
        from streamlit_app.src.preproc.aug import geo_light
        from streamlit_app.src.preproc.colors import build_tensor as build_tensor_colors

IMG_SIZE = 224

def process_uploaded_image(file, profile="rgb_base", split="test"):
    """
    Procesa una imagen subida a Streamlit aplicando el pipeline segment-first.
    Soporta perfiles de color: rgb_base, rgb_h, rgb_s.
    Devuelve un collage PIL.Image para visualizar el flujo.
    """
    try:
        # Leer imagen
        img = Image.open(file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        a_orig = np.asarray(img, dtype=np.uint8)

        # 1) Viñeteado
        vres = detect_and_mask_vignette(img)
        pil_v = vres.img_fixed
        vign_mask = getattr(vres, "mask", None)
        a_vfix = np.asarray(pil_v, dtype=np.uint8)

        # 2) Segmentación
        mask_raw, _ = segment_lesion(a_vfix, size=(IMG_SIZE, IMG_SIZE))
        lesion_mask = mask_raw.astype(np.uint8)

        # 3) Pelo
        clean, hair_mask, _ = remove_hair(a_vfix)
        hair_mask = hair_mask.astype(np.uint8)

        # 4) Composición final
        final_lp = compose_lesion_preserved(a_vfix, clean, lesion_mask)

        # 5) Espacio de color según perfil
        color_space = {
            "rgb_base": "RGB",
            "rgb_h": "RGB+H",
            "rgb_s": "RGB+S"
        }.get(profile, "RGB")

        npy4 = None
        if color_space in ("RGB+H", "RGB+S"):
            key = "rgb_h" if color_space == "RGB+H" else "rgb_s"
            tensor = build_tensor_colors(final_lp, profile=key)  # float32 [0,1,4]
            npy4 = tensor.astype(np.float32)

            # Para mostrar visualmente el canal extra:
            channel_extra = npy4[..., 3]  # H o S
            ch_vis = (channel_extra * 255).astype(np.uint8)
            ch_vis_rgb = ImageOps.colorize(Image.fromarray(ch_vis).convert("L"),
                                           black="black", white="cyan").convert("RGB")
            extra_label = "Canal extra (H)" if color_space == "RGB+H" else "Canal extra (S)"
        else:
            ch_vis_rgb = Image.fromarray(final_lp)
            extra_label = "Espacio RGB"

        # 6) Crear collage
        vign_overlay = color_overlay(a_orig, vign_mask, color_hi="orange", alpha=0.45) if vign_mask is not None else img
        lesion_vis = ImageOps.colorize(Image.fromarray(lesion_mask).convert("L"), black="black", white="red").convert("RGB")
        hair_vis = ImageOps.colorize(Image.fromarray(hair_mask).convert("L"), black="black", white="yellow").convert("RGB")

        collage = six_panel_collage(
            img, vign_overlay, pil_v, lesion_vis, hair_vis, ch_vis_rgb,
            titles=("Original", "Viñeteado", "Viñeta corregida", "Máscara de lesión", "Máscara de pelo", extra_label)
        )
        st.session_state["preproc"] = {
            "orig_pil": img,                         # PIL
            "vfix_pil": pil_v,                       # PIL
            "final_pil": Image.fromarray(final_lp),  # PIL
            "lesion_mask": lesion_mask,              # np.uint8 [0,255]
            "hair_mask": hair_mask,                  # np.uint8 [0,255]
            "hair_cov": float(_),                    # si tu remove_hair devuelve cobertura; si no, pon 0.0
            "img_size": (IMG_SIZE, IMG_SIZE),
        }
        return collage

    except Exception as e:
        print("❌ Error:", e)
        return None
