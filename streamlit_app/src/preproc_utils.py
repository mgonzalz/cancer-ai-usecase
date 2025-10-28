# src/preproc_utils.py
from __future__ import annotations
from PIL import Image, ImageDraw, ImageFont, ImageOps

# -----------------------------
# 🔤 Función auxiliar para centrar títulos
# -----------------------------
def _title_center(draw: ImageDraw.ImageDraw, x: int, W: int, y: int, text: str, font):
    """Escribe un título centrado encima de cada panel."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((x + (W - tw)//2, y), text, fill=(240, 240, 240), font=font)


# -----------------------------
# 🎨 Overlay coloreado (para viñeteado, pelo, etc.)
# -----------------------------
def color_overlay(base_rgb_u8, mask_u8, color_lo="black", color_hi="orange", alpha=0.45):
    """
    Colorea una máscara (0/255) y la mezcla con la imagen base.
    - base_rgb_u8: np.ndarray [H,W,3], uint8
    - mask_u8: np.ndarray [H,W], uint8 0/255
    - color_lo / color_hi: colores de mapeo
    - alpha: opacidad de superposición
    """
    base = Image.fromarray(base_rgb_u8)
    mL = Image.fromarray(mask_u8).convert("L")
    mRGB = ImageOps.colorize(mL, black=color_lo, white=color_hi).convert("RGB")
    return Image.blend(base, mRGB, alpha=alpha)


# -----------------------------
# 🧩 Collage 6-panel (3x2)
# -----------------------------
def six_panel_collage(img1, img2, img3, img4, img5, img6, titles=None):
    """
    Crea un collage 3x2 con títulos.
    - img1..img6: PIL.Image (224x224 aprox)
    - titles: lista o tupla de 6 strings
    """
    if titles is None:
        titles = ("1", "2", "3", "4", "5", "6")

    W, H = img1.size
    pad = 10
    topbar = 50
    cols, rows = 3, 2
    canvas = Image.new("RGB", (cols*W + (cols+1)*pad, rows*H + (rows+1)*pad + topbar), (18, 18, 18))
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)

    tiles = [img1, img2, img3, img4, img5, img6]
    x = pad
    y = pad + topbar
    for i, pil in enumerate(tiles):
        if pil.size != (W, H):
            pil = pil.resize((W, H))
        canvas.paste(pil, (x, y))
        _title_center(draw, x, W, y - 30, titles[i], font)
        x += W + pad
        if (i+1) % 3 == 0:
            x = pad
            y += H + pad

    return canvas


# -----------------------------
# 🧱 Collage 4-panel (opcional)
# -----------------------------
def four_panel_collage(img1, img2, img3, img4, titles=None):
    """
    Variante 2x2 por si quieres menos paneles.
    """
    if titles is None:
        titles = ("1", "2", "3", "4")

    W, H = img1.size
    pad = 10
    topbar = 50
    canvas = Image.new("RGB", (2*W + 3*pad, 2*H + 3*pad + topbar), (18, 18, 18))
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)

    tiles = [img1, img2, img3, img4]
    x = pad
    y = pad + topbar
    for i, pil in enumerate(tiles):
        if pil.size != (W, H):
            pil = pil.resize((W, H))
        canvas.paste(pil, (x, y))
        _title_center(draw, x, W, y - 30, titles[i], font)
        x += W + pad
        if (i+1) % 2 == 0:
            x = pad
            y += H + pad

    return canvas
