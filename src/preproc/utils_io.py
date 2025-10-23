# Utils for I/O and image panel generation
# Purpose: helper functions to read images, list datasets,
# and create side-by-side or multi-panel image comparisons.
#
# Notes:
# - Uses Pillow for image handling and drawing.
# - Functions to create 2, 3, and 4 panel comparisons with titles.
# - Designed for use in preprocessing pipelines.
from __future__ import annotations

import pathlib
import random
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(root: str) -> list[tuple[str, str, str]]:
    """
    Returns [(split,label,path_abs), ...] under root/.cache/images/original/<split>/<label>/*.*
    """
    root = pathlib.Path(root)
    out = []
    for split in ("train", "test", "external_val"):
        for label in ("Benign", "Malignant"):
            p = root / split / label
            if not p.exists():
                continue
            for f in p.rglob("*"):
                if f.suffix.lower() in IMG_EXTS:
                    out.append((split, label, str(f)))
    return out


def pick_random_images(root: str, k: int = 5) -> list[tuple[str, str, str]]:
    random.seed(1337)
    pool = list_images(root)
    random.shuffle(pool)
    return pool[: min(k, len(pool))]


def ensure_dir(p: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def imread_rgb(path: str, img_size: int | None = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img_size:
        img = img.resize((img_size, img_size))
    return img


def pil_textsize(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont
) -> Tuple[int, int]:
    # Compatible con Pillow moderno (textsize deprecado)
    bbox = draw.textbbox((0, 0), text, font=font)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def side_by_side(
    img_left: Image.Image,
    img_right: Image.Image,
    title_left="original",
    title_right="processed",
    pad=8,
    bg=(18, 18, 18),
    fg=(240, 240, 240),
) -> Image.Image:
    w, h = img_left.size
    assert img_right.size == (w, h)
    font = ImageFont.load_default()
    # cabecera
    tmp = Image.new("RGB", (w * 2 + pad * 3, h + pad * 3 + 14), bg)
    draw = ImageDraw.Draw(tmp)
    # títulos
    tw1, th1 = pil_textsize(draw, title_left, font)
    tw2, th2 = pil_textsize(draw, title_right, font)
    ytxt = pad
    draw.text((pad + (w - tw1) // 2, ytxt), title_left, fill=fg, font=font)
    draw.text((pad * 2 + w + (w - tw2) // 2, ytxt), title_right, fill=fg, font=font)
    # imágenes
    tmp.paste(img_left, (pad, pad * 2 + 14))
    tmp.paste(img_right, (pad * 2 + w, pad * 2 + 14))
    return tmp


def three_panel(
    img_a: Image.Image,
    img_b: Image.Image,
    img_c: Image.Image,
    titles=("original", "mask", "overlay"),
    pad=8,
    bg=(18, 18, 18),
    fg=(240, 240, 240),
) -> Image.Image:
    w, h = img_a.size
    assert img_b.size == (w, h) and img_c.size == (w, h)
    font = ImageFont.load_default()
    canvas = Image.new("RGB", (w * 3 + pad * 4, h + pad * 3 + 14), bg)
    draw = ImageDraw.Draw(canvas)
    for i, t in enumerate(titles):
        tw, th = pil_textsize(draw, t, font)
        x = pad + i * (w + pad) + (w - tw) // 2
        draw.text((x, pad), t, fill=fg, font=font)
    y = pad * 2 + 14
    canvas.paste(img_a, (pad, y))
    canvas.paste(img_b, (pad * 2 + w, y))
    canvas.paste(img_c, (pad * 3 + 2 * w, y))
    return canvas


def four_panel(
    img_a: Image.Image,
    img_b: Image.Image,
    img_c: Image.Image,
    img_d: Image.Image,
    titles=("original", "keep-mask (1=útil)", "overlay (zona recorte)", "final"),
    pad=8,
    bg=(18, 18, 18),
    fg=(240, 240, 240),
) -> Image.Image:
    w, h = img_a.size
    assert all(im.size == (w, h) for im in (img_b, img_c, img_d))
    font = ImageFont.load_default()
    canvas = Image.new("RGB", (w * 4 + pad * 5, h + pad * 3 + 14), bg)
    draw = ImageDraw.Draw(canvas)
    for i, t in enumerate(titles):
        tw, th = pil_textsize(draw, t, font)
        x = pad + i * (w + pad) + (w - tw) // 2
        draw.text((x, pad), t, fill=fg, font=font)
    y = pad * 2 + 14
    canvas.paste(img_a, (pad, y))
    canvas.paste(img_b, (pad * 2 + w, y))
    canvas.paste(img_c, (pad * 3 + 2 * w, y))
    canvas.paste(img_d, (pad * 4 + 3 * w, y))
    return canvas
