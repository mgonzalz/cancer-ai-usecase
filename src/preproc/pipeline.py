# Pipeline for preprocessing skin lesion images:
# 1) Vignette detection and correction
# 2) Hair removal
# 3) Lesion segmentation and lesion-preserved composition
# 4) Optional augmentation (geometric + light)
# Purpose: Prepare images for training/testing with different color spaces (RGB, RGB+H, RGB+S)
#
# Notes: This pipeline applies a series of preprocessing steps to each image, ensuring
# that the input data is consistent and suitable for model training and evaluation.

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from aug import geo_light
from colors import build_tensor as build_tensor_colors  # para npy4
from hair import compose_lesion_preserved, remove_hair
from PIL import Image, ImageDraw, ImageFont, ImageOps
from segment import clean_mask_with_hair, segment_lesion
from utils_io import ensure_dir, imread_rgb
from vignette import detect_and_mask_vignette

RNG = random.Random(1337)
IMG_SIZE = 224


def discover_images(root: Path, split: str) -> list[Path]:
    """Discover all images under root/split/{Benign,Malignant} with common extensions.
    Args:
        - root: Path to the root image directory.
        - split: Dataset split ('train' or 'test').
    Returns:
        - A list of Paths.
    """
    out = []
    for label in ("Benign", "Malignant"):
        d = root / split / label
        if not d.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            out += list(d.glob(ext))
    return out


def make_preview_1x4(
    out_png: Path, orig_u8: np.ndarray, hair_mask_u8: np.ndarray, final_u8: np.ndarray
):
    """[Original | Hair mask | Mask over image | Final lesion-preserved]
    Args:
        - out_png: Path to save the preview PNG.
        - orig_u8: Original image as uint8 [H,W,3].
        - hair_mask_u8: Hair mask as uint8 [H,W] (0/255).
        - final_u8: Final lesion-preserved image as uint8 [H,W,3].
    Returns:
        - None (saves PNG to out_png).
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    W, H = orig_u8.shape[1], orig_u8.shape[0]
    pad = 10
    canvas = Image.new("RGB", (W * 4 + pad * 5, H + 60), (18, 18, 18))
    font = Image.font = (
        ImageFont.load_default() if hasattr(Image, "font") else ImageFont.load_default()
    )
    draw = ImageDraw.Draw(canvas)
    img_orig = Image.fromarray(orig_u8)
    # Hair mask colorizada
    mask_vis = Image.fromarray(hair_mask_u8).convert("L")
    mask_rgb = ImageOps.colorize(mask_vis, black="black", white="yellow").convert("RGB")
    # Overlay
    overlay = cv2.addWeighted(
        orig_u8, 0.7, cv2.cvtColor(hair_mask_u8, cv2.COLOR_GRAY2RGB), 0.3, 0
    )
    overlay = Image.fromarray(overlay)
    final_img = Image.fromarray(final_u8)
    tiles = [
        ("Original (vignette-fixed)", img_orig),
        ("Hair mask", mask_rgb),
        ("Mask over image", overlay),
        ("Final (lesion preserved)", final_img),
    ]
    x = pad
    for title, pil in tiles:
        canvas.paste(pil, (x, 40))
        tw, th = draw.textbbox((0, 0), title, font=font)[2:]
        draw.text((x + (W - tw) // 2, 12), title, fill=(240, 240, 240), font=font)
        x += W + pad
    canvas.save(out_png)


def process_one(path_img: Path, prof: dict, split: str) -> dict:
    """
    Vignette -> Hair removal -> Segment -> Lesion-preserved compose
    Aug only if split == 'train' and augment_level == 'light'
    Export: PNG (RGB). If color_space in {RGB+H, RGB+S} and save_npy4, also export .npy [H,W,4].
    Args:
        - path_img: Path to the input image.
        - prof: Augmentation profile dictionary.
        - split: Dataset split ('train' or 'test').
    Returns:
        - A dictionary with:
            - "rgb_final": Final RGB image as uint8 [H,W,3].
            - "hair_mask": Hair mask as uint8 [H,W] (0/255).
            - "mask_clean": Cleaned lesion mask as uint8 [H,W] (0/255).
            - "hair_cov": Hair coverage ratio (float).
            - "vignette_detected": Whether vignette was detected (bool).
            - "dark_ratio": Dark pixel ratio in vignette area (float).
            - "npy4": Optional 4-channel tensor as float32 [H,W,4] or None.
            - "color_space": Color space used ("RGB", "RGB+H", or "RGB+S").
    """
    # 1) Load & resize
    pil = imread_rgb(str(path_img), img_size=IMG_SIZE)
    # 2) Vignette (Always ON in our profiles)
    if prof.get("vignette_mask", True):
        vres = detect_and_mask_vignette(pil)
        pil_v = vres.img_fixed
        vignette_detected = bool(vres.detected)
        dark_ratio = float(vres.ratio_dark)
    else:
        pil_v = pil
        vignette_detected = False
        dark_ratio = 0.0

    # 3) Hair removal (Always ON in our profiles)
    a_v = np.asarray(pil_v, dtype=np.uint8)
    clean, hair_mask, hair_cov = remove_hair(
        a_v, kernel_size=9, bh_threshold=10, inpaint_radius=3, inpaint_method="telea"
    )

    # 4) Segmentation (if enabled)
    if prof.get("segment", True):
        mask_raw, _ = segment_lesion(clean, size=(IMG_SIZE, IMG_SIZE))
        mask_clean = clean_mask_with_hair(mask_raw, hair_mask)
    else:
        mask_clean = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)

    # 5) Lesion-preserved compose
    final_lp = compose_lesion_preserved(
        a_v, clean, mask_clean
    )  # original=vignette-fixed, fondo=clean

    # 6) Augment (Only if train split and augment_level != 'none')
    aug_level = prof.get("augment_level", "light")
    out_rgb = final_lp
    if split == "train" and aug_level != "none":
        x = out_rgb.astype(np.float32) / 255.0
        x_aug = geo_light(x)
        out_rgb = (x_aug * 255.0 + 0.5).astype(np.uint8)

    # 7) Optional: 4-channel NPY if color_space in {RGB+H, RGB+S}
    color_space = prof.get("color_space", "RGB").upper()
    save_npy4 = bool(prof.get("save_npy4", False))
    npy4 = None
    if save_npy4 and color_space in ("RGB+H", "RGB+S"):
        if color_space == "RGB+H":
            tensor = build_tensor_colors(out_rgb, profile="rgb_h")  # Float32 [0,1], 4ch
        else:
            tensor = build_tensor_colors(out_rgb, profile="rgb_s")
        npy4 = tensor.astype(np.float32)  # [H,W,4] Float32

    return {
        "rgb_final": out_rgb,
        "hair_mask": hair_mask,
        "mask_clean": mask_clean,
        "hair_cov": hair_cov,
        "vignette_detected": vignette_detected,
        "dark_ratio": dark_ratio,
        "npy4": npy4,
        "color_space": color_space,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-root", default="./.cache/images/original")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--profiles-yaml", default="config/aug_profiles.yaml")
    ap.add_argument(
        "--profile",
        required=True,
        help="Name in profiles YAML, e.g., rgb_base | rgb_h | rgb_s",
    )
    ap.add_argument("--max-previews-per-class", type=int, default=5)
    args = ap.parse_args()

    img_root = Path(args.img_root)
    proc_root = Path("./.cache/images/processed") / args.profile / args.split
    prev_root = Path("./.cache/images/preview") / args.profile / args.split
    meta_root = Path("./results/preproc") / args.profile / args.split
    ensure_dir(proc_root)
    ensure_dir(prev_root)
    ensure_dir(meta_root)

    # Load profiles
    with open(args.profiles_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prof = cfg["profiles"][args.profile]

    # Discover images (limit previews)
    all_imgs = discover_images(img_root, args.split)
    if not all_imgs:
        print(f"[WARN] no images under {img_root}/{args.split}")
        return

    # For previews, pick up to N per class
    picks = []
    per_label = {"Benign": [], "Malignant": []}
    for p in all_imgs:
        label = p.parent.name
        if len(per_label[label]) < args.max_previews_per_class:
            per_label[label].append(p)
            picks.append(p)

    # Process everything (preview subset also gets preview)
    meta = []
    for p in all_imgs:
        label = p.parent.name
        out = process_one(p, prof, args.split)

        # Save PNG (final lesion-preserved, augmented if train)
        rel = Path(label) / (p.stem + ".png")
        out_png = proc_root / rel
        out_png.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out["rgb_final"]).save(out_png)

        # Optional NPY 4ch
        if out["npy4"] is not None:
            out_npy = proc_root / Path(label) / (p.stem + "__4ch.npy")
            np.save(out_npy, out["npy4"])

        # Previews for sampled ones
        if p in picks:
            prev_png = prev_root / (label + "__" + p.stem + "__panel4.png")
            make_preview_1x4(
                prev_png,
                orig_u8=np.asarray(imread_rgb(str(p), IMG_SIZE)),
                hair_mask_u8=out["hair_mask"],
                final_u8=out["rgb_final"],
            )

        meta.append(
            {
                "path": str(p),
                "label": label,
                "split": args.split,
                "out_png": str(out_png),
                "has_npy4": out["npy4"] is not None,
                "color_space": out["color_space"],
                "hair_cov": float(out["hair_cov"]),
                "vignette_detected": bool(out["vignette_detected"]),
                "dark_ratio": float(out["dark_ratio"]),
            }
        )

    # Save run metadata
    with open(meta_root / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[OK] PNGs at: {proc_root}")
    print(f"[OK] Previews at: {prev_root}")
    print(f"[OK] Meta at: {meta_root}")


if __name__ == "__main__":
    main()
