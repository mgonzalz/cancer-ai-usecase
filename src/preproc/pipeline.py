# Pipeline for preprocessing skin lesion images (SEGMENT-FIRST)
# Pipeline:
#   0) Vignette detection & correction
#   1) First segmentation (on vignette-corrected image)
#   2) Hair removal on the full image
#   3) Lesion-preserved composition (inside = original, outside = hair-removed)
#   4) Augment (train only)
#   5) Export PNG; optional NPY [H,W,4] for RGB+H / RGB+S
#
# Salidas:
#   - PNGs: All images for train/test/external_val
#   - Previews: Just a few 6-panel collages for quick visual inspection
#   - Metadatas: JSON with info per image

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from aug import geo_light
from colors import build_tensor as build_tensor_colors
from hair import compose_lesion_preserved, remove_hair
from segment import segment_lesion
from utils_io import *
from vignette import detect_and_mask_vignette

RNG = random.Random(1337)
random.seed(1337)
IMG_SIZE = 224

def color_overlay(base_rgb_u8: np.ndarray, mask_u8: np.ndarray, color_lo="black", color_hi="yellow", alpha=0.45) -> Image.Image:
    """
    Mask overlay: base RGB + mask (0/255) colored from color_lo to color_hi with alpha blending.
    Args:
        base_rgb_u8: uint8 RGB image (H,W,3).
        mask_u8: uint8 mask (H,W), 0/255.
        color_lo: color for mask=0.
        color_hi: color for mask=255.
        alpha: blending factor in [0,1].
    Returns:
        PIL Image RGB with overlay.
    """
    base = Image.fromarray(base_rgb_u8)  # RGB
    mL = Image.fromarray(mask_u8).convert("L")
    mRGB = ImageOps.colorize(mL, black=color_lo, white=color_hi).convert("RGB")
    return Image.blend(base, mRGB, alpha=alpha)

# Discover images
def discover_images(root: Path, split: str) -> list[Path]:
    out = []
    for label in ("Benign", "Malignant"):
        d = root / split / label
        if not d.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            out += list(d.glob(ext))
    return out


# Process one image
def process_one(path_img: Path, prof: dict, split: str) -> dict:
    """
    Process one image according to the pipeline:
        0) Vignette detection & correction
        1) First segmentation (on vignette-corrected image)
        2) Hair removal on the full image
        3) Lesion-preserved composition (inside = original, outside = hair-removed)
        4) Augment (train only)
        5) Export PNG; optional NPY [H,W,4] for RGB+H / RGB+S
    Args:
        path_img: Path to input image.
        prof: profile dictionary from YAML.
        split: "train", "test", or "external_val".
    Returns:
        dict with outputs and artefacts for collage.
    """
    # Download image and vignette correction
    pil_orig = imread_rgb(str(path_img), img_size=IMG_SIZE)  # ORIGINAL (for collage)
    if prof.get("vignette_mask", True):
        vres = detect_and_mask_vignette(pil_orig)
        pil_v = vres.img_fixed                       # PIL with vignette corrected
        vignette_detected = bool(getattr(vres, "detected", False))
        dark_ratio = float(getattr(vres, "ratio_dark", 0.0))
        vign_mask = getattr(vres, "mask", None)      # or None if not exposed
        if vign_mask is not None:
            # Ensure uint8 0/255
            if vign_mask.dtype != np.uint8:
                vign_mask_u8 = (vign_mask > 0).astype(np.uint8)*255
            else:
                vign_mask_u8 = vign_mask
        else:
            vign_mask_u8 = None
    else:
        pil_v = pil_orig
        vignette_detected, dark_ratio, vign_mask_u8 = False, 0.0, None

    a_orig = np.asarray(pil_orig, dtype=np.uint8)
    a_vfix = np.asarray(pil_v, dtype=np.uint8)

    # 1) Segment FIRST (on vignette-corrected image)
    mask_raw, _lesion_rgb = segment_lesion(a_vfix, size=(IMG_SIZE, IMG_SIZE))
    lesion_mask = mask_raw.astype(np.uint8)  # 0/255

    # 2) Hair removal on the FULL image (use a_vfix as consistent base)
    clean, hair_mask, hair_cov = remove_hair(
        a_vfix, kernel_size=9, bh_threshold=10, inpaint_radius=3, inpaint_method="telea"
    )
    hair_mask = hair_mask.astype(np.uint8)   # 0/255

    # 3) Composición lesion-preserved (inside=fixed original, outside=hair-removed)
    final_lp = compose_lesion_preserved(a_vfix, clean, lesion_mask)

    # 4) Augment (only train)
    out_rgb = final_lp
    aug_level = prof.get("augment_level", "light")
    if split == "train" and aug_level != "none":
        x = out_rgb.astype(np.float32) / 255.0
        x_aug = geo_light(x)
        out_rgb = (x_aug * 255.0 + 0.5).astype(np.uint8)

    # 5) NPY 4ch (optional)
    color_space = prof.get("color_space", "RGB").upper()
    save_npy4 = bool(prof.get("save_npy4", False))
    npy4 = None
    if save_npy4 and color_space in ("RGB+H", "RGB+S"):
        key = "rgb_h" if color_space == "RGB+H" else "rgb_s"
        tensor = build_tensor_colors(out_rgb, profile=key)  # float32 [0,1], 4ch
        npy4 = tensor.astype(np.float32)

    # Collage artefacts
    # 2) Vignette overlay (if mask exists); if not, show original with warning
    if vign_mask_u8 is not None:
        vign_overlay = color_overlay(a_orig, vign_mask_u8, color_lo="black", color_hi="orange", alpha=0.45)
        title_vign = "Vignette (overlay)"
    else:
        vign_overlay = Image.fromarray(a_orig)
        title_vign = "Vignette (overlay: no mask)"

    # 4) Lesion mask (color)
    lesion_vis = ImageOps.colorize(Image.fromarray(lesion_mask).convert("L"),
                                   black="black", white="red").convert("RGB")
    # 5) Hair mask (color)
    hair_vis = ImageOps.colorize(Image.fromarray(hair_mask).convert("L"),
                                 black="black", white="yellow").convert("RGB")

    return {
        "rgb_final": out_rgb,                     # uint8 [H,W,3]
        "hair_mask": hair_mask,                   # uint8 [H,W]
        "lesion_mask": lesion_mask,               # uint8 [H,W]
        "hair_cov": float(hair_cov),
        "vignette_detected": vignette_detected,
        "dark_ratio": float(dark_ratio),
        "vign_overlay_pil": vign_overlay,         # PIL
        "vign_title": title_vign,
        "orig_pil": pil_orig,                     # PIL
        "vfix_pil": pil_v,                        # PIL
        "lesion_vis_pil": lesion_vis,             # PIL (RGB coloreada)
        "hair_vis_pil": hair_vis,                 # PIL (RGB coloreada)
        "clean_rgb": clean,                       # uint8 [H,W,3]
        "npy4": npy4,
        "color_space": color_space,
    }


# Main function: process all images in a split with a given profile
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-root", default="./.cache/images/original")
    ap.add_argument("--split", default="train", choices=["train", "test", "external_val"])
    ap.add_argument("--profiles-yaml", default="config/aug_profiles.yaml")
    ap.add_argument("--profile", required=True, help="Name of the profile in the YAML")
    ap.add_argument("--max-previews-per-class", type=int, default=5)
    args = ap.parse_args()

    img_root = Path(args.img_root)
    proc_root = Path("./.cache/images/processed") / args.profile / args.split
    prev_root = Path("./.cache/images/preview") / args.profile / args.split
    prev_collage_root = prev_root / "collage6"      # <- collage folder
    meta_root = Path("./results/preproc") / args.profile / args.split
    ensure_dir(proc_root)
    ensure_dir(prev_collage_root)
    ensure_dir(meta_root)

    # Load profile
    with open(args.profiles_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prof = cfg["profiles"][args.profile]

    # Discover images
    all_imgs = discover_images(img_root, args.split)
    if not all_imgs:
        print(f"[WARN] no images under {img_root}/{args.split}")
        return

    # Picks for previews
    picks = []
    per_label = {"Benign": [], "Malignant": []}
    triples = []
    for p in all_imgs:
        label = p.parent.name
        triples.append((args.split, label, p))
        if len(per_label[label]) < args.max_previews_per_class:
            per_label[label].append(p)
            picks.append((args.split, label, p))

    meta = []
    for split, label, p in triples:
        out = process_one(p, prof, split)

        # 1) Save final PNG
        rel = Path(label) / (p.stem + ".png")
        out_png = proc_root / rel
        out_png.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out["rgb_final"]).save(out_png)

        # 2) Save NPY4 if applicable
        if out["npy4"] is not None:
            out_npy = proc_root / Path(label) / (p.stem + "__4ch.npy")
            np.save(out_npy, out["npy4"])

        # 3) Collage 6-panel ONLY for picks
        if (split, label, p) in picks:
            collage = six_panel(
                out["orig_pil"],
                out["vign_overlay_pil"],
                out["vfix_pil"],
                out["lesion_vis_pil"],
                out["hair_vis_pil"],
                Image.fromarray(out["rgb_final"]),
                titles=("Original", out["vign_title"], "Vignette corrected",
                        "Lesion mask", "Hair mask", "Final (lesion-preserved)")
            )
            stem = f"{split}__{label}__{p.stem}"
            collage.save(prev_collage_root / f"{stem}__C_collage6.png")
            print(f"[OK][COLLAGE6] {stem} hair_cov={out['hair_cov']:.4f} vignette={out['vignette_detected']}")

        # 4) Metadata
        meta.append({
            "path": str(p),
            "label": label,
            "split": split,
            "out_png": str(out_png),
            "has_npy4": out["npy4"] is not None,
            "color_space": out["color_space"],
            "hair_cov": float(out["hair_cov"]),
            "vignette_detected": bool(out["vignette_detected"]),
            "dark_ratio": float(out["dark_ratio"]),
        })

    # Save metadata JSON
    with open(meta_root / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[OK] PNGs at: {proc_root}")
    print(f"[OK] Collages at: {prev_collage_root}")
    print(f"[OK] Meta at: {meta_root}")


if __name__ == "__main__":
    main()
