# Vignette Detector & Masker (Radial Profile + Hough Fallback)
# Purpose:
#   Detect circular peripheral darkening (vignetting) in images and mask out
#   the outer ring so only the central, well-lit region remains. Produces:
#     - A masked RGB image with the black border applied,
#     - A binary keep-mask (1=useful disk, 0=masked periphery),
#     - A boolean "detected" flag, and
#     - A quick "Ratio Dark" metric describing how dark the masked ring is.
#
# Method (Robust, two stages):
#   1) Radial profile:
#      - Compute the mean brightness for thin concentric rings from the center.
#      - Compare the ring brightness to a fraction of the central brightness.
#      - Use the earliest strong drop (validated by the gradient) as the border.
#   2) Fallback with HoughCircles (optional):
#      - If the radial profile is inconclusive, detect a large circle and accept
#        it as the border if the outside ring is sufficiently darker than center.
#
# When to tweak:
#   - inner_frac: size of the central disk used to estimate “good” brightness.
#   - drop_k: ring is considered vignetted if brightness < (drop_k * center).
#   - min_margin: minimum relative drop required to accept a detection.
#   - min_edge_frac / max_edge_frac: radial search range for the border.
#
# Notes:
#   - Input images are expected as PIL.Image (RGB).
#   - OpenCV is used for grayscale conversion, blurring, and HoughCircles.

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class VignetteResult:
    """
    Container for the vignette detection/masking output.
    Attributes:
        - img_fixed (PIL.Image.Image): RGB image where the outer ring (beyond the
            detected edge) is masked to black. If no vignette is detected, it is
            the original image.
        - ring_mask (np.ndarray): Binary mask (H, W), where 1 marks the useful inner disk
            that is kept, and 0 marks the outer ring that is masked.
        - detected (bool): True if a likely vignette border was detected (by radial profile
            or Hough fallback); False otherwise.
        - ratio_dark (float): Quick estimate of the fraction of dark pixels in the masked
            outer crown (relative to the central brightness threshold). Only informative.
    """

    img_fixed: Image.Image  # RGB with border masked (or same if none)
    ring_mask: np.ndarray  # 0=Masked periphery, 1=Useful area
    detected: bool
    ratio_dark: float  # Quick estimate of outer crown darkness


def detect_and_mask_vignette(
    img: Image.Image,
    inner_frac: float = 0.55,  # Lesion disk for central brightness estimate
    ring_width_px: int = 3,  # Ring thickness for radial profile
    drop_k: float = 0.78,  # Considered vignette when ring falls below X% of center brightness
    min_edge_frac: float = 0.55,  # Start looking for edge from X% of radius
    max_edge_frac: float = 0.98,  # Up to X% (almost outer edge)
    min_margin: float = 0.06,  # Minimum relative drop to accept detection
    use_hough_fallback: bool = True,
) -> VignetteResult:
    """
    Robust vignette detection via radial intensity profile with optional Hough fallback.
    The algorithm:
        1) Compute a radial brightness profile (mean intensity per thin ring) from the center.
        2) Search for a pronounced drop relative to the central brightness (Thresholded + Gradient check).
        3) If found, keep only the inner disk up to that radius and black out the exterior ring.
        4) If the profile is inconclusive and `use_hough_fallback=True`, try a circle detection
            (HoughCircles) to estimate the border, validating with a brightness drop check.
    Args:
        - img (PIL.Image.Image): Input RGB image.
        inner_frac (float): Radius fraction (0–1) for the central disk used to estimate
            the reference mean brightness (proxy for lesion/central region).
        - ring_width_px (int): Half-thickness (in pixels) of the rings used to compute
            the radial profile. Each radius averages a band [r - ring_width_px, r + ring_width_px].
        - drop_k (float): Relative threshold (0–1) w.r.t. the central mean. A ring is
            considered dark if mean < inner_mean * drop_k (e.g., 0.78).
        - min_edge_frac (float): Lower bound (fraction of max radius) where we start to
            look for the vignette edge (avoid the very central area).
        - max_edge_frac (float): Upper bound (fraction of max radius) up to which we
            compute the radial profile (close to the outer border).
        - min_margin (float): Minimum required relative drop
            (inner_mean - ring_mean) / inner_mean to accept a detection.
        - use_hough_fallback (bool): Whether to run a conservative HoughCircles fallback
            when the radial profile is inconclusive.

    Returns:
        - VignetteResult: Dataclass with the masked image (`img_fixed`), the binary keep-mask
            (`ring_mask`), a detection flag (`detected`), and a dark ratio estimate (`ratio_dark`).
    """

    # Convert to array and grayscale; slight blur to reduce noise in the profile.
    rgb = np.asarray(img)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Coordinate grids and per-pixel radius from center
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = r.max()

    # Central statistics (used as brightness reference)
    inner_mask = r <= inner_frac * rmax
    inner_vals = gray[inner_mask]
    inner_mean = float(inner_vals.mean()) if inner_vals.size else 120.0

    # Radial profile: mean intensity per thin ring between [min_edge_frac, max_edge_frac]
    r_min = int(min_edge_frac * rmax)
    r_max = int(max_edge_frac * rmax)
    radii = np.arange(r_min, r_max)  # sample radius pixel-by-pixel
    prof = []
    for rr in radii:
        ring = (r >= rr - ring_width_px) & (r <= rr + ring_width_px)
        vals = gray[ring]
        prof.append(float(vals.mean()) if vals.size else inner_mean)
    prof = np.array(prof, dtype=np.float32)

    # Look for the "edge": 1) relative threshold, 2) local gradient/derivative check
    thresh_rel = inner_mean * drop_k
    below = np.where(prof < thresh_rel)[0]
    r_edge_idx = None

    if below.size > 0:
        # First crossing below the relative threshold
        r_edge_idx = int(below[0])

        # Confirm with local gradient (seek strongest negative slope near that point)
        grad = np.gradient(prof)
        win = slice(max(0, r_edge_idx - 4), min(len(prof), r_edge_idx + 6))
        r_edge_idx = int(win.start + np.argmin(grad[win]))

        # Validate minimum relative drop
        rel_drop = (inner_mean - prof[r_edge_idx]) / max(inner_mean, 1e-6)
        if rel_drop < min_margin:
            r_edge_idx = None

    # Fallback with Hough if radial profile is inconclusive
    if r_edge_idx is None and use_hough_fallback:
        # Detect a large circle; use its radius as the vignette border
        # Conservative: better to keep a bit more outer area than cut into the lesion
        g_blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
        circles = cv2.HoughCircles(
            g_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min(h, w) // 3,
            param1=100,
            param2=30,
            minRadius=int(0.45 * rmax),
            maxRadius=int(0.95 * rmax),
        )
        if circles is not None and len(circles) > 0:
            c = circles[0][0]
            r_hough = float(c[2])

            # Validate using brightness: outside the Hough radius should be notably darker
            ring = (r >= r_hough) & (r <= rmax)
            out_vals = gray[ring]
            if out_vals.size > 0:
                rel_drop = (inner_mean - float(out_vals.mean())) / max(inner_mean, 1e-6)
                if rel_drop >= min_margin * 0.7:  # un pelín más permisivo en fallback
                    # Build keep-mask from r_edge (1=keep inside; 0=mask outside)
                    r_edge = r_hough
                else:
                    r_edge = None
            else:
                r_edge = None
            if r_edge is not None:
                # Build keep-mask from r_edge (1=keep inside; 0=mask outside)
                keep = (r <= r_edge).astype(np.uint8)
                fixed = (rgb * keep[..., None]).astype(np.uint8)
                return VignetteResult(
                    Image.fromarray(fixed),
                    keep,
                    True,
                    float(
                        (gray[(r > r_edge) & (r <= rmax)] < thresh_rel).mean()
                        if thresh_rel
                        else 0.0
                    ),
                )

    # If we found an edge by radial profile:
    if r_edge_idx is not None:
        r_edge = float(radii[r_edge_idx])
        keep = (r <= r_edge).astype(np.uint8)  # 1=disco útil, 0=fuera del borde
        fixed = (rgb * keep[..., None]).astype(np.uint8)

        # Estimate dark_ratio in the masked crown (informative only)
        crown = (r > r_edge) & (r <= rmax)
        dark_ratio = (
            float((gray[crown] < (inner_mean * drop_k)).mean()) if crown.any() else 0.0
        )
        return VignetteResult(Image.fromarray(fixed), keep, True, dark_ratio)

    # If nothing was conclusive then return original image and an all-ones mask
    keep = np.ones((h, w), np.uint8)
    return VignetteResult(img, keep, False, 0.0)
