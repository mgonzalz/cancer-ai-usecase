# Hair Removal & Lesion-Preserved Composer for Melanoma Preprocessing
# Purpose: This module performs DullRazor-style hair removal (Black-Hat + inpainting)
# and a lesion-preserved composition that keeps the lesion pixels EXACTLY as
# in the original image while cleaning the background. Includes a mini-test
# that generates visual panels for quick inspection.
#
# Notes:
#   - Run vignette correction BEFORE hair removal to avoid confusing the dark
#     dermatoscope border with hair.
#   - The lesion_mask must be 0/255 (converted internally to 0/1).
#   - Parameter effects:
#       * Kernel Size (High): Detects thicker hairs but may affect lesion edges.
#       * Bh Threshold (High): Stricter, fewer false detections, may miss fine hairs.
#       * Inpaint Radius (High): Smoother fill, may create halos.
#       * Inpaint Method: "telea" = natural-looking; "ns" = better continuity.

from typing import Literal, Tuple

import cv2
import numpy as np


def remove_hair(
    rgb: np.ndarray,
    kernel_size: int = 9,
    bh_threshold: int = 10,
    inpaint_radius: int = 5,
    inpaint_method: Literal["telea", "ns"] = "telea",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Remove hairs using Black-Hat + inpainting (DullRazor-like).
    Args:
        rgb: uint8 RGB image (H,W,3).
        kernel_size: structuring element size for black-hat.
        bh_threshold: binary threshold on black-hat response.
        inpaint_radius: radius passed to cv2.inpaint.
        inpaint_method: 'telea' or 'ns'.

    Returns:
        clean_rgb: hair-inpainted RGB (uint8).
        hair_mask: binary mask (uint8, 0/255) of thin hair filaments.
        hair_coverage: fraction in [0,1] of pixels flagged as hair.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, hair_mask = cv2.threshold(blackhat, bh_threshold, 255, cv2.THRESH_BINARY)

    method = cv2.INPAINT_TELEA if inpaint_method == "telea" else cv2.INPAINT_NS
    clean_rgb = cv2.inpaint(rgb, hair_mask, inpaint_radius, method)

    hair_coverage = float((hair_mask > 0).mean())
    return clean_rgb, hair_mask, hair_coverage


def compose_lesion_preserved(
    orig_rgb: np.ndarray, hair_clean_rgb: np.ndarray, lesion_mask: np.ndarray
) -> np.ndarray:
    """
    Compose a final image that preserves the lesion from the ORIGINAL image
    and uses the hair-removed image for the background.
    - orig_rgb: uint8 (H,W,3), original (ya con vignette aplicado si procede)
    - hair_clean_rgb: uint8 (H,W,3), salida de remove_hair
    - lesion_mask: uint8 (H,W), 0/255
    Returns: uint8 (H,W,3)
    """
    assert orig_rgb.dtype == np.uint8 and hair_clean_rgb.dtype == np.uint8
    assert orig_rgb.shape == hair_clean_rgb.shape and orig_rgb.shape[2] == 3
    m = (lesion_mask > 0).astype(np.uint8)[..., None]  # (H,W,1) 0/1
    out = (m * orig_rgb + (1 - m) * hair_clean_rgb).astype(np.uint8)
    return out
