# Lesion Segmentation & Mask Cleaning for Melanoma Preprocessing
# Purpose: This module isolates the lesion region from a dermatoscopic RGB image using
# color-space segmentation (HSV-S channel + Otsu thresholding) and then
# refines the mask by removing thin hair filaments. It is part of the
# preprocessing pipeline for the melanoma classification project.
#
# Notes:
#   - Input images should be hair-cleaned.
#   - The segmentation relies on the saturation (S) channel, which provides
#     good contrast between skin and lesion regardless of lighting conditions.
#   - The largest and most centered connected component is assumed to be the
#     lesion; internal holes are filled automatically.
#   - clean_mask_with_hair() subtracts any hair remnants and applies
#     morphological closing/opening to smooth borders.
#   - Output masks are binary (0 background, 255 lesion) and can be used for:
#       * Lesion-preserved image composition (see hair.py)
#       * Feature extraction or shape analysis
#       * Visual diagnostics and EDA

import cv2
import numpy as np
from typing import Tuple

def segment_lesion(rgb: np.ndarray, size=(224, 224)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment lesion from a hair-cleaned RGB image using S-channel + Otsu.
    Args:
        - rgb: uint8 RGB image (H,W,3).
        - size: desired output size (width, height).
    Returns:
        - mask (uint8 0/255), lesion_rgb (uint8 RGB with mask applied)
    """
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1]
    S_blur = cv2.GaussianBlur(S, (5, 5), 0)

    _, mask = cv2.threshold(S_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Keep the largest-ish, near-center component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    H, W = mask.shape
    center = np.array([W / 2.0, H / 2.0])
    if num_labels > 1:
        valid = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]
            dist = np.linalg.norm(np.array([cx, cy]) - center)
            valid.append((i, area, dist))
        # Prefer big and centered
        label = max(valid, key=lambda t: (t[1], -t[2]))[0]
        mask = np.where(labels == label, 255, 0).astype(np.uint8)

    # Fill internal holes
    inv = cv2.bitwise_not(mask)
    n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(inv)
    for i in range(1, n2):
        x, y, ww, hh, _ = stats2[i]
        if x > 0 and y > 0 and (x + ww) < W and (y + hh) < H:
            inv[lab2 == i] = 0
    mask = cv2.bitwise_not(inv)

    lesion_rgb = cv2.bitwise_and(rgb, rgb, mask=mask)
    return mask, lesion_rgb
