# Geometric & Light Augmentation for Melanoma Image Preprocessing
# Purpose: This module performs light geometric augmentations on dermatoscopic images
# without altering their color balance. It is used to increase dataset
# diversity while preserving the clinical appearance of lesions.
#
# Notes:
#   - Operates on normalized float32 arrays in [0,1].
#   - Applies random horizontal/vertical flips, small rotations, and zoom
#     (either cropping or reflective padding).
#   - Intended for “mild” augmentations suitable for training CNN models where
#     excessive distortion could alter lesion morphology.
#   - The augmentation preserves lesion colors and global illumination.
#   - Compatible with OpenCV float32 pipelines (cv2.warpAffine, cv2.resize).
#   - Typically used inside the data loader or preprocessing pipeline.

from __future__ import annotations

import random

import cv2
import numpy as np


def geo_light(
    x: np.ndarray,
    max_rot_deg: float = 10.0,
    min_zoom: float = 1.00,
    max_zoom: float = 1.20,
    flip_h_prob: float = 0.2,
    flip_v_prob: float = 0.2,
) -> np.ndarray:
    """
    Geometric light augmentation (color-safe).
    Args:
        - x: float32 [H,W,C] in [0,1]
        - max_rot_deg: float, maximum rotation angle in degrees
        - min_zoom: float, minimum zoom factor
        - max_zoom: float, maximum zoom factor
        - flip_h_prob: float, probability of horizontal flip
        - flip_v_prob: float, probability of vertical flip
    Returns:
        - float32 [H,W,C] in [0,1]
    """
    assert x.ndim == 3 and x.dtype == np.float32
    H, W, C = x.shape
    out = x.copy()

    # Flips
    if random.random() < flip_h_prob:
        out = out[:, ::-1, :]
    if random.random() < flip_v_prob:
        out = out[::-1, :, :]

    # Rotation
    ang = random.uniform(-max_rot_deg, max_rot_deg)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    out = cv2.warpAffine(
        out, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )

    # Zoom
    scale = random.uniform(min_zoom, max_zoom)
    if abs(scale - 1.0) > 1e-6:
        nh, nw = int(H * scale), int(W * scale)
        resized = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if scale > 1.0:  # crop centrado
            y0 = (nh - H) // 2
            x0 = (nw - W) // 2
            out = resized[y0:y0 + H, x0:x0 + W, :]
        else:  # pad reflect
            pad_t = (H - nh) // 2
            pad_l = (W - nw) // 2
            out = cv2.copyMakeBorder(
                resized,
                pad_t,
                H - nh - pad_t,
                pad_l,
                W - nw - pad_l,
                borderType=cv2.BORDER_REFLECT_101,
            )
    return np.clip(out, 0.0, 1.0).astype(np.float32)
