# Color tensor builders for CNN inputs.
# Purpose: Provide small, dependable utilities to:
# 1) Normalize RGB images to [0,1] float32.
# 2) Append the HSV H or S channels (normalized) to RGB, yielding 4-channel
#    tensors used in experiments like RGB+H or RGB+S.
#
# Notes:
# - OpenCV expects BGR by default, but this module **intentionally** works in RGB. Conversions use cv2.COLOR_RGB2HSV (not BGR).
# - Input images must be uint8 RGB in shape [H, W, 3]. Outputs are float32.
# - H is normalized by 179.0 (OpenCV hue range), S by 255.0.
# - Use `build_tensor(..., profile=...)` to request: "rgb", "rgb_h", or "rgb_s".

from __future__ import annotations
import numpy as np
import cv2
from PIL import Image, ImageOps

def to_numpy01(img: Image.Image) -> np.ndarray:
    """PIL RGB -> float32 [0,1]"""
    return (np.asarray(img).astype(np.float32) / 255.0)

def add_h_channel(rgb01: np.ndarray) -> np.ndarray:
    """
    Adds H (HSV) normalized channel:
      - input: float32 [0,1], 3 channels
      - output: float32 [0,1], 4 channels (RGB + H_norm)
    """
    assert rgb01.ndim == 3 and rgb01.shape[2] == 3
    hsv = cv2.cvtColor((rgb01*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    H = (hsv[...,0].astype(np.float32) / 179.0)[..., None]  # 0..1
    return np.concatenate([rgb01, H], axis=-1)

def add_s_channel(rgb01: np.ndarray) -> np.ndarray:
    """
    Adds S (HSV) normalized channel:
      - input: float32 [0,1], 3 channels
      - output: float32 [0,1], 4 channels (RGB + S_norm)
    """
    assert rgb01.ndim == 3 and rgb01.shape[2] == 3
    hsv = cv2.cvtColor((rgb01*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    S = (hsv[...,1].astype(np.float32) / 255.0)[..., None]  # 0..1
    return np.concatenate([rgb01, S], axis=-1)

def build_tensor(img_rgb_uint8: np.ndarray, profile: str = "rgb") -> np.ndarray:
    """
    Builds the input tensor according to 'profile':
      - 'rgb'    -> [H,W,3] float32 [0,1]
      - 'rgb_h'  -> [H,W,4] (RGB + H_norm)
      - 'rgb_s'  -> [H,W,4] (RGB + S_norm)
    """
    assert img_rgb_uint8.ndim == 3 and img_rgb_uint8.shape[2] == 3 and img_rgb_uint8.dtype == np.uint8
    x = img_rgb_uint8.astype(np.float32)/255.0
    prof = profile.lower()
    if prof == "rgb":
        return x
    elif prof == "rgb_h":
        return add_h_channel(x)
    elif prof == "rgb_s":
        return add_s_channel(x)
    else:
        raise ValueError(f"Unknown profile '{profile}'")
