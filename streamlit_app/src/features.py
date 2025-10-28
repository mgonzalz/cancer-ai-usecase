# src/features.py
from __future__ import annotations
import numpy as np
import cv2

# ---------- helpers básicos ----------
def _safe_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.copy()
    if m.ndim == 3:
        m = m[...,0]
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8) * 255
    return m

def _contour_area_perimeter(mask_u8: np.ndarray) -> tuple[int, float]:
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0, 0.0
    c = max(cnts, key=cv2.contourArea)
    area = int(cv2.contourArea(c))
    peri = float(cv2.arcLength(c, True))
    return area, peri

def _diameter_pixels(mask_u8: np.ndarray) -> float:
    # Diámetro = mayor distancia entre puntos de contorno (distancia euclídea)
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    pts = c.reshape(-1, 2)
    if pts.shape[0] < 2:
        return 0.0
    # truco: convex hull para reducir pares
    hull = cv2.convexHull(pts)
    hull_pts = hull.reshape(-1, 2)
    dmax = 0.0
    for i in range(len(hull_pts)):
        for j in range(i+1, len(hull_pts)):
            d = np.linalg.norm(hull_pts[i] - hull_pts[j])
            if d > dmax:
                dmax = d
    return float(dmax)

# ---------- ABCD ----------
def asymmetry_index(mask_u8: np.ndarray) -> float:
    """
    Índice de asimetría (0..1). 0 = totalmente simétrica, 1 = muy asimétrica.
    Promedia simetría horizontal y vertical: 1 - (intersección / unión) tras flip.
    """
    m = (mask_u8 > 0).astype(np.uint8)
    mh = np.flip(m, axis=1)  # flip horizontal
    mv = np.flip(m, axis=0)  # flip vertical
    def jaccard(a,b):
        inter = (a & b).sum()
        union = (a | b).sum()
        return inter/union if union>0 else 0.0
    a_h = 1.0 - jaccard(m, mh)
    a_v = 1.0 - jaccard(m, mv)
    return float((a_h + a_v)/2.0)

def border_irregularity(mask_u8: np.ndarray) -> float:
    """
    Irregularidad de borde vía circularidad: 4πA / P^2.
    1 = círculo perfecto, <1 más irregular. Devolvemos (1 - circularidad) para que 0=regular, 1=muy irregular.
    """
    area, peri = _contour_area_perimeter(mask_u8)
    if area <= 0 or peri <= 0:
        return 0.0
    circ = (4.0 * np.pi * area) / (peri ** 2 + 1e-8)
    circ = max(0.0, min(1.0, float(circ)))
    return float(1.0 - circ)

def color_metrics(img_rgb_u8: np.ndarray, mask_u8: np.ndarray, k: int = 4) -> dict:
    """
    Métricas simples de color dentro de la lesión:
    - Promedio y desviación en HSV
    - Número de clusters significativos (kmeans) con >5% de los píxeles de lesión
    """
    m = (mask_u8 > 0)
    if m.sum() == 0:
        return {"h_mean":0,"s_mean":0,"v_mean":0,"h_std":0,"s_std":0,"v_std":0,"clusters_sig":0}
    hsv = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2HSV)
    roi = hsv[m]
    h, s, v = roi[:,0].astype(np.float32), roi[:,1].astype(np.float32), roi[:,2].astype(np.float32)

    h_mean, s_mean, v_mean = float(h.mean()), float(s.mean()), float(v.mean())
    h_std,  s_std,  v_std  = float(h.std()),  float(s.std()),  float(v.std())

    # KMeans en HSV (H,S,V normalizados)
    X = np.stack([h/179.0, s/255.0, v/255.0], axis=1).astype(np.float32)
    # criterios y kmeans de OpenCV
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    # si menos puntos que k, reducimos k
    kk = min(k, X.shape[0])
    if kk < 2:
        clusters_sig = 1
    else:
        compact, labels, centers = cv2.kmeans(X, kk, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.flatten(), minlength=kk)
        thr = 0.05 * counts.sum()
        clusters_sig = int((counts >= thr).sum())

    return {
        "h_mean": round(h_mean,2), "s_mean": round(s_mean,2), "v_mean": round(v_mean,2),
        "h_std": round(h_std,2),   "s_std": round(s_std,2),   "v_std": round(v_std,2),
        "clusters_sig": clusters_sig
    }

def diameter_pixels(mask_u8: np.ndarray) -> float:
    return round(_diameter_pixels(mask_u8), 2)

def hair_coverage(mask_hair_u8: np.ndarray, mask_lesion_u8: np.ndarray) -> float:
    """Porcentaje de píxeles de la lesión que están afectados por la máscara de pelo."""
    L = (mask_lesion_u8 > 0)
    H = (mask_hair_u8 > 0)
    total = L.sum()
    if total == 0:
        return 0.0
    covered = (L & H).sum()
    return float(100.0 * covered / total)
