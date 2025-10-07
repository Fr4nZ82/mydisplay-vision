# -*- coding: utf-8 -*-
"""
Utility di visualizzazione per MyDisplay Vision:
- resize keep-aspect per i frame di debug
- disegno riquadri + etichette sopra i volti
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple


def resize_keep_aspect(img: np.ndarray, target_w: int) -> np.ndarray:
    """
    Ridimensiona mantenendo l'aspect ratio. Se target_w <= 0, restituisce l'immagine originale.
    """
    if img is None or img.size == 0 or target_w is None or target_w <= 0:
        return img
    h, w = img.shape[:2]
    if w == 0 or w == target_w:
        return img
    scale = target_w / float(w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_box_with_label(
    img: np.ndarray,
    bbox: Tuple[float, float, float, float],
    label: str,
    color: Tuple[int, int, int] = (0, 180, 0),
) -> None:
    """
    Disegna un riquadro (x,y,w,h) e un box testuale sopra il volto.
    """
    x, y, w, h = map(int, bbox)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # fondo scuro per la label
    y1 = max(0, y - th - 6)
    cv2.rectangle(img, (x, y1), (x + tw + 6, y), (0, 0, 0), -1)
    cv2.putText(
        img,
        label,
        (x + 3, max(12, y - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
