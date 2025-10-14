# -*- coding: utf-8 -*-
"""
Utility di visualizzazione/geom per MyDisplay Vision:
- resize keep-aspect per i frame di debug
- disegno riquadri + etichette
- helper geometrici (IoU, point-in-box)
- associazione face->person per il runtime
- disegno tripwire normalizzata
"""
from __future__ import annotations
from typing import Tuple, List, Dict, Optional
import cv2
import numpy as np

# ------------ Resize / Overlay ------------

# Ridimensiona l'immagine preservando il rapporto d'aspetto; utile per anteprime/debug. target_w è la larghezza in pixel (<=0: nessuna modifica)
def resize_keep_aspect(img: np.ndarray, target_w: int) -> np.ndarray:
    """Ridimensiona mantenendo l'aspect ratio. Se target_w <= 0, restituisce l'immagine originale."""
    if img is None or img.size == 0 or target_w is None or target_w <= 0:
        return img
    h, w = img.shape[:2]
    if w == 0 or w == target_w:
        return img
    scale = target_w / float(w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# Disegna un riquadro (x,y,w,h) e un'etichetta testo sopra; modifica 'img' in-place; colore in BGR, spessore configurabile
def draw_box_with_label(
    img: np.ndarray,
    bbox: Tuple[float, float, float, float],
    label: str,
    color: Tuple[int, int, int] = (0, 180, 0),
    thickness: int = 1,
) -> None:

    """Disegna un riquadro (x,y,w,h) e un box testuale sopra."""
    x, y, w, h = map(int, bbox)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y1 = max(0, y - th - 6)
    cv2.rectangle(img, (x, y1), (x + tw + 6, y), (0, 0, 0), -1)
    cv2.putText(img, label, (x + 3, max(12, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# Disegna una linea 'tripwire' con punti normalizzati (0..1) rispetto a img; band_px controlla lo spessore, color è BGR
def draw_tripwire(img: np.ndarray,
                  roi_tripwire: Tuple[Tuple[float, float], Tuple[float, float]],
                  band_px: int = 8,
                  color: Tuple[int, int, int] = (255, 0, 0)) -> None:
    """Disegna la tripwire con coordinate normalizzate."""
    h, w = img.shape[:2]
    (axn, ayn), (bxn, byn) = roi_tripwire
    ax, ay = int(axn * w), int(ayn * h)
    bx, by = int(bxn * w), int(byn * h)
    cv2.line(img, (ax, ay), (bx, by), color, max(1, int(band_px)))


# ------------ Geom helpers ------------

# Calcola l'IoU tra due bbox in formato (x,y,w,h); ritorna un float in [0,1]
def iou_xywh(a: Tuple[float, float, float, float],
             b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1e-6)


# Controlla se il centro della bbox interna ricade all'interno della bbox esterna
def center_in(inner_xywh: Tuple[float, float, float, float],
              outer_xywh: Tuple[float, float, float, float]) -> bool:
    ix, iy, iw, ih = inner_xywh
    ox, oy, ow, oh = outer_xywh
    cx, cy = ix + iw * 0.5, iy + ih * 0.5
    return (ox <= cx <= ox + ow) and (oy <= cy <= oy + oh)


# Associa facce rilevate alle tracce persona usando IoU e/o 'center_in'; cada traccia ottiene al più una faccia
def associate_faces_to_tracks(
    face_dets: List[Tuple[Tuple[float, float, float, float], float]],
    person_tracks: List[Dict],
    iou_th: float = 0.2,
    use_center_in: bool = True
) -> Dict[int, Optional[Tuple[float, float, float, float]]]:
    """
    face_dets: [((x,y,w,h), score), ...]
    person_tracks: [{'track_id', 'bbox':(x,y,w,h), ...}, ...]
    Ritorna: { track_id: face_bbox or None }
    """
    assoc: Dict[int, Optional[Tuple[float, float, float, float]]] = {}
    used = set()
    for t in person_tracks:
        tb = tuple(map(float, t["bbox"]))
        best_iou, best_idx = 0.0, -1
        for i, (fb, fs) in enumerate(face_dets):
            if i in used:
                continue
            ok = (iou_xywh(tb, fb) >= iou_th) or (use_center_in and center_in(fb, tb))
            if ok:
                cand = iou_xywh(tb, fb)
                if cand > best_iou:
                    best_iou, best_idx = cand, i
        assoc[int(t["track_id"])] = face_dets[best_idx][0] if best_idx >= 0 else None
        if best_idx >= 0:
            used.add(best_idx)
    return assoc


def associate_faces_to_tracks_with_kps(
    face_dets_kps: List[Tuple[Tuple[Tuple[float, float, float, float], float], np.ndarray]],
    person_tracks: List[Dict],
    iou_th: float = 0.2,
    use_center_in: bool = True
) -> Dict[int, Optional[Tuple[Tuple[float, float, float, float], np.ndarray]]]:
    """
    Variante che preserva anche i 5 landmark: restituisce { track_id: (face_bbox, kps5) or None }
    """
    assoc: Dict[int, Optional[Tuple[Tuple[float, float, float, float], np.ndarray]]] = {}
    used = set()
    for t in person_tracks:
        tb = tuple(map(float, t["bbox"]))
        best_iou, best_idx = 0.0, -1
        for i, (det, kps) in enumerate(face_dets_kps):
            fb, fs = det
            if i in used:
                continue
            ok = (iou_xywh(tb, fb) >= iou_th) or (use_center_in and center_in(fb, tb))
            if ok:
                cand = iou_xywh(tb, fb)
                if cand > best_iou:
                    best_iou, best_idx = cand, i
        if best_idx >= 0:
            assoc[int(t["track_id"])] = (face_dets_kps[best_idx][0][0], face_dets_kps[best_idx][1])
            used.add(best_idx)
        else:
            assoc[int(t["track_id"])] = None
    return assoc

