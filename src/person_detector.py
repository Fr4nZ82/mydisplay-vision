# -*- coding: utf-8 -*-
"""
person_detector.py
------------------
Rilevamento persone (classe COCO 0) da modelli YOLO (v5/v8/v11) esportati in ONNX.
Restituisce: [ ((x,y,w,h), score), ... ] in coordinate dell'immagine d'ingresso.

Config:
- model_path: percorso ONNX
- img_size:   lato di input (es. 640)
- score_th:   soglia conf
- iou_th:     IoU per NMS
- max_det:    max output boxes
- backend_id/target_id: backend/target per OpenCV DNN
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple

def _letterbox(img, new_size=640, color=(114,114,114)):
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    out = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    out[top:top+nh, left:left+nw] = resized
    return out, r, left, top

def _nms_xywh(boxes: List[List[float]], scores: List[float], iou_th: float, max_det: int):
    if not boxes:
        return []
    b = np.array(boxes, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)
    idxs = cv2.dnn.NMSBoxes(b.tolist(), s.tolist(), score_threshold=0.0, nms_threshold=float(iou_th))
    if isinstance(idxs, tuple) or isinstance(idxs, list):
        idxs = np.array(idxs).reshape(-1)
    elif hasattr(idxs, "flatten"):
        idxs = idxs.flatten()
    idxs = idxs[:max_det]
    return idxs.tolist()

class PersonDetector:
    def __init__(self,
                 model_path: str,
                 img_size: int = 640,
                 score_th: float = 0.35,
                 iou_th: float = 0.45,
                 max_det: int = 200,
                 backend_id: int = 0,
                 target_id: int = 0):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        try:
            self.net.setPreferableBackend(int(backend_id))
            self.net.setPreferableTarget(int(target_id))
        except Exception:
            pass
        self.size = int(img_size)
        self.score_th = float(score_th)
        self.iou_th = float(iou_th)
        self.max_det = int(max_det)
        self._dbg_shape_printed = False

    def detect(self, bgr) -> List[Tuple[Tuple[float,float,float,float], float]]:
      if bgr is None or bgr.size == 0:
          return []
      H0, W0 = bgr.shape[:2]

      # letterbox -> blob
      inp, r, left, top = _letterbox(bgr, self.size)
      blob = cv2.dnn.blobFromImage(inp, scalefactor=1/255.0, size=(self.size, self.size), swapRB=True, crop=False)
      self.net.setInput(blob)
      out = self.net.forward()

      # stampa la shape una sola volta (utile per capire che export hai)
      if not self._dbg_shape_printed:
          try:
              print(f"[person_det] out.shape={tuple(out.shape)}")
          except Exception:
              pass
          self._dbg_shape_printed = True

      # ------ Normalizza a vari casi noti ------
      o = out
      end2end = False  # true se l'ONNX ha già NMS e produce (N,6)

      # squeeze batch se presente
      if o.ndim == 3 and o.shape[0] == 1:
          o = o[0]  # -> (*,*)

      # Caso END2END NMS: (N,6) o (1,N,6) già reso (N,6) sopra
      if o.ndim == 2 and o.shape[1] == 6:
          end2end = True
          preds = o  # [x1,y1,x2,y2,score,cls]
      elif o.ndim == 3 and o.shape[-1] == 6:
          end2end = True
          preds = o.reshape(-1, 6)
      else:
          # C grezzo: 84 o 85
          if o.ndim == 2 and o.shape[0] in (84, 85):
              o = o.transpose(1, 0)  # (84,N) -> (N,84)
          elif o.ndim == 3 and o.shape[1] in (84, 85):
              o = o[0].transpose(1, 0)
          elif o.ndim == 3 and o.shape[2] in (84, 85):
              o = o[0]
          elif o.ndim != 2:
              flat = o.reshape(-1)
              C = 85 if (flat.size % 85 == 0) else (84 if flat.size % 84 == 0 else None)
              if C is None:
                  return []
              o = flat.reshape(-1, C)

          if o.ndim != 2 or o.shape[1] not in (84, 85):
              return []

          C = o.shape[1]
          preds = o  # (N,84/85) -> [x,y,w,h,(obj?), 80 class...]

      dets_xyxy = []
      scores = []

      if end2end:
          # --- Già con NMS: preds=(N,6) [x1,y1,x2,y2,score,cls] ---
          for x1, y1, x2, y2, conf, cls in preds:
              cls = int(cls)
              if cls != 0:       # COCO 'person'
                  continue
              conf = float(conf)
              if conf < self.score_th:
                  continue

              # alcuni export danno coord normalizzate 0..1
              mx = max(x1, y1, x2, y2)
              if mx <= 1.5:
                  x1 *= self.size; y1 *= self.size; x2 *= self.size; y2 *= self.size

              # back to original (rimuovi padding, poi /r)
              x1 = (x1 - left) / r; y1 = (y1 - top) / r
              x2 = (x2 - left) / r; y2 = (y2 - top) / r

              # clamp
              x1 = float(max(0.0, min(W0 - 1.0, x1)))
              y1 = float(max(0.0, min(H0 - 1.0, y1)))
              x2 = float(max(0.0, min(W0 - 1.0, x2)))
              y2 = float(max(0.0, min(H0 - 1.0, y2)))
              if x2 <= x1 or y2 <= y1:
                  continue

              dets_xyxy.append([x1, y1, x2, y2])
              scores.append(conf)

          # niente NMS: è già stato applicato nell'ONNX
          if not dets_xyxy:
              return []
          boxes_xywh = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in dets_xyxy]
          out_det = [((float(x), float(y), float(w), float(h)), float(s))
                    for (x, y, w, h), s in zip(boxes_xywh, scores)]
          return out_det

      else:
          # --- Raw head: preds=(N,84/85) ---
          C = preds.shape[1]
          for row in preds:
              x, y, w, h = row[0:4]

              # coord normalizzate? (tipicamente 0..1)
              if max(x, y, w, h) <= 1.5:
                  x *= self.size; y *= self.size; w *= self.size; h *= self.size

              if C >= 85:
                  obj = float(row[4])
                  cls_scores = row[5:]
                  ci = int(np.argmax(cls_scores))
                  conf = float(obj * cls_scores[ci])
              else:
                  cls_scores = row[4:]
                  ci = int(np.argmax(cls_scores))
                  conf = float(cls_scores[ci])

              if ci != 0:
                  continue
              if conf < self.score_th:
                  continue

              # xywh(center) -> xyxy (sul letterbox)
              x1 = x - w/2; y1 = y - h/2
              x2 = x + w/2; y2 = y + h/2

              # back to original
              x1 = (x1 - left) / r; y1 = (y1 - top) / r
              x2 = (x2 - left) / r; y2 = (y2 - top) / r

              # clamp
              x1 = float(max(0.0, min(W0 - 1.0, x1)))
              y1 = float(max(0.0, min(H0 - 1.0, y1)))
              x2 = float(max(0.0, min(W0 - 1.0, x2)))
              y2 = float(max(0.0, min(H0 - 1.0, y2)))
              if x2 <= x1 or y2 <= y1:
                  continue

              dets_xyxy.append([x1, y1, x2, y2])
              scores.append(conf)

          if not dets_xyxy:
              return []

          # NMS su xywh
          boxes_xywh = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in dets_xyxy]
          keep = _nms_xywh(boxes_xywh, scores, float(self.iou_th), int(self.max_det))
          out_det = []
          for i in keep:
              x, y, w, h = boxes_xywh[i]
              out_det.append(((float(x), float(y), float(w), float(h)), float(scores[i])))
          return out_det
