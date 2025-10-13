# -*- coding: utf-8 -*-
"""
person_detector.py
------------------
Rilevamento persone con rilevamento automatico del backend:
- ONNX (YOLO v5/v8/v11 ecc.) via OpenCV DNN
- OpenVINO IR (.xml + .bin, es. person-detection-retail-0013)

Restituisce: [ ((x,y,w,h), score), ... ] in coordinate dell'immagine d'ingresso.

Config principali:
- model_path: percorso modello (.onnx oppure .xml)
- img_size:   lato di input per YOLO (letterbox). Ignorato in modalità OpenVINO
- score_th:   soglia conf
- iou_th:     IoU per NMS (solo YOLO raw)
- max_det:    max output boxes (post-NMS)
- backend_id/target_id: backend/target per OpenCV DNN (ONNX)
- ov_device:  dispositivo OpenVINO (es. "CPU", "GPU")
"""
from __future__ import annotations
import os
import cv2
import numpy as np
from typing import List, Tuple

try:
    from openvino.runtime import Core as _OVCore
except Exception:
    _OVCore = None


def _letterbox(img, new_size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    out = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = resized
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
                 target_id: int = 0,
                 ov_device: str = "CPU"):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        self.size = int(img_size)
        self.score_th = float(score_th)
        self.iou_th = float(iou_th)
        self.max_det = int(max_det)
        self._dbg_shape_printed = False

        ext = os.path.splitext(model_path)[1].lower()
        self.mode = "onnx" if ext == ".onnx" else ("openvino" if ext == ".xml" else "unknown")

        # ONNX (YOLO)
        if self.mode == "onnx":
            self.net = cv2.dnn.readNetFromONNX(model_path)
            try:
                self.net.setPreferableBackend(int(backend_id))
                self.net.setPreferableTarget(int(target_id))
            except Exception:
                pass
            self._ov_compiled = None
            self._ov_in_wh = None

        # OpenVINO IR (es. person-detection-retail-0013)
        elif self.mode == "openvino":
            if _OVCore is None:
                raise RuntimeError("OpenVINO non disponibile ma richiesto per .xml")
            core = _OVCore()
            model = core.read_model(model_path)
            compiled = core.compile_model(model, ov_device)
            self._ov_compiled = compiled
            # prova a rilevare la dimensione input (NCHW)
            try:
                ishape = list(compiled.inputs[0].shape)
                # shape tipica: [1, 3, H, W]
                if len(ishape) == 4:
                    H, W = int(ishape[2]), int(ishape[3])
                    self._ov_in_wh = (W, H)
                else:
                    self._ov_in_wh = (544, 320)  # fallback ragionevole
            except Exception:
                self._ov_in_wh = (544, 320)
            self.net = None
        else:
            raise ValueError(f"Estensione modello non supportata: {ext}")

    # ----------------- ONNX YOLO path -----------------
    def _detect_yolo(self, bgr) -> List[Tuple[Tuple[float, float, float, float], float]]:
        H0, W0 = bgr.shape[:2]
        inp, r, left, top = _letterbox(bgr, self.size)
        blob = cv2.dnn.blobFromImage(inp, scalefactor=1/255.0, size=(self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()

        if not self._dbg_shape_printed:
            try:
                print(f"[person_det/yolo] out.shape={tuple(out.shape)}")
            except Exception:
                pass
            self._dbg_shape_printed = True

        o = out
        end2end = False

        if o.ndim == 3 and o.shape[0] == 1:
            o = o[0]

        if o.ndim == 2 and o.shape[1] == 6:
            end2end = True
            preds = o  # [x1,y1,x2,y2,score,cls]
        elif o.ndim == 3 and o.shape[-1] == 6:
            end2end = True
            preds = o.reshape(-1, 6)
        else:
            if o.ndim == 2 and o.shape[0] in (84, 85):
                o = o.transpose(1, 0)
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
            preds = o

        dets_xyxy, scores = [], []
        if end2end:
            for x1, y1, x2, y2, conf, cls in preds:
                if int(cls) != 0:
                    continue
                conf = float(conf)
                if conf < self.score_th:
                    continue
                # alcuni export danno coord normalizzate 0..1
                if max(x1, y1, x2, y2) <= 1.5:
                    x1 *= self.size; y1 *= self.size; x2 *= self.size; y2 *= self.size
                x1 = (x1 - left) / r; y1 = (y1 - top) / r
                x2 = (x2 - left) / r; y2 = (y2 - top) / r
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
            boxes_xywh = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in dets_xyxy]
            return [((float(x), float(y), float(w), float(h)), float(s)) for (x, y, w, h), s in zip(boxes_xywh, scores)]
        else:
            C = preds.shape[1]
            for row in preds:
                x, y, w, h = row[0:4]
                if max(x, y, w, h) <= 1.5:
                    x *= self.size; y *= self.size; w *= self.size; h *= self.size
                if C >= 85:
                    obj = float(row[4]); cls_scores = row[5:]
                    ci = int(np.argmax(cls_scores)); conf = float(obj * cls_scores[ci])
                else:
                    cls_scores = row[4:]
                    ci = int(np.argmax(cls_scores)); conf = float(cls_scores[ci])
                if ci != 0 or conf < self.score_th:
                    continue
                x1 = x - w/2; y1 = y - h/2; x2 = x + w/2; y2 = y + h/2
                x1 = (x1 - left) / r; y1 = (y1 - top) / r
                x2 = (x2 - left) / r; y2 = (y2 - top) / r
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
            boxes_xywh = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in dets_xyxy]
            keep = _nms_xywh(boxes_xywh, scores, float(self.iou_th), int(self.max_det))
            return [((float(boxes_xywh[i][0]), float(boxes_xywh[i][1]), float(boxes_xywh[i][2]), float(boxes_xywh[i][3])), float(scores[i])) for i in keep]

    # ----------------- OpenVINO IR path -----------------
    def _detect_openvino(self, bgr) -> List[Tuple[Tuple[float, float, float, float], float]]:
        H0, W0 = bgr.shape[:2]
        W_in, H_in = self._ov_in_wh
        img = cv2.resize(bgr, (W_in, H_in), interpolation=cv2.INTER_LINEAR)
        x = np.transpose(img.astype(np.float32), (2, 0, 1))[None, ...]  # BGR, 0..255
        res_map = self._ov_compiled([x])
        outputs = [np.array(res_map[o]) for o in self._ov_compiled.outputs]
        out = outputs[0]
        if not self._dbg_shape_printed:
            try:
                print(f"[person_det/ov] out.shape={tuple(out.shape)}")
            except Exception:
                pass
            self._dbg_shape_printed = True

        # Formati tipici: [1,1,N,7] o [N,7]
        dets = out
        if dets.ndim == 4:
            dets = dets.reshape(-1, dets.shape[-1])
        elif dets.ndim == 3:
            dets = dets.reshape(-1, dets.shape[-1])
        elif dets.ndim == 2:
            pass
        else:
            # formato non previsto
            return []

        out_list = []
        for row in dets:
            if row.size < 6:
                continue
            # OpenVINO DetectionOutput: [image_id, label, conf, x1, y1, x2, y2]
            if row.size >= 7:
                conf = float(row[2])
                x1n, y1n, x2n, y2n = float(row[3]), float(row[4]), float(row[5]), float(row[6])
            else:
                # fallback: [x1,y1,x2,y2,score]
                conf = float(row[4]); x1n, y1n, x2n, y2n = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            if conf < self.score_th:
                continue
            # coordinate generalmente normalizzate [0,1]
            if max(x1n, y1n, x2n, y2n) <= 1.5:
                x1 = x1n * W0; y1 = y1n * H0; x2 = x2n * W0; y2 = y2n * H0
            else:
                # se fossero in pixel dell'input, scala a immagine originale
                x1 = x1n * (W0 / float(W_in)); y1 = y1n * (H0 / float(H_in))
                x2 = x2n * (W0 / float(W_in)); y2 = y2n * (H0 / float(H_in))
            x1 = float(max(0.0, min(W0 - 1.0, x1)))
            y1 = float(max(0.0, min(H0 - 1.0, y1)))
            x2 = float(max(0.0, min(W0 - 1.0, x2)))
            y2 = float(max(0.0, min(H0 - 1.0, y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            out_list.append(((x1, y1, x2 - x1, y2 - y1), conf))
        return out_list

    def detect(self, bgr) -> List[Tuple[Tuple[float, float, float, float], float]]:
        if bgr is None or bgr.size == 0:
            return []
        if self.mode == "onnx":
            return self._detect_yolo(bgr)
        elif self.mode == "openvino":
            return self._detect_openvino(bgr)
        return []

