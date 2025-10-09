# -*- coding: utf-8 -*-
"""
BodyReID: estrazione embedding corpo da crop persona (BGR).
Supporta modelli tipici:
- OSNet (es. osnet_x0_25_msmt17.onnx): input 256x128 RGB, mean/std ImageNet
- Intel OMZ person-reidentification-retail-0288.onnx: input 256x128 BGR (float), scala 0..1 tipica

Heuristica:
- Se il path contiene 'osnet' => usa RGB + normalizzazione ImageNet
- Altrimenti usa BGR + scala 0..1

Output: vettore 1D float32 L2-normalizzato, oppure None se fallisce.
"""
from __future__ import annotations
from typing import Optional, Tuple
import os
import numpy as np
import cv2

class BodyReID:
    def __init__(self,
                 model_path: str,
                 backend_id: int = 0,
                 target_id: int = 0,
                 input_size: Tuple[int, int] = (128, 256)) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"BodyReID model not found: {model_path}")
        self.net = cv2.dnn.readNetFromONNX(model_path)
        try:
            self.net.setPreferableBackend(int(backend_id))
            self.net.setPreferableTarget(int(target_id))
        except Exception:
            pass
        # input_size: (W, H)
        self.in_w, self.in_h = int(input_size[0]), int(input_size[1])
        name = os.path.basename(model_path).lower()
        self.mode = "osnet" if "osnet" in name else "intel"

    @staticmethod
    def _l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        n = float(np.linalg.norm(x) + eps)
        return (x / n).astype(np.float32)

    def embed(self, person_bgr: np.ndarray) -> Optional[np.ndarray]:
        if person_bgr is None or person_bgr.size == 0:
            return None
        h, w = person_bgr.shape[:2]
        if h < 20 or w < 20:
            return None
        try:
            if self.mode == "osnet":
                # BGR->RGB, resize 256x128 (H,W), to 0..1, mean/std ImageNet, NCHW
                img = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
                x = img.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                x = (x - mean) / std
                blob = np.transpose(x, (2, 0, 1))[None, ...]
            else:
                # Intel OMZ: BGR, resize, scala 0..1, NCHW
                img = cv2.resize(person_bgr, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
                x = (img.astype(np.float32) / 255.0)
                blob = np.transpose(x, (2, 0, 1))[None, ...]
            self.net.setInput(blob)
            out = self.net.forward()
            vec = np.asarray(out).reshape(-1).astype(np.float32)
            if vec.size == 0:
                return None
            return self._l2_normalize(vec)
        except Exception:
            return None