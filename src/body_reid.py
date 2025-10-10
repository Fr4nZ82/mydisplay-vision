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

try:
    from openvino.runtime import Core as _OVCore
except Exception:
    _OVCore = None

class BodyReID:
    def __init__(self,
                 model_path: str,
                 backend_id: int = 0,
                 target_id: int = 0,
                 input_size: Tuple[int, int] = (128, 256)) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"BodyReID model not found: {model_path}")

        self.in_w, self.in_h = int(input_size[0]), int(input_size[1])
        name = os.path.basename(model_path).lower()
        self.mode = "osnet" if "osnet" in name else "intel"
        ext = os.path.splitext(model_path)[1].lower()
        self._backend = "onnx"
        self._ov_compiled = None

        if ext == ".xml" and _OVCore is not None:
            try:
                core = _OVCore()
                model = core.read_model(model_path)
                self._ov_compiled = core.compile_model(model, "CPU")
                self._backend = "openvino"
            except Exception:
                self._ov_compiled = None
                self._backend = "onnx"

        if self._backend == "onnx":
            self.net = cv2.dnn.readNetFromONNX(model_path)
            try:
                self.net.setPreferableBackend(int(backend_id))
                self.net.setPreferableTarget(int(target_id))
            except Exception:
                pass


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
            if self._backend == "onnx":
                # (preprocess identico a prima; osnet in RGB normalizzato, intel in BGR/255)
                if self.mode == "osnet":
                    img = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
                    x = img.astype(np.float32) / 255.0
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    x = (x - mean) / std
                    blob = np.transpose(x, (2, 0, 1))[None, ...]
                else:
                    img = cv2.resize(person_bgr, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
                    blob = np.transpose((img.astype(np.float32) / 255.0), (2, 0, 1))[None, ...]

                self.net.setInput(blob)
                out = self.net.forward()
                vec = np.asarray(out).reshape(-1).astype(np.float32)
                return self._l2_normalize(vec)

            else:  # OpenVINO
                if self.mode == "osnet":
                    img = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
                    x = img.astype(np.float32) / 255.0
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    x = (x - mean) / std
                    blob = np.transpose(x, (2, 0, 1))[None, ...]
                else:
                    img = cv2.resize(person_bgr, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
                    blob = np.transpose((img.astype(np.float32) / 255.0), (2, 0, 1))[None, ...]

                res_map = self._ov_compiled([blob])
                first_out = res_map[self._ov_compiled.outputs[0]]
                vec = np.asarray(first_out).reshape(-1).astype(np.float32)
                return self._l2_normalize(vec)
        except Exception:
            return None