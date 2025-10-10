# -*- coding: utf-8 -*-
"""
Age/Gender classification via ONNX Runtime (CPU).

Supporta:
- MODELLI SEPARATI: age_model_path + gender_model_path
- MODELLO COMBINATO: combined_model_path (es. Intel age-gender-recognition-retail-0013.onnx, InsightFace genderage.onnx)

Comportamento:
- Se esiste combined_model_path → usa quello
- Altrimenti se esistono age_model_path e gender_model_path → usa quelli
- Altrimenti fallback sicuro: ritorna unknown con conf=0.0

Preprocess:
- separati: BGR→RGB, resize, normalizza a [0..1]
- combinato (default pensato per Intel Retail 0013): BGR, resize a 62x62, float32 0..255 (niente /255)
  * Parametri configurabili: combined_input_size, combined_bgr_input, combined_scale01, combined_age_scale.

Output atteso:
- gender: {'male'|'female'|'unknown'}
- ageBucket: uno tra default ("0-13","14-24","25-34","35-44","45-54","55-64","65+") o "unknown"
- confidence: conf per il genere (max softmax tra M/F) in [0,1]
"""

from __future__ import annotations
import os
import time
from typing import Dict, Tuple, Any, Optional, List

import cv2
import numpy as np

try:
    from openvino.runtime import Core as _OVCore
except Exception:
    _OVCore = None
try:
    import onnxruntime as ort
except Exception:
    ort = None  # gestito in fallback


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-9)


class AgeGenderClassifier:
    def __init__(
        self,
        combined_model_path: str = "",
        model_path: str = "",  # <--- AGGIUNTO per compat
        combined_input_size: Tuple[int, int] = (62, 62),
        combined_bgr_input: bool = True,
        combined_scale01: bool = False,
        combined_age_scale: float = 100.0,
        combined_gender_order: Tuple[str, str] = ("female", "male"),
        age_buckets: Tuple[str, ...] = ("0-13", "14-24", "25-34", "35-44", "45-54", "55-64", "65+"),
        input_size: Tuple[int, int] = (224, 224),
        cls_min_face_px: int = 64,
        cls_min_conf: float = 0.35,
    ) -> None:

        self._enabled = False
        self._mode = "none"  # "onnx" | "openvino" | "none"
        self._sess_combined = None
        self._ov_compiled = None
        self._ov_input = None

        # <<< INIZIALIZZA ATTRIBUTI USATI DAL PREPROCESS >>>
        self.c_input_w, self.c_input_h = map(int, combined_input_size)
        self.c_bgr = bool(combined_bgr_input)
        self.c_scale01 = bool(combined_scale01)
        self.c_age_scale = float(combined_age_scale)
        self.c_gender_order = tuple(combined_gender_order)
        self.age_buckets = tuple(age_buckets)
        self.cls_min_face_px = int(cls_min_face_px)
        self.cls_min_conf = float(cls_min_conf)
        self.input_w, self.input_h = map(int, input_size)  # per eventuale ramo "separate"

        # NOTA: usiamo SOLO il modello combinato.
        mp = model_path or combined_model_path
        if not mp or not os.path.exists(mp):
            return

        ext = os.path.splitext(mp)[1].lower()
        if ext == ".onnx":
            try:
                import onnxruntime as ort
                so = ort.SessionOptions(); so.log_severity_level = 3
                self._sess_combined = ort.InferenceSession(mp, so, providers=["CPUExecutionProvider"])
                self._mode = "onnx"
                self._enabled = True
                return
            except Exception:
                self._sess_combined = None

        if ext == ".xml" and _OVCore is not None:
            try:
                core = _OVCore()
                model = core.read_model(mp)  # .xml (carica .bin associato)
                self._ov_compiled = core.compile_model(model, "CPU")
                self._ov_input = self._ov_compiled.inputs[0]
                self._mode = "openvino"
                self._enabled = True
                return
            except Exception:
                self._ov_compiled = None
                self._ov_input = None



    @property
    def enabled(self) -> bool:
        return bool(self._enabled)

    # ---------- Preprocess ----------

    def _preprocess_separate(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """RGB, [0..1], NCHW @ input_size (default 224x224)"""
        if face_bgr is None or face_bgr.size == 0:
            return None
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # HWC->CHW
        x = np.expand_dims(x, 0)        # NCHW
        return x

    def _preprocess_combined(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Pensato per Intel 0013 e InsightFace:
        - Intel: BGR 62x62 float32 (0..255)
        - InsightFace: spesso BGR/RGB 96x96 float32; se serve, combined_scale01 True
        """
        if face_bgr is None or face_bgr.size == 0:
            return None

        if self.c_bgr:
            img = face_bgr
        else:
            img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(img, (self.c_input_w, self.c_input_h), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32)
        if self.c_scale01:
            x = x / 255.0
        x = np.transpose(x, (2, 0, 1))  # HWC->CHW
        x = np.expand_dims(x, 0)        # NCHW
        return x

    # ---------- Helpers ----------

    @staticmethod
    def _bucketize_age(age_est: float, age_buckets: Tuple[str, ...]) -> str:
        bins = ((0, 13), (14, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 200))
        labels = age_buckets
        for (lo, hi), lab in zip(bins, labels):
            if lo <= age_est <= hi:
                return lab
        return "unknown"

    # ---------- Inference ----------

    def infer(self, face_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Input: face crop (BGR). Se troppo piccolo (<cls_min_face_px) → unknown.
        Ritorna dict: {gender, ageBucket, confidence, ts}
        """
        now = time.time()

        if face_bgr is None or min(face_bgr.shape[:2]) < self.cls_min_face_px:
            return {"gender": "unknown", "ageBucket": "unknown", "confidence": 0.0, "ts": now}

        if not self.enabled:
            return {"gender": "unknown", "ageBucket": "unknown", "confidence": 0.0, "ts": now}

        try:
            # --- ONNX combined ---
            if self._mode == "onnx" and self._sess_combined is not None:
                x = self._preprocess_combined(face_bgr)
                if x is None:
                    return {"gender":"unknown","ageBucket":"unknown","confidence":0.0,"ts":now}
                outs = self._sess_combined.run(None, { self._sess_combined.get_inputs()[0].name: x })
                flat = [np.array(o).reshape(-1) for o in outs]

                age_val, gender_vec = None, None
                if len(flat) == 1 and flat[0].size == 3:
                    v = flat[0]; gender_vec, age_val = v[:2], float(v[2]) * self.c_age_scale
                else:
                    for v in flat:
                        if v.size == 2: gender_vec = v
                        elif v.size == 1: age_val = float(v[0]) * self.c_age_scale

                if gender_vec is None or age_val is None:
                    return {"gender":"unknown","ageBucket":"unknown","confidence":0.0,"ts":now}

                probs = _softmax(gender_vec.astype(np.float32))
                fem_idx = 0 if self.c_gender_order[0] == "female" else 1
                male_idx = 1 - fem_idx
                female_p, male_p = float(probs[fem_idx]), float(probs[male_idx])
                gender = "male" if male_p >= female_p else "female"
                g_conf = max(male_p, female_p)
                if g_conf < self.cls_min_conf:
                    gender = "unknown"
                age_bucket = self._bucketize_age(float(age_val), self.age_buckets)
                return {"gender":gender,"ageBucket":age_bucket,"confidence":float(g_conf),"ts":now}

            # --- OpenVINO combined ---
            if self._mode == "openvino" and self._ov_compiled is not None:
                x = self._preprocess_combined(face_bgr)
                if x is None:
                    return {"gender":"unknown","ageBucket":"unknown","confidence":0.0,"ts":now}
                res_map = self._ov_compiled([x])
                flat = [np.array(res_map[out]).reshape(-1) for out in self._ov_compiled.outputs]

                age_val, gender_vec = None, None
                if len(flat) == 1 and flat[0].size == 3:
                    v = flat[0]; gender_vec, age_val = v[:2], float(v[2]) * self.c_age_scale
                else:
                    for v in flat:
                        if v.size == 2: gender_vec = v
                        elif v.size == 1: age_val = float(v[0]) * self.c_age_scale

                if gender_vec is None or age_val is None:
                    return {"gender":"unknown","ageBucket":"unknown","confidence":0.0,"ts":now}

                probs = _softmax(gender_vec.astype(np.float32))
                fem_idx = 0 if self.c_gender_order[0] == "female" else 1
                male_idx = 1 - fem_idx
                female_p, male_p = float(probs[fem_idx]), float(probs[male_idx])
                gender = "male" if male_p >= female_p else "female"
                g_conf = max(male_p, female_p)
                if g_conf < self.cls_min_conf:
                    gender = "unknown"
                age_bucket = self._bucketize_age(float(age_val), self.age_buckets)
                return {"gender":gender,"ageBucket":age_bucket,"confidence":float(g_conf),"ts":now}


                # Heuristics per OUTPUT:
                # - Caso A (Intel): 2 uscite -> [age_norm scalar], [gender probs 2]
                # - Caso B (InsightFace): 1 uscita vettore 3 -> [female_logit, male_logit, age_norm] (oppure shape 1x3x1x1)
                age_val = None
                gender_vec = None

                # normalizza forme a 1D
                flat_outs = []
                for o in outs:
                    arr = np.array(o)
                    flat_outs.append(arr.reshape(-1))

                if len(flat_outs) == 1 and flat_outs[0].size == 3:
                    # InsightFace-like concatenato: [f_logit, m_logit, age_norm]
                    v = flat_outs[0]
                    gender_vec = v[:2]
                    age_val = float(v[2]) * self.c_age_scale
                else:
                    # Cerca un tensore di size 2 (genere) e uno di size 1 (età)
                    for v in flat_outs:
                        if v.size == 2:
                            gender_vec = v
                        elif v.size == 1:
                            age_val = float(v[0]) * self.c_age_scale

                # Se non ho trovato i campi, fallback
                if gender_vec is None or age_val is None:
                    return {"gender": "unknown", "ageBucket": "unknown", "confidence": 0.0, "ts": now}

                # Probabilità genere (softmax se serve)
                if gender_vec.ndim == 1 and gender_vec.size == 2:
                    probs = _softmax(gender_vec.astype(np.float32))
                    # Ordine: default ("female","male"), configurabile
                    fem_idx = 0 if self.c_gender_order[0] == "female" else 1
                    mal_idx = 1 - fem_idx
                    female_p, male_p = float(probs[fem_idx]), float(probs[mal_idx])
                else:
                    female_p, male_p = 0.5, 0.5

                gender = "male" if male_p >= female_p else "female"
                g_conf = max(male_p, female_p)
                if g_conf < self.cls_min_conf:
                    gender = "unknown"

                # Bucket età
                age_bucket = self._bucketize_age(float(age_val), self.age_buckets)

                return {"gender": gender, "ageBucket": age_bucket, "confidence": float(g_conf), "ts": now}

            # ---- Modalità SEPARATI ----
            elif self._mode == "separate" and (self._sess_age is not None and self._sess_gender is not None):
                x = self._preprocess_separate(face_bgr)
                if x is None:
                    return {"gender": "unknown", "ageBucket": "unknown", "confidence": 0.0, "ts": now}

                # GENDER
                g_inputs = {self._sess_gender.get_inputs()[0].name: x}
                g_out = self._sess_gender.run(None, g_inputs)[0].reshape(-1)
                if g_out.size == 2:
                    g_probs = _softmax(g_out.astype(np.float32))
                    female_p, male_p = float(g_probs[0]), float(g_probs[1])
                else:
                    return {"gender": "unknown", "ageBucket": "unknown", "confidence": 0.0, "ts": now}

                gender = "male" if male_p >= female_p else "female"
                g_conf = max(male_p, female_p)
                if g_conf < self.cls_min_conf:
                    gender = "unknown"

                # AGE
                a_inputs = {self._sess_age.get_inputs()[0].name: x}
                a_out = self._sess_age.run(None, a_inputs)[0].squeeze()
                age_bucket = "unknown"
                if a_out.ndim == 1 and a_out.shape[0] == len(self.age_buckets):
                    a_probs = _softmax(a_out.astype(np.float32))
                    idx = int(np.argmax(a_probs))
                    age_bucket = self.age_buckets[idx]
                elif np.isscalar(a_out) or (a_out.ndim == 0):
                    age_est = float(a_out)
                    age_bucket = self._bucketize_age(age_est, self.age_buckets)

                return {"gender": gender, "ageBucket": age_bucket, "confidence": float(g_conf), "ts": now}

            # ---- Nessun modello disponibile ----
            else:
                return {"gender": "unknown", "ageBucket": "unknown", "confidence": 0.0, "ts": now}

        except Exception:
            return {"gender": "unknown", "ageBucket": "unknown", "confidence": 0.0, "ts": now}
