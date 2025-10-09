# -*- coding: utf-8 -*-
"""
reid_memory.py
--------------
Re-identificazione con memoria e TTL, con due backend:
- SFace (OpenCV contrib) se disponibile (FaceRecognizerSF_create)
- ArcFace ONNX via OpenCV DNN altrimenti (es. arcfaceresnet100-8.onnx)

Uso:
    from .reid_memory import FaceReID
    reid = FaceReID(model_path="models/arcfaceresnet100-8.onnx", similarity_th=0.36, cache_size=1000, memory_ttl_sec=600)
    gid = reid.assign_global_id(face_crop_bgr, kps5=None)  # kps5 opzionale (usato solo per SFace)
    reid.touch(gid)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
import time
import math

import numpy as np
import cv2


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))


def _l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x) + eps
    return x / n


def _warp_face_bgr(img: np.ndarray, kps: np.ndarray, out_size=(112, 112)) -> Optional[np.ndarray]:
    """
    Allinea il volto usando 5 landmarks (YuNet). Se i landmarks non sono disponibili, restituisce None.
    """
    if kps is None:
        return None
    kps = np.asarray(kps, dtype=np.float32)
    if kps.shape != (5, 2):
        return None

    # Punti canonici per 112x112 (tipici per SFace/ArcFace)
    dst = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(kps, dst, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(img, M, out_size, flags=cv2.INTER_LINEAR, borderValue=0)


class _BackendSFace:
    """
    Backend SFace basato su OpenCV contrib (FaceRecognizerSF).
    """
    def __init__(self, model_path: str):
        creator = getattr(cv2, "FaceRecognizerSF_create", None)
        if not callable(creator):
            raise RuntimeError("OpenCV non espone FaceRecognizerSF_create (serve opencv-contrib-python).")
        self.recognizer = creator(model_path, "")
        # output embedding è già normalizzato per cosine
        # feature(img) accetta volto allineato (112x112 BGR)

    def feat(self, face_bgr_112: np.ndarray) -> Optional[np.ndarray]:
        if face_bgr_112 is None or face_bgr_112.size == 0:
            return None
        try:
            f = self.recognizer.feature(face_bgr_112)
            # alcune build restituiscono shape (1,128). Rendiamo vettore 1D
            f = np.asarray(f).reshape(-1).astype(np.float32)
            return _l2_normalize(f)
        except Exception:
            return None


class _BackendArcFaceONNX:
    """
    Backend ArcFace ONNX generico via OpenCV DNN.
    Modelli tipici: arcfaceresnet100-8.onnx, face-recognition-resnet100-arcface-onnx.onnx
    Preprocess comune:
      - RGB 112x112
      - (img - 127.5) / 128.0  (≈ [-1, 1])
    Output: embedding (512,) o (1,512). Normalizzato L2 per cosine.
    """
    def __init__(self, model_path: str):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        # Se disponibile, preferisci ONNXRuntime / OpenVINO via DNN backends (facoltativo)
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception:
            pass

    @staticmethod
    def _preprocess(face_bgr: np.ndarray) -> np.ndarray:
        # Converte a RGB, resize 112x112, normalizza a [-1, 1] circa
        img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        blob = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW
        return blob

    def feat(self, face_bgr_any: np.ndarray) -> Optional[np.ndarray]:
        if face_bgr_any is None or face_bgr_any.size == 0:
            return None
        try:
            blob = self._preprocess(face_bgr_any)
            self.net.setInput(blob)
            out = self.net.forward()
            vec = np.asarray(out).reshape(-1).astype(np.float32)
            return _l2_normalize(vec)
        except Exception:
            return None


class FaceReID:
    """
    Re-identificazione con memoria:
      - calcola un embedding per ogni nuova faccia
      - cerca il best match nella cache via cosine
      - se sim >= threshold: riusa lo stesso global_id
      - altrimenti assegna un nuovo global_id

    Config:
      similarity_th: soglia cosine (0.3..0.5 tipico)
      cache_size:    max persone ricordate
      memory_ttl_sec:scadenza della memoria (secondi)
    """
    def __init__(self,
                 model_path: str,
                 similarity_th: float = 0.365,
                 cache_size: int = 1000,
                 memory_ttl_sec: int = 600,
                 bank_size: int = 10,
                 merge_sim: float = 0.55,
                 prefer_oldest: bool = True):
        self.sim_th = float(similarity_th)
        self.cache_size = int(cache_size)
        self.ttl = int(memory_ttl_sec)
        self.bank_size = int(bank_size)
        self.merge_sim = float(merge_sim)
        self.prefer_oldest = bool(prefer_oldest)

        self.enabled = True
        self.next_global_id = 1
        # global_id -> {'feats': [np.ndarray,...], 'last': ts, 'created': ts, 'hits': int}
        self.mem: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
        # alias map: nuovo_id -> id_canonico (vecchio)
        self.alias: Dict[int, int] = {}

        # Scegli backend: prova SFace, altrimenti ArcFace DNN
        self.backend_name = "unknown"
        self.backend = None
        try:
            # euristica: se il nome contiene 'sface' prova SFace prima
            if "sface" in (model_path or "").lower():
                self.backend = _BackendSFace(model_path)
                self.backend_name = "sface"
            else:
                # prova lo SFace comunque (se disponibile) — se fallisce, usa DNN
                try:
                    self.backend = _BackendSFace(model_path)
                    self.backend_name = "sface"
                except Exception:
                    self.backend = _BackendArcFaceONNX(model_path)
                    self.backend_name = "arcface_onnx"
        except Exception:
            # fallback duro: disabilita
            self.backend = None
            self.enabled = False
            self.backend_name = "disabled"

    # ----------------- memoria & utilità -----------------

    def _prune(self, now_ts: float):
        # rimuovi scaduti per TTL
        dead: List[int] = []
        for pid, info in self.mem.items():
            if now_ts - info['last'] > self.ttl:
                dead.append(pid)
        for pid in dead:
            self.mem.pop(pid, None)
            self.alias = {k: v for k, v in self.alias.items() if v != pid and k != pid}

        # limita la dimensione della cache
        overflow = len(self.mem) - self.cache_size
        if overflow > 0:
            # pop dai più vecchi (OrderedDict in inserimento/aggiornamento)
            for _ in range(overflow):
                self.mem.popitem(last=False)

    def _sim_to_pid(self, pid: int, feat: np.ndarray) -> float:
        info = self.mem.get(pid)
        if not info:
            return -1.0
        feats = info['feats']
        if not feats:
            return -1.0
        return max(float(np.dot(f, feat) / ((np.linalg.norm(f)+1e-9)*(np.linalg.norm(feat)+1e-9))) for f in feats)

    def _best_match(self, feat: np.ndarray) -> Tuple[Optional[int], float, Optional[Tuple[int,float]]]:
        best_id, best_sim = None, -1.0
        second = (None, -1.0)
        for pid in self.mem.keys():
            sim = self._sim_to_pid(pid, feat)
            if sim > best_sim:
                second = (best_id, best_sim)
                best_id, best_sim = pid, sim
            elif sim > second[1]:
                second = (pid, sim)
        # tie-break: se sim ~ uguali e prefer_oldest, scegli l'id più vecchio
        if self.prefer_oldest and second[0] is not None:
            if abs(best_sim - second[1]) <= 0.02 and second[0] < best_id:
                best_id, best_sim, second = second[0], second[1], (best_id, best_sim)
        return best_id, best_sim, second

    def _add_feat(self, pid: int, feat: np.ndarray, now: float):
        info = self.mem.get(pid)
        if not info:
            return
        feats = info['feats']
        feats.append(feat)
        if len(feats) > self.bank_size:
            feats.pop(0)
        info['last'] = now
        info['hits'] += 1
        self.mem[pid] = info

    def canon(self, pid: int) -> int:
        # risale catena alias finché possibile
        while pid in self.alias:
            pid = self.alias[pid]
        return pid

    # ----------------- API principale -----------------

    def assign_global_id(self, face_bgr_crop: np.ndarray, kps5: Optional[np.ndarray] = None,
                         now_ts: Optional[float] = None) -> int:
        """
        Ritorna un global_id stabile (se possibile). Se backend disabilitato, assegna ID nuovo.
        - face_bgr_crop: crop del volto (BGR) — se kps5 disponibili, si tenta allineamento (SFace).
        - kps5: landmarks 5-points (shape (5,2) in pixel) opzionali.
        """
        now = time.time() if now_ts is None else now_ts
        self._prune(now)

        if not self.enabled or self.backend is None:
            gid = self.next_global_id
            self.next_global_id += 1
            return gid

        # Se backend è SFace e ho landmarks: allinea a 112x112 per robustezza
        face_for_feat = None
        if self.backend_name == "sface" and kps5 is not None:
            try:
                face_for_feat = _warp_face_bgr(face_bgr_crop, np.asarray(kps5, dtype=np.float32), (112,112))
            except Exception:
                face_for_feat = None

        # fallback: usa il crop com'è (verrà resize dal backend ArcFace)
        if face_for_feat is None:
            face_for_feat = face_bgr_crop

        feat = self.backend.feat(face_for_feat)
        if feat is None:
            gid = self.next_global_id
            self.next_global_id += 1
            return gid

        pid, sim, _ = self._best_match(feat)
        if pid is not None and sim >= self.sim_th:
            self._add_feat(pid, feat, now)
            return self.canon(pid)

        # Nessun match sopra soglia → crea nuovo id
        new_id = self.next_global_id
        self.next_global_id += 1
        self.mem[new_id] = {'feats': [feat], 'last': now, 'created': now, 'hits': 1}

        # subito dopo la creazione, prova un MERGE con l'id migliore (anche se sotto sim_th)
        best_pid, best_sim, _ = self._best_match(feat)
        if best_pid is not None and best_pid != new_id and best_sim >= self.merge_sim:
            # aliasa new → best (preferisci il più vecchio)
            older = min(best_pid, new_id) if self.prefer_oldest else best_pid
            newer = new_id if older == best_pid else best_pid
            self.alias[newer] = older
            # unisci feats nel più vecchio
            for f in list(self.mem[newer]['feats']):
                self._add_feat(older, f, now)
            self.mem.pop(newer, None)
            return self.canon(older)
        return new_id

    def touch(self, global_id: int, now_ts: Optional[float] = None):
        now = time.time() if now_ts is None else now_ts
        pid = self.canon(global_id)
        info = self.mem.get(pid)
        if info:
            info['last'] = now
            self.mem[pid] = info
