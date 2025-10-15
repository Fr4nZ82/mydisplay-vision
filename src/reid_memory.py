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

Opzionale:
    - Backend corpo: reid.set_body_backend(backend) dove backend.embed(bgr)->np.ndarray
    - Passare body_bgr_crop a assign_global_id(...) per usare il re-id del corpo
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
import time

import numpy as np
import cv2

try:
    from openvino.runtime import Core as _OVCore
except Exception:
    _OVCore = None

# logging strutturato
try:
    from .logs import log_event as _log_event
except Exception:
    def _log_event(*args, **kwargs):
        return None

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
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception:
            pass

    @staticmethod
    def _preprocess(face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        blob = np.transpose(img, (2, 0, 1))[None, ...]
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


class _BackendArcFaceOV:
    """
    Backend ArcFace per OpenVINO IR (.xml+.bin), es. GhostFaceNet_W1.3_S1_ArcFace.
    Preprocess: RGB 112x112, (img - 127.5)/128.0. Il layout atteso può essere NCHW (1,3,112,112) o NHWC (1,112,112,3).
    """
    def __init__(self, model_path: str):
        if _OVCore is None:
            raise RuntimeError("OpenVINO runtime non disponibile")
        core = _OVCore()
        model = core.read_model(model_path)
        self.compiled = core.compile_model(model, "CPU")
        # Prova a dedurre layout input: preferisci NHWC se l'ultima dim è 3
        self.prefer_nhwc = True
        try:
            inp = self.compiled.input(0)
            shp = list(getattr(inp, "shape", []))
            # shape può contenere Dimension; prova a castare a int
            dims = []
            for d in shp:
                try:
                    dims.append(int(d))
                except Exception:
                    # per Dimension, prova get_length()
                    try:
                        dims.append(int(d.get_length()))
                    except Exception:
                        dims.append(-1)
            if len(dims) == 4:
                if dims[-1] == 3:
                    self.prefer_nhwc = True
                elif dims[1] == 3:
                    self.prefer_nhwc = False
        except Exception:
            pass

    # --- Preprocess CHW (NCHW) ---
    @staticmethod
    def _prep_arcface_rgb_chw(face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        blob = np.transpose(img, (2, 0, 1))[None, ...]
        return blob

    @staticmethod
    def _prep_arcface_bgr_chw(face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        img = (img - 127.5) / 128.0
        blob = np.transpose(img, (2, 0, 1))[None, ...]
        return blob

    @staticmethod
    def _prep_unit_rgb_chw(face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        blob = np.transpose(img, (2, 0, 1))[None, ...]
        return blob

    @staticmethod
    def _prep_unit_bgr_chw(face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        blob = np.transpose(img, (2, 0, 1))[None, ...]
        return blob

    # --- Preprocess HWC (NHWC) ---
    @staticmethod
    def _prep_arcface_rgb_hwc(face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        img = (img - 127.5) / 128.0
        blob = img[None, ...]  # (1,112,112,3)
        return blob

    @staticmethod
    def _prep_arcface_bgr_hwc(face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        img = (img - 127.5) / 128.0
        blob = img[None, ...]
        return blob

    @staticmethod
    def _prep_unit_rgb_hwc(face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        blob = img[None, ...]
        return blob

    @staticmethod
    def _prep_unit_bgr_hwc(face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        blob = img[None, ...]
        return blob

    def feat(self, face_bgr_any: np.ndarray) -> Optional[np.ndarray]:
        if face_bgr_any is None or face_bgr_any.size == 0:
            return None
        # Prepara lista di varianti: prova prima il layout dedotto
        variants_nchw = [
            ("arcface_rgb_chw", self._prep_arcface_rgb_chw),
            ("arcface_bgr_chw", self._prep_arcface_bgr_chw),
            ("unit_rgb_chw", self._prep_unit_rgb_chw),
            ("unit_bgr_chw", self._prep_unit_bgr_chw),
        ]
        variants_nhwc = [
            ("arcface_rgb_hwc", self._prep_arcface_rgb_hwc),
            ("arcface_bgr_hwc", self._prep_arcface_bgr_hwc),
            ("unit_rgb_hwc", self._prep_unit_rgb_hwc),
            ("unit_bgr_hwc", self._prep_unit_bgr_hwc),
        ]
        variants = (variants_nhwc + variants_nchw) if self.prefer_nhwc else (variants_nchw + variants_nhwc)

        for name, fn in variants:
            try:
                blob = fn(face_bgr_any)
                res = self.compiled([blob])
                out = list(res.values())[0]
                vec = np.asarray(out).reshape(-1).astype(np.float32)
                if vec.size == 0 or not np.isfinite(vec).all():
                    raise RuntimeError("empty_or_nan_vec")
                try:
                    _log_event("REID_DBG", kind="face", step="ov_feat_ok", backend="arcface_ov", preproc=name, dim=int(vec.size))
                except Exception:
                    pass
                return _l2_normalize(vec)
            except Exception as e:
                try:
                    _log_event("REID_DBG", kind="face", step="ov_try_fail", backend="arcface_ov", preproc=name, err=str(e))
                except Exception:
                    pass
                continue
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
    def __init__(self, model_path: str, similarity_th: float = 0.365, cache_size: int = 1000, memory_ttl_sec: int = 600, bank_size: int = 10):
        self.sim_th = float(similarity_th)
        self.cache_size = int(cache_size)
        self.ttl = int(memory_ttl_sec)
        self.bank_size = int(bank_size)
        # --- presence params ---
        self.on_evict = None                 # callable(gid:int, meta:dict, last_ts:float)
        self.require_face_if_available = True   # abilita la policy di gating corpo→solo-ID-con-volto


        self.enabled = True
        self.next_global_id = 1
        # global_id -> {
        #   'feats': [face_embed,...],
        #   'body': [body_embed,...],            # NEW
        #   'last': ts, 'created': ts, 'hits': int,
        #   'meta': {'gender_hist': Counter, 'age_hist': Counter}
        # }
        self.mem: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
        # alias map: nuovo_id -> id_canonico (vecchio)
        self.alias: Dict[int, int] = {}


                # Scegli backend: SFace (OpenCV contrib), ArcFace ONNX (OpenCV DNN) o ArcFace OpenVINO IR
        self.backend_name = "unknown"
        self.backend = None
        try:
            name_l = (model_path or "").lower()
            ext = (model_path.split(".")[-1].lower() if model_path else "")
            if "sface" in name_l:
                # Prova SFace (opencv-contrib)
                self.backend = _BackendSFace(model_path)
                self.backend_name = "sface"
            elif ext == "xml" and _OVCore is not None:
                # ArcFace OpenVINO (es. GhostFaceNet ArcFace IR)
                self.backend = _BackendArcFaceOV(model_path)
                self.backend_name = "arcface_ov"
            else:
                # ArcFace ONNX via OpenCV DNN
                self.backend = _BackendArcFaceONNX(model_path)
                self.backend_name = "arcface_onnx"
        except Exception:
            self.backend = None
            self.enabled = False
            self.backend_name = "disabled"


        # ---- NEW: body re-id backend + policy ----
        self.body_backend = None     # oggetto con .embed(bgr)->vec
        self.body_only_th = 0.80     # soglia per matching solo-corpo
        self.allow_body_seed = True  # consente creare ID con solo corpo
        self.debug = False           # stampa decisioni di assegnazione se True

    # ----------------- memoria & utilità -----------------

    def _prune(self, now_ts: float):
        dead: List[int] = []
        for pid, info in self.mem.items():
            if now_ts - info['last'] > self.ttl:
                dead.append(pid)
        for pid in dead:
            info = self.mem.pop(pid, None)
            self.alias = {k: v for k, v in self.alias.items() if v != pid and k != pid}
            # NEW: callback presenza
            if info and callable(self.on_evict):
                meta = info.get('meta', {})
                self.on_evict(int(pid), meta, float(info.get('last', now_ts)))

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

        if not feats or feat is None:
            return -1.0
        return max(float(np.dot(f, feat) / ((np.linalg.norm(f)+1e-9)*(np.linalg.norm(feat)+1e-9))) for f in feats) 

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
    
    def set_id_policy(self, require_face_if_available=None, body_only_th=None, allow_body_seed=None, min_face_px=None, face_body_bias=None, min_body_h_px=None):
        if require_face_if_available is not None:
            self.require_face_if_available = bool(require_face_if_available)
        if body_only_th is not None:
            self.body_only_th = float(body_only_th)
        if allow_body_seed is not None:
            self.allow_body_seed = bool(allow_body_seed)
        if min_face_px is not None:
            try:
                self.min_face_px = int(min_face_px)
            except Exception:
                self.min_face_px = 0
        if face_body_bias is not None:
            try:
                self.face_body_bias = float(face_body_bias)
            except Exception:
                self.face_body_bias = 0.0
        if min_body_h_px is not None:
            try:
                self.min_body_h_px = int(min_body_h_px)
            except Exception:
                self.min_body_h_px = 0



    # --- NEW: setter backend/body + debug ---
    def set_body_backend(self, backend) -> None:
        """backend: oggetto con metodo embed(person_bgr)->np.ndarray|None"""
        self.body_backend = backend

    def set_debug(self, verbose: bool) -> None:
        self.debug = bool(verbose)

    # --- NEW: banca corpo e similarità ---
    def _add_body(self, pid: int, vec: np.ndarray, now: float):
        info = self.mem.get(pid)

        if not info:
            return
        bank = info.setdefault('body', [])
        bank.append(np.asarray(vec, dtype=np.float32))
        if len(bank) > self.bank_size:
            bank.pop(0)
        info['last'] = now
        info['hits'] += 1
        self.mem[pid] = info

    def _merge_ids(self, winner: int, loser: int) -> int:
        """Unisce i dati del loser nel winner e cancella il loser. Ritorna l'ID vincente."""
        if winner == loser:
            return self.canon(winner)
        w = self.mem.get(winner)
        l = self.mem.get(loser)
        if w is None or l is None:
            # se uno non esiste, ritorna quello esistente
            if w is not None:
                return self.canon(winner)
            if l is not None:
                return self.canon(loser)
            return self.canon(winner)
        # merge feats
        for f in l.get('feats', []) or []:
            try:
                w.setdefault('feats', []).append(np.asarray(f, dtype=np.float32))
            except Exception:
                pass
        # limit bank size
        if len(w.get('feats', [])) > self.bank_size:
            w['feats'] = w['feats'][-self.bank_size:]
        # merge body
        for b in l.get('body', []) or []:
            try:
                w.setdefault('body', []).append(np.asarray(b, dtype=np.float32))
            except Exception:
                pass
        if len(w.get('body', [])) > self.bank_size:
            w['body'] = w['body'][-self.bank_size:]
        # merge meta
        wm = w.setdefault('meta', {'gender_hist': {}, 'age_hist': {}})
        lm = l.get('meta', {}) or {}
        for k, v in (lm.get('gender_hist', {}) or {}).items():
            wm['gender_hist'][k] = int(wm['gender_hist'].get(k, 0)) + int(v)
        for k, v in (lm.get('age_hist', {}) or {}).items():
            wm['age_hist'][k] = int(wm['age_hist'].get(k, 0)) + int(v)
        # timestamps / hits
        try:
            w['last'] = max(float(w.get('last', 0.0)), float(l.get('last', 0.0)))
        except Exception:
            pass
        try:
            w['created'] = min(float(w.get('created', w.get('last', 0.0))), float(l.get('created', l.get('last', 0.0))))
        except Exception:
            pass
        try:
            w['hits'] = int(w.get('hits', 1)) + int(l.get('hits', 0))
        except Exception:
            pass
        # salva e cancella loser
        self.mem[winner] = w
        # mappa alias per sicurezza, così canon() redirige eventuali riferimenti obsoleti
        self.alias[loser] = winner
        if loser in self.mem:
            del self.mem[loser]
        if self.debug:
            try:
                _log_event("REID_MERGE", winner=int(winner), loser=int(loser))
            except Exception:
                pass
        return self.canon(winner)

    def _sim_to_pid_body(self, pid: int, vec: np.ndarray) -> float:
        info = self.mem.get(pid)
        if not info:
            return -1.0

        bank = info.get('body', [])
        if not bank or vec is None:
            return -1.0

        return max(_cosine_sim(b, vec) for b in bank)

    # --- NEW: fusione ranking (rispetta soglie finali) ---
    @staticmethod
    def _fuse(face_sim: float, body_sim: float, _unused: float = None) -> float:
        parts = []
        if face_sim is not None and face_sim >= 0.0:
            parts.append((float(face_sim), 1.0))
        if body_sim is not None and body_sim >= 0.0:
            parts.append((float(body_sim), 0.7))
        if not parts:
            return -1.0
        num = sum(s * w for s, w in parts)
        den = sum(w for _, w in parts)
        return float(num / (den + 1e-9))

    # ----------------- API principale -----------------
    def assign_global_id(self, face_bgr_crop: Optional[np.ndarray], kps5: Optional[np.ndarray] = None, now_ts: Optional[float] = None, body_bgr_crop: Optional[np.ndarray] = None) -> int:

        now = time.time() if now_ts is None else now_ts
        self._prune(now)

        # 1) embedding volto (se disponibile)
        feat = None
        # Applica filtro dimensione minima volto per evitare inquinamento della banca
        min_face_px = int(getattr(self, 'min_face_px', 0) or 0)
        if min_face_px > 0 and face_bgr_crop is not None:
            try:
                fh, fw = face_bgr_crop.shape[:2]
                if min(fw, fh) < min_face_px:
                    if self.debug:
                        try:
                            _log_event("REID_DBG", kind="face", step="min_px_filter", fw=int(fw), fh=int(fh), min_px=int(min_face_px))
                        except Exception:
                            pass
                    face_bgr_crop = None
            except Exception:
                pass
        if self.enabled and self.backend is not None and face_bgr_crop is not None:
            face_for_feat = None
            # Applica warp con 5 landmark sia per SFace sia per backend ArcFace (OV/ONNX) se disponibili
            if kps5 is not None:
                try:
                    face_for_feat = _warp_face_bgr(face_bgr_crop, np.asarray(kps5, dtype=np.float32), (112,112))
                except Exception:
                    face_for_feat = None
            if face_for_feat is None:
                face_for_feat = face_bgr_crop
            feat = self.backend.feat(face_for_feat)
            if self.debug:
                try:
                    if feat is None:
                        _log_event("REID_DBG", kind="face", step="feat_none", backend=str(self.backend_name), kps=bool(kps5 is not None), shape=(int(face_for_feat.shape[1]), int(face_for_feat.shape[0])) if hasattr(face_for_feat, 'shape') else None)
                    else:
                        _log_event("REID_DBG", kind="face", step="feat_ok", dim=(int(feat.shape[0]) if hasattr(feat, 'shape') else None))
                except Exception:
                    pass

        # 2) embedding corpo (se backend presente + crop valido)
        body_vec = None
        # Applica filtro altezza minima del crop corpo per evitare inquinamento della banca
        min_body_h_px = int(getattr(self, 'min_body_h_px', 0) or 0)
        if min_body_h_px > 0 and body_bgr_crop is not None:
            try:
                bh = int(body_bgr_crop.shape[0])
                if bh < min_body_h_px:
                    if self.debug:
                        try:
                            _log_event("REID_DBG", kind="body", step="min_h_filter", bh=int(bh), min_h=int(min_body_h_px))
                        except Exception:
                            pass
                    body_bgr_crop = None
            except Exception:
                pass
        if self.body_backend is not None and body_bgr_crop is not None:
            try:
                body_vec = self.body_backend.embed(body_bgr_crop)
                if body_vec is not None:
                    body_vec = _l2_normalize(np.asarray(body_vec, dtype=np.float32))
                else:
                    if self.debug:
                        try:
                            _log_event("REID_DBG", kind="body", step="feat_none", backend=str(getattr(self.body_backend, 'mode', getattr(self.body_backend, '__class__', type('x',(),{})).__name__)), shape=(int(body_bgr_crop.shape[1]), int(body_bgr_crop.shape[0])) if hasattr(body_bgr_crop, 'shape') else None)
                        except Exception:
                            pass
            except Exception:
                body_vec = None

        # Nessun segnale → non creare un nuovo ID (evita "ID vuoti")
        if feat is None and body_vec is None:
            if self.debug:
                try:
                    _log_event("REID_SKIP", reason="no_signals")
                except Exception:
                    pass
            return -1

        # Costruisci candidati con similarità disponibili
        def _id_has_face(pid: int) -> bool:
            info = self.mem.get(pid)
            return bool(info and len(info.get('feats', [])) > 0)

        # Precalcola sim per ranking/debug
        cand = []
        for pid in self.mem.keys():
            fs = self._sim_to_pid(pid, feat) if feat is not None else -1.0
            bs = self._sim_to_pid_body(pid, body_vec) if body_vec is not None else -1.0
            fused = self._fuse(fs, bs, None)
            cand.append((pid, fs, bs, fused))
        cand.sort(key=lambda r: r[3], reverse=True)


        # ------------- Decisioni -------------
        chosen = None
        reason = "new"

        # Pre-selezione best per face/body (senza gating) per gestire merge
        best_face = max(cand, key=lambda r: r[1]) if (feat is not None and cand) else None
        best_body_all = max(cand, key=lambda r: r[2]) if (body_vec is not None and cand) else None

        # Se entrambi (face,body) puntano allo stesso ID e sono sopra soglia, scegli quello con sim maggiore (bias alla faccia)
        bias = float(getattr(self, 'face_body_bias', 0.0) or 0.0)
        if chosen is None and best_face and best_body_all and best_face[0] == best_body_all[0] and best_face[1] >= self.sim_th and best_body_all[2] >= self.body_only_th:
            if (best_face[1] + bias) >= best_body_all[2]:
                chosen = int(best_face[0])
                reason = f"face>=body (bias={bias:.3f})"
            else:
                chosen = int(best_body_all[0])
                reason = f"body>face (bias={bias:.3f})"

        # A) Se c'è volto e supera soglia → match per volto
        if chosen is None and best_face and best_face[1] >= self.sim_th:
            chosen = int(best_face[0])
            reason = f"face>=th({best_face[1]:.3f})"


        # B) Altrimenti, prova corpo con gate sugli ID ancorati (face) se configurato
        if chosen is None and body_vec is not None:
            # filtra candidati se richiesto
            cand_body = [(pid, fs, bs, fused) for (pid, fs, bs, fused) in cand
                        if (not self.require_face_if_available) or _id_has_face(pid)]
            best_body = max(cand_body, key=lambda r: r[2]) if cand_body else None
            if best_body and best_body[2] >= self.body_only_th:
                chosen = int(best_body[0])
                reason = f"body>=th({best_body[2]:.3f})"

        # C) Merge se in questo frame abbiamo sia face che body forti ma su ID diversi: vince la faccia
        if best_face and best_body_all and best_face[0] != best_body_all[0] and best_face[1] >= self.sim_th and best_body_all[2] >= self.body_only_th:
            winner = int(best_face[0])
            loser = int(best_body_all[0])
            winner = self._merge_ids(winner, loser)
            chosen = winner
            reason = f"merge(face_win) f={best_face[1]:.3f} b={best_body_all[2]:.3f}"

        # D) Se nulla supera le soglie → crea nuovo
        if chosen is None:
            new_id = self.next_global_id
            self.next_global_id += 1
            self.mem[new_id] = {'feats': [], 'body': [], 'last': now, 'created': now, 'hits': 1,
                                'meta': {'gender_hist': {}, 'age_hist': {}}}
            # seeding banche
            if feat is not None:
                self._add_feat(new_id, feat, now)
            if body_vec is not None and self.allow_body_seed:
                self._add_body(new_id, body_vec, now)
            if self.debug:
                top3 = cand[:3]
                try:
                    _log_event(
                        "REID_NEW",
                        gid=int(new_id),
                        reason=str(reason),
                        top3=[{"pid": int(p), "face": float(round(f, 4)), "body": float(round(b, 4))} for (p, f, b, _) in top3],
                    )
                except Exception:
                    pass
            return new_id

        # Match su ID esistente
        pid = self.canon(chosen)
        if feat is not None:
            self._add_feat(pid, feat, now)
        if body_vec is not None:
            # Se il match è guidato dal volto ma il corpo non somiglia al body attuale,
            # sostituiamo la banca corpo con il nuovo embedding per correggere dati sbagliati
            try:
                sim_b = self._sim_to_pid_body(pid, body_vec)
            except Exception:
                sim_b = -1.0
            if sim_b < self.body_only_th:
                info = self.mem.get(pid)
                if info is not None:
                    info['body'] = [np.asarray(body_vec, dtype=np.float32)]
                    info['last'] = now
                    info['hits'] = int(info.get('hits', 0)) + 1
                    self.mem[pid] = info
            else:
                self._add_body(pid, body_vec, now)
        if self.debug:
            top3 = cand[:3]
            try:
                _log_event(
                    "REID_MATCH",
                    gid=int(pid),
                    reason=str(reason),
                    top3=[{"pid": int(p), "face": float(round(f, 4)), "body": float(round(b, 4))} for (p, f, b, _) in top3],
                )
            except Exception:
                pass
        return pid

    def touch(self, global_id: int, now_ts: Optional[float] = None):
        now = time.time() if now_ts is None else now_ts
        pid = self.canon(global_id)
        info = self.mem.get(pid)
        if info:
            info['last'] = now
            self.mem[pid] = info

    def update_meta(self, pid: int, gender: str, age_bucket: str):
        pid = self.canon(pid)
        info = self.mem.get(pid)
        if not info:
            return
        m = info.setdefault('meta', {'gender_hist': {}, 'age_hist': {}})
        if gender:
            m['gender_hist'][gender] = int(m['gender_hist'].get(gender, 0)) + 1
        if age_bucket:
            m['age_hist'][age_bucket] = int(m['age_hist'].get(age_bucket, 0)) + 1
