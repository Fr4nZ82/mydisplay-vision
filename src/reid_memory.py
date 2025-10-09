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

def _appearance_signature(bgr: np.ndarray, bins: int = 24, min_area_px: int = 900) -> Optional[np.ndarray]:
    if bgr is None or bgr.size == 0:
        return None
    h, w = bgr.shape[:2]
    if h * w < max(1, int(min_area_px)):
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256]).astype(np.float32)
    hist = cv2.normalize(hist, None).reshape(-1)
    # add simple geometric cues: aspect ratio and relative area (bounded)
    ar = np.array([w / (h + 1e-6), (h * w) / 1e6], dtype=np.float32)  # scale area roughly
    return _l2_normalize(np.concatenate([hist, ar], axis=0))

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
        # --- presence/appearance params (usati dal runtime e da assign_global_id) ---
        self.on_evict = None                 # callable(gid:int, meta:dict, last_ts:float)
        self.appearance_bins = 24
        self.appearance_min_area_px = 900
        self.appearance_weight = 0.35  # usato solo se volto + aspetto sono entrambi disponibili
        self.require_face_if_available = True   # abilita la policy di gating corpo→solo-ID-con-volto
        self.app_only_min_th = 0.82             # soglia più severa per match solo-aspetto
        # soglia dedicata per matching SOLO aspetto (più severa)
        self.app_th = 0.82
        self.face_gate = max(self.sim_th, 0.42)  # soglia minima per dire che due facce sono la stessa persona

        self.enabled = True
        self.next_global_id = 1
        # global_id -> {
        #   'feats': [face_embed,...],

        #   'app': [appearance_sig,...],
        #   'body': [body_embed,...],            # NEW
        #   'last': ts, 'created': ts, 'hits': int,

        #   'meta': {'gender_hist': Counter, 'age_hist': Counter}
        # }
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
        if self.prefer_oldest and second[0] is not None and best_id is not None:
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
    
    # --- setter usato dal runtime per configurare i parametri aspetto ---
    def set_appearance_params(
        self,
        bins: int = 24,
        min_area_px: int = 900,
        weight: float = 0.35,
        app_th: Optional[float] = None,   # compat: alias del vecchio nome
    ):
        self.appearance_bins = int(bins)
        self.appearance_min_area_px = int(min_area_px)
        self.appearance_weight = float(weight)
        if app_th is not None:
            # aggiorna ENTRAMBE, così la soglia ha effetto sul matching corrente
            self.app_only_min_th = float(app_th)
            self.app_th = float(app_th)



    def set_id_policy(self, appearance_weight=None, app_only_min_th=None, require_face_if_available=None, face_gate=None,
                      body_only_th=None, allow_body_seed=None):
        if appearance_weight is not None:
            self.appearance_weight = float(appearance_weight)
        if app_only_min_th is not None:
            self.app_only_min_th = float(app_only_min_th)
        if require_face_if_available is not None:
            self.require_face_if_available = bool(require_face_if_available)
        if face_gate is not None:
            self.face_gate = float(face_gate)
        if body_only_th is not None:
            self.body_only_th = float(body_only_th)
        if allow_body_seed is not None:
            self.allow_body_seed = bool(allow_body_seed)







































































































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


    def _sim_to_pid_body(self, pid: int, vec: np.ndarray) -> float:
        info = self.mem.get(pid)
        if not info:
            return -1.0


        bank = info.get('body', [])
        if not bank or vec is None:
            return -1.0

        return max(_cosine_sim(b, vec) for b in bank)




    # --- banca aspetto (già presente) ---
    def _add_app(self, pid: int, app_sig: np.ndarray, now: float):
        info = self.mem.get(pid)
        if not info:
            return
        bank = info.setdefault('app', [])
        bank.append(app_sig)
        if len(bank) > self.bank_size:
            bank.pop(0)
        info['last'] = now
        info['hits'] += 1
        self.mem[pid] = info

    def _sim_to_pid_app(self, pid: int, sig: np.ndarray) -> float:
        info = self.mem.get(pid)
        if not info:
            return -1.0
        bank = info.get('app', [])
        if not bank:
            return -1.0
        return max(_cosine_sim(f, sig) for f in bank)

    # --- NEW: fusione ranking (rispetta soglie finali) ---
    @staticmethod
    def _fuse(face_sim: float, body_sim: float, app_sim: float) -> float:
        parts = []
        if face_sim is not None and face_sim >= 0.0:
            parts.append((float(face_sim), 1.0))
        if body_sim is not None and body_sim >= 0.0:
            parts.append((float(body_sim), 0.7))
        if app_sim is not None and app_sim >= 0.0:
            parts.append((float(app_sim), 0.3))
        if not parts:
            return -1.0
        num = sum(s * w for s, w in parts)
        den = sum(w for _, w in parts)
        return float(num / (den + 1e-9))

    # ----------------- API principale -----------------
    def assign_global_id(self,
                         face_bgr_crop: Optional[np.ndarray],
                         kps5: Optional[np.ndarray] = None,
                         now_ts: Optional[float] = None,
                         appearance_bgr_crop: Optional[np.ndarray] = None,
                         body_bgr_crop: Optional[np.ndarray] = None) -> int:
        now = time.time() if now_ts is None else now_ts
        self._prune(now)

        # 1) embedding volto (se disponibile)
        feat = None
        if self.enabled and self.backend is not None and face_bgr_crop is not None:
            face_for_feat = None
            if self.backend_name == "sface" and kps5 is not None:
                try:
                    face_for_feat = _warp_face_bgr(face_bgr_crop, np.asarray(kps5, dtype=np.float32), (112,112))
                except Exception:
                    face_for_feat = None
            if face_for_feat is None:
                face_for_feat = face_bgr_crop
            feat = self.backend.feat(face_for_feat)

        # 2) embedding corpo (se backend presente + crop valido)
        body_vec = None
        if self.body_backend is not None and body_bgr_crop is not None:
            try:
                body_vec = self.body_backend.embed(body_bgr_crop)
                if body_vec is not None:
                    body_vec = _l2_normalize(np.asarray(body_vec, dtype=np.float32))
            except Exception:
                body_vec = None

        # 3) firma aspetto legacy (colore vestiti)
        app_sig = None
        if appearance_bgr_crop is not None:
            app_sig = _appearance_signature(
                appearance_bgr_crop,
                bins=self.appearance_bins,
                min_area_px=self.appearance_min_area_px
            )

        # Nessun segnale → nuovo ID
        if feat is None and body_vec is None and app_sig is None:
            gid = self.next_global_id
            self.next_global_id += 1
            self.mem[gid] = {'feats': [], 'app': [], 'body': [], 'last': now, 'created': now, 'hits': 1,
                             'meta': {'gender_hist': {}, 'age_hist': {}}}
            if self.debug:
                print(f"[ReID] new id(no signals) -> G{gid}")
            return gid

        # Costruisci candidati con similarità disponibili
        def _id_has_face(pid: int) -> bool:
            info = self.mem.get(pid)
            return bool(info and len(info.get('feats', [])) > 0)

        # Precalcola sim per ranking/debug
        cand = []
        for pid in self.mem.keys():
            fs = self._sim_to_pid(pid, feat) if feat is not None else -1.0
            bs = self._sim_to_pid_body(pid, body_vec) if body_vec is not None else -1.0
            asim = self._sim_to_pid_app(pid, app_sig) if app_sig is not None else -1.0
            fused = self._fuse(fs, bs, asim)
            cand.append((pid, fs, bs, asim, fused))
        cand.sort(key=lambda r: r[4], reverse=True)

        # ------------- Decisioni -------------
        chosen = None
        reason = "new"

        # A) Se c'è volto e supera soglia → match per volto
        if feat is not None:
            best_face = max(cand, key=lambda r: r[1]) if cand else None
            if best_face and best_face[1] >= self.sim_th:
                chosen = int(best_face[0])
                reason = f"face>=th({best_face[1]:.3f})"

        # B) Altrimenti, prova corpo con gate sugli ID ancorati (face) se configurato
        if chosen is None and body_vec is not None:
            # filtra candidati se richiesto
            cand_body = [(pid, fs, bs, asim, fused) for (pid, fs, bs, asim, fused) in cand
                         if (not self.require_face_if_available) or _id_has_face(pid)]
            best_body = max(cand_body, key=lambda r: r[2]) if cand_body else None
            if best_body and best_body[2] >= self.body_only_th:
                chosen = int(best_body[0])
                reason = f"body>=th({best_body[2]:.3f})"

        # C) In mancanza, valuta aspetto legacy (più severo)
        if chosen is None and app_sig is not None:
            cand_app = [(pid, fs, bs, asim, fused) for (pid, fs, bs, asim, fused) in cand
                        if (not self.require_face_if_available) or _id_has_face(pid)]
            best_app = max(cand_app, key=lambda r: r[3]) if cand_app else None
            if best_app and best_app[3] >= self.app_only_min_th:
                chosen = int(best_app[0])
                reason = f"app>=th({best_app[3]:.3f})"

        # D) Se nulla supera le soglie → crea nuovo
        if chosen is None:
            new_id = self.next_global_id
            self.next_global_id += 1
            self.mem[new_id] = {'feats': [], 'app': [], 'body': [], 'last': now, 'created': now, 'hits': 1,
                                'meta': {'gender_hist': {}, 'age_hist': {}}}
            # seeding banche
            if feat is not None:
                self._add_feat(new_id, feat, now)
            if body_vec is not None and self.allow_body_seed:
                self._add_body(new_id, body_vec, now)
            if app_sig is not None:
                # non vincolante: aggiungi anche aspetto legacy (opzionale)
                self._add_app(new_id, app_sig, now)
            if self.debug:
                top3 = cand[:3]
                print(f"[ReID] NEW -> G{new_id} | reason={reason} top3={[(p, round(f,3), round(b,3), round(a,3)) for (p,f,b,a,_) in top3]}")
            return new_id

        # Match su ID esistente
        pid = self.canon(chosen)
        if feat is not None:
            self._add_feat(pid, feat, now)
        if body_vec is not None:
            self._add_body(pid, body_vec, now)
        if app_sig is not None:
            self._add_app(pid, app_sig, now)

        if self.debug:
            top3 = cand[:3]
            print(f"[ReID] MATCH -> G{pid} | reason={reason} top3={[(p, round(f,3), round(b,3), round(a,3)) for (p,f,b,a,_) in top3]}")
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
