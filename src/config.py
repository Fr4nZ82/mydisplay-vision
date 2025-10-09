# -*- coding: utf-8 -*-
"""
AppConfig
---------
Carica config.json e fornisce attributi con default sensati.
Compatibile col tuo main e con la nuova pipeline (tracker, tripwire, aggregator, age/gender, modello combinato).

NOTE:
- Il tuo config.json attuale continua a funzionare senza modifiche.
- Se è presente un blocco "yunet": { onnx_path, score_th, nms_th, top_k }, viene usato per valorizzare i campi detector_* se non già impostati.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Tuple


_DEFAULTS: Dict[str, Any] = {
    # --- Camera / Stream / API ---
    "camera": 0,
    "width": 1920,
    "height": 1080,
    "target_fps": 10.0,

    "debug_enabled": True,
    "debug_stream_fps": 5,
    "debug_resize_width": 960,  # 0 = nessun resize, 960 consigliato per /debug

    "api_host": "127.0.0.1",
    "api_port": 8080,

    "rtsp_transport": "tcp",       # "tcp" o "udp"
    "rtsp_buffer_frames": 2,       # buffer interno del demuxer (frame)
    "rtsp_open_timeout_ms": 4000,  # timeout apertura
    "rtsp_read_timeout_ms": 4000,  # timeout lettura
    "rtsp_reconnect_sec": 2.0,     # attesa prima di riaprire
    "rtsp_max_failures": 60,       # quante read fallite (50ms ciascuna) prima di riaprire

    # --- Face detector (YuNet) ---
    # Compat: se presente "yunet": { onnx_path, score_th, nms_th, top_k } verrà mappato qui.
    "detector_model": "models/face_detection_yunet_2023mar.onnx",
    "detector_score_th": 0.8,
    "detector_nms_iou": 0.3,
    "detector_top_k": 5000,
    "detector_backend": 0,
    "detector_target": 0,
    "detector_resize_width": 640,  # opzionale: resize solo per il detector

    # --- Classifier: modelli SEPARATI (opzionali) ---
    "age_model_path": "models/age.onnx",
    "gender_model_path": "models/gender.onnx",
    "age_buckets": ["0-13", "14-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    "cls_min_face_px": 64,
    "cls_min_conf": 0.35,
    "cls_interval_ms": 300,

    # --- Classifier: MODELLO COMBINATO (consigliato) ---
    # Default tarati per Intel age-gender-recognition-retail-0013 (BGR 62x62, out: age/100 + prob[F,M])
    # Se non esiste il file indicato, il classifier prova i modelli separati; se non ci sono neppure quelli -> fallback unknown.
    "combined_model_path": "models/age-gender-recognition-retail-0013.onnx",
    "combined_input_size": [62, 62],
    "combined_bgr_input": True,
    "combined_scale01": False,
    "combined_age_scale": 100.0,
    "combined_gender_order": ["female", "male"],

    # --- Tracker ---
    "tracker_max_age": 15,
    "tracker_min_hits": 2,
    "tracker_iou_th": 0.35,

    # --- ROI / Tripwire ---
    "roi_tripwire": [[0.1, 0.5], [0.9, 0.5]],  # linea normalizzata A->B
    "roi_direction": "both",                    # "both" | "a2b" | "b2a"
    "roi_band_px": 12,

    # --- Re-Identification (facoltativo, usa YuNet landmarks + SFace) ---
    "reid_enabled": True,
    "reid_model_path": "models/face_recognition_sface_2021dec.onnx",
    "reid_similarity_th": 0.365,   # ~ soglia tipica SFace cosine (regola in base ai test)
    "reid_cache_size": 1000,       # max persone ricordate
    "reid_memory_ttl_sec": 600,    # 10 minuti: mantieni l'associazione global_id
    "reid_bank_size": 10,          # NUM descrittori per ID (feature bank)
    "reid_merge_sim": 0.55,        # se un nuovo embedding è così simile a un ID esistente → alias merge
    "reid_prefer_oldest": True,    # in caso di dubbio, tieni l'ID più vecchio

    # --- Dedup conteggi (non riconteggiare stessa persona entro questo TTL) ---
    "count_dedup_ttl_sec": 600,

    # --- Aggregazione / Metriche ---
    "metrics_window_sec": 60,
    "metrics_retention_min": 120,
}


class AppConfig:
    # Attributi valorizzati in load()
    def __init__(self, data: Dict[str, Any]) -> None:
        # copia pulita
        d = dict(_DEFAULTS)
        d.update(data or {})

        # Merge opzionale con blocco "yunet"
        yunet = d.get("yunet")
        if isinstance(yunet, dict):
            d.setdefault("detector_model", yunet.get("onnx_path", d["detector_model"]))
            d.setdefault("detector_score_th", yunet.get("score_th", d["detector_score_th"]))
            d.setdefault("detector_nms_iou", yunet.get("nms_th", d["detector_nms_iou"]))
            d.setdefault("detector_top_k", yunet.get("top_k", d["detector_top_k"]))

        # Esponi come attributi
        for k, v in d.items():
            setattr(self, k, v)

    @staticmethod
    def load(path: str = "config.json") -> "AppConfig":
        data: Dict[str, Any] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        return AppConfig(data)

    # Utile per /config (API read-only)
    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in _DEFAULTS.keys():
            out[k] = getattr(self, k, _DEFAULTS[k])
        # includi anche "yunet" se presente nel file originale
        if hasattr(self, "yunet"):
            out["yunet"] = getattr(self, "yunet")
        return out
