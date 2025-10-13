# -*- coding: utf-8 -*-
"""
src/config.py
AppConfig (solo parametri, nessun path a modelli)

- Niente più percorsi a file modello.
- Niente oggetti annidati (rimosso blocco "yunet").
- I modelli verranno risolti automaticamente da runtime in base a:
    models/
      face/
      person/
      genderage/
      reid_face/
      reid_body/
  con sottocartelle: openvino/ (xml+bin) e onnx/ (onnx).
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

    "debug_enabled": False,
    "debug_stream_fps": 5,
    "debug_resize_width": 960,  # 0 = nessun resize

    "api_host": "127.0.0.1",
    "api_port": 8080,

    # --- RTSP (se usi una IP cam) ---
    "rtsp_transport": "tcp",        # "tcp" | "udp"
    "rtsp_buffer_frames": 2,
    "rtsp_open_timeout_ms": 4000,
    "rtsp_read_timeout_ms": 4000,
    "rtsp_reconnect_sec": 2.0,
    "rtsp_max_failures": 60,

    # --- Logging strutturato ---
    "log_enabled": True,
    "log_dir": "logs",
    "log_level": "INFO",           # DEBUG | INFO | WARNING | ERROR
    "log_rotate_mb": 10,
    "log_keep": 5,

    # --- Counting mode ---
    "count_mode": "presence",       # "presence" | "tripwire"
    "presence_ttl_sec": 600,

    # --- Person detector (solo parametri: il modello è auto) ---
    "person_img_size": 640,
    "person_score_th": 0.26,
    "person_iou_th": 0.45,
    "person_max_det": 200,
    "person_backend": 0,            # OpenCV DNN backend id
    "person_target": 0,             # OpenCV DNN target id

    # Associazione volto->persona
    "face_assoc_iou_th": 0.20,
    "face_assoc_center_in": True,

    # --- Face detector (YuNet, solo parametri) ---
    "detector_score_th": 0.8,
    "detector_nms_iou": 0.3,
    "detector_top_k": 5000,
    "detector_backend": 0,
    "detector_target": 0,
    "detector_resize_width": 640,

    # --- Classifier: MODELLO COMBINATO (age+gender) ---
    # Default coerenti con Intel age-gender-recognition-retail-0013:
    # input 62x62 BGR, output (age/100, prob[F,M])
    "combined_input_size": [62, 62],
    "combined_bgr_input": True,
    "combined_scale01": False,
    "combined_age_scale": 100.0,
    "combined_gender_order": ["female", "male"],

    # Soglie generali per il classifier
    "age_buckets": ["0-13", "14-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    "cls_min_face_px": 64,
    "cls_min_conf": 0.35,
    "cls_interval_ms": 300,

    # --- Tracker ---
    "tracker_max_age": 8,
    "tracker_min_hits": 4,
    "tracker_iou_th": 0.35,

    # --- ROI / Tripwire ---
    "roi_tripwire": [[0.1, 0.5], [0.9, 0.5]],
    "roi_direction": "both",        # "both" | "a2b" | "b2a"
    "roi_band_px": 12,

    # --- Face Re-Identification ---
    "reid_enabled": True,
    "reid_similarity_th": 0.365,
    "reid_cache_size": 1000,
    "reid_memory_ttl_sec": 600,
    "reid_bank_size": 10,
    "reid_require_face_if_available": True,
    "debug_reid_verbose": False,

    # --- Body ReID ---
    "body_reid_input_w": 128,
    "body_reid_input_h": 256,
    "body_reid_backend": 0,
    "body_reid_target": 0,
    "body_only_th": 0.80,
    "reid_allow_body_seed": True,

    # --- Dedup conteggi ---
    "count_dedup_ttl_sec": 600,

    # --- Aggregazione / Metriche ---
    "metrics_window_sec": 60,
    "metrics_retention_min": 120,
}

class AppConfig:
    def __init__(self, data: Dict[str, Any]) -> None:
        d = dict(_DEFAULTS)
        d.update(data or {})
        # Espone tutti i campi come attributi
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

    # Utile per esporre la config via API (/config)
    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in _DEFAULTS.keys():
            out[k] = getattr(self, k, _DEFAULTS[k])
        return out
