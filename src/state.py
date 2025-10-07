# -*- coding: utf-8 -*-
"""
state.py
---------
Stato condiviso tra pipeline e API:
- HealthState: snapshot/aggiornamento health, JPEG debug buffer, FPS dello stream
- Espone anche get_aggregator() e get_config() per i nuovi endpoint
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import threading


@dataclass
class _StateData:
    camera_ok: bool = False
    fps: float = 0.0
    size: list[int] = field(default_factory=lambda: [0, 0])  # [w, h]
    since: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    frames_total: int = 0
    last_update: Optional[str] = None


class HealthState:
    """
    Thread-safe stato runtime per API e debug.
    Compatibile con api.py:
      - snapshot()
      - get_debug_jpeg() / set_debug_jpeg()
      - get_stream_fps() / set_stream_fps()
      - get_aggregator()
      - get_config()
    """

    def __init__(self, stream_fps: float = 5.0) -> None:
        self._lock = threading.RLock()
        self._data = _StateData()
        self._debug_jpeg: Optional[bytes] = None
        self._stream_fps: float = float(stream_fps)

        # opzionali, usati dagli endpoint /metrics/minute e /config
        self._aggregator = None
        self._config_obj = None

    # --------- Health ---------

    def update(self, *, camera_ok: bool, fps: float, width: int, height: int, frames_inc: int = 0) -> None:
        with self._lock:
            self._data.camera_ok = bool(camera_ok)
            self._data.fps = float(fps)
            self._data.size = [int(width), int(height)]
            self._data.frames_total += int(frames_inc)
            self._data.last_update = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            d = self._data
            return {
                "camera_ok": d.camera_ok,
                "fps": d.fps,
                "size": list(d.size),
                "since": d.since,
                "frames_total": d.frames_total,
                "last_update": d.last_update,
            }

    # --------- Debug frame (JPEG) ---------

    def set_debug_jpeg(self, jpg: Optional[bytes]) -> None:
        with self._lock:
            self._debug_jpeg = jpg

    def get_debug_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._debug_jpeg

    # --------- Stream FPS ---------

    def set_stream_fps(self, fps: float) -> None:
        with self._lock:
            self._stream_fps = float(fps)

    def get_stream_fps(self) -> float:
        with self._lock:
            return self._stream_fps

    # --------- Aggregator + Config (per nuovi endpoint) ---------

    def set_aggregator(self, aggregator) -> None:
        with self._lock:
            self._aggregator = aggregator

    def get_aggregator(self):
        with self._lock:
            return self._aggregator

    def set_config_obj(self, cfg_obj) -> None:
        with self._lock:
            self._config_obj = cfg_obj

    def get_config(self):
        """
        Restituisce un dict serializzabile della configurazione, se disponibile.
        - Se l'oggetto ha .to_dict() lo usa.
        - Altrimenti prova ad esporre gli attributi fondamentali.
        """
        with self._lock:
            cfg = self._config_obj
        if cfg is None:
            return None

        # 1) preferisci metodo dedicato
        try:
            if hasattr(cfg, "to_dict") and callable(getattr(cfg, "to_dict")):
                return cfg.to_dict()
        except Exception:
            pass

        # 2) fallback: estrai gli attributi più comuni
        keys = [
            "camera", "width", "height", "target_fps", "debug_enabled", "debug_resize_width",
            "debug_stream_fps", "api_host", "api_port",
            # detector
            "detector_model", "detector_score_th", "detector_nms_iou", "detector_top_k",
            "detector_backend", "detector_target", "detector_resize_width",
            # classifier
            "age_model_path", "gender_model_path", "age_buckets",
            "cls_min_face_px", "cls_min_conf", "cls_interval_ms",
            # tracker
            "tracker_max_age", "tracker_min_hits", "tracker_iou_th",
            # roi / metrics
            "roi_tripwire", "roi_direction", "roi_band_px",
            "metrics_window_sec", "metrics_retention_min",
        ]
        out = {}
        for k in keys:
            out[k] = getattr(cfg, k, None)
        return out
