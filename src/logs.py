# -*- coding: utf-8 -*-
"""
logs.py
--------
Semplice logging strutturato per la pipeline.
- setup_logging(cfg): inizializza una sessione con file JSONL a rotazione
- log_event(ev, **fields): scrive una riga JSON con ts, ev, sid

Note:
- I log sono pensati per il debug umano e per essere facilmente parsati
- I file vengono scritti in <log_dir>/events-<YYYYmmdd_HHMMSS>.jsonl
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_SESSION_ID: Optional[str] = None
_EVENT_LOGGER: Optional[logging.Logger] = None
_LOG_PATH: Optional[str] = None  # aggiunto


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_session_id() -> Optional[str]:
    return _SESSION_ID


def get_log_path() -> Optional[str]:
    return _LOG_PATH


def setup_logging(cfg) -> None:
    global _SESSION_ID, _EVENT_LOGGER, _LOG_PATH  # aggiornato
    if _EVENT_LOGGER is not None:
        return  # già configurato

    log_enabled = bool(getattr(cfg, "log_enabled", True))
    if not log_enabled:
        return

    log_dir = str(getattr(cfg, "log_dir", "logs") or "logs")
    os.makedirs(log_dir, exist_ok=True)

    _SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    level_str = str(getattr(cfg, "log_level", "INFO")).upper()
    level = getattr(logging, level_str, logging.INFO)

    # Logger eventi strutturati (JSONL)
    _EVENT_LOGGER = logging.getLogger("vision.events")
    _EVENT_LOGGER.setLevel(level)
    _EVENT_LOGGER.propagate = False

    rotate_mb = max(1, int(getattr(cfg, "log_rotate_mb", 10)))
    keep = max(1, int(getattr(cfg, "log_keep", 5)))

    fname = os.path.join(log_dir, f"events-{_SESSION_ID}.jsonl")
    try:
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(fname, maxBytes=rotate_mb * 1024 * 1024, backupCount=keep, encoding="utf-8")
    except Exception:
        fh = logging.FileHandler(fname, encoding="utf-8")
    fmt = logging.Formatter("%(message)s")
    fh.setFormatter(fmt)
    _EVENT_LOGGER.addHandler(fh)
    _LOG_PATH = fname  # salva percorso

    # Log iniziale di sessione
    base_info = {
        "ev": "SESSION_START",
        "ts": _now_iso(),
        "sid": _SESSION_ID,
        "cfg": {
            "camera": getattr(cfg, "camera", None),
            "target_fps": getattr(cfg, "target_fps", None),
            "debug_enabled": getattr(cfg, "debug_enabled", None),
            "debug_stream_fps": getattr(cfg, "debug_stream_fps", None),
            "count_mode": getattr(cfg, "count_mode", None),
        },
    }
    try:
        _EVENT_LOGGER.info(json.dumps(base_info, ensure_ascii=False))
    except Exception:
        pass


def log_event(ev: str, **fields: Any) -> None:
    """
    Scrive una riga JSON strutturata. Se non configurato, è no-op.
    Campi standard: ts, ev, sid
    """
    if _EVENT_LOGGER is None:
        return
    obj: Dict[str, Any] = {"ts": _now_iso(), "ev": str(ev), "sid": _SESSION_ID}
    for k, v in fields.items():
        try:
            # prova a serializzare; se fallisce, rappresentazione safe
            json.dumps(v)
            obj[k] = v
        except Exception:
            obj[k] = str(v)
    try:
        _EVENT_LOGGER.info(json.dumps(obj, ensure_ascii=False))
    except Exception:
        pass

