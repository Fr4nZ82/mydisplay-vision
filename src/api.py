# -*- coding: utf-8 -*-
"""
src/api.py
FastAPI app with:
- /health, /stats
- /debug (HTML), /debug/frame, /debug/stream 
- /metrics/minute?last=N
- /config
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, Response, JSONResponse
from typing import Any, Optional
from datetime import datetime, timezone
from pathlib import Path
import time

WEB_DIR = Path(__file__).resolve().parent / "web"   # -> src/web
DEBUG_HTML = WEB_DIR / "debug.html"

def _uptime_sec(since_iso: str | None) -> int | None:
    if not since_iso:
        return None
    try:
        since = datetime.fromisoformat(since_iso)
        now = datetime.now(timezone.utc)
        return int((now - since).total_seconds())
    except Exception:
        return None


def _get_config_from_state(state) -> Optional[dict]:
    try:
        if hasattr(state, "get_config") and callable(getattr(state, "get_config")):
            cfg = state.get_config()
        elif isinstance(state, dict) and "config" in state:
            cfg = state["config"]
        elif hasattr(state, "config"):
            cfg = getattr(state, "config")
        else:
            return None
        if isinstance(cfg, dict):
            return cfg
        data = getattr(cfg, "data", None)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


def _get_aggregator_from_state(state):
    try:
        if hasattr(state, "get_aggregator") and callable(getattr(state, "get_aggregator")):
            return state.get_aggregator()
        if isinstance(state, dict) and "aggregator" in state:
            return state["aggregator"]
        if hasattr(state, "aggregator"):
            return getattr(state, "aggregator")
    except Exception:
        pass
    return None


def build_app(state) -> FastAPI:
    app = FastAPI(title="MyDisplay Vision", version="0.1.0")

    # ---------------------- Endpoints "classici" ----------------------

    @app.get("/health")
    def health() -> dict[str, Any]:
        s = state.snapshot()
        return {
            "ok": s["camera_ok"],
            "fps": s["fps"],
            "size": s["size"],           # [w, h]
            "camera": "OK" if s["camera_ok"] else "DOWN",
            "since": s["since"],
            "version": "0.1.0",
        }

    @app.get("/stats")
    def stats() -> dict[str, Any]:
        s = state.snapshot()
        return {
            "since": s["since"],
            "uptimeSec": _uptime_sec(s["since"]),
            "framesTotal": s["frames_total"],
            "lastUpdate": s["last_update"],
            "lastFps": s["fps"],
            "size": s["size"],
            "cameraOk": s["camera_ok"],
        }

    @app.get("/debug", response_class=HTMLResponse)
    def debug_page() -> HTMLResponse:
        if DEBUG_HTML.exists():
            return HTMLResponse(DEBUG_HTML.read_text(encoding="utf-8"))
        # Fallback (se il file non esiste ancora)
        return HTMLResponse("<h1>MyDisplay Vision — Debug</h1><p>Manca src/web/debug.html</p>")
    
    @app.get("/debug/data")
    def debug_data():
        try:
            payload = state.get_reid_debug()
            return JSONResponse(payload)
        except Exception:
            return JSONResponse({'mem': [], 'active': {}}, status_code=200)

    @app.get("/debug/frame")
    def debug_frame() -> Response:
        jpg = state.get_debug_jpeg()
        if jpg is None:
            return Response(content=b"", media_type="image/jpeg", status_code=204)
        return Response(content=jpg, media_type="image/jpeg")

    @app.get("/debug/stream")
    def debug_stream():
        boundary = "frame"

        def gen():
            while True:
                jpg = state.get_debug_jpeg()
                if jpg is None:
                    time.sleep(0.1)
                    continue

                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                    + jpg + b"\r\n"
                )

                try:
                    fps = max(state.get_stream_fps(), 0.1)
                except Exception:
                    fps = 5.0
                time.sleep(1.0 / fps)

        return StreamingResponse(
            gen(),
            media_type=f"multipart/x-mixed-replace; boundary={boundary}",
            headers={"Cache-Control": "no-store"},
        )

    # ---------------------- NUOVI Endpoints ----------------------

    @app.get("/metrics/minute")
    def metrics_minute(last: int = 10, includeCurrent: int = 1):
        agg = _get_aggregator_from_state(state)
        if agg and hasattr(agg, "get_last"):
            try:
                last_int = max(1, min(int(last), 500))
            except Exception:
                last_int = 10
            out = list(agg.get_last(last_int))
            if includeCurrent and hasattr(agg, "get_current"):
                cur = agg.get_current()
                if cur:
                    # evita duplicato se coincide temporalmente con l'ultima chiusa
                    if not out or (out and out[-1].get("ts") != cur.get("ts")):
                        out.append(cur)
            return out
        return []

    @app.get("/config")
    def get_config():
        cfg = _get_config_from_state(state)
        if cfg is None:
            return JSONResponse({"error": "config not available"}, status_code=404)
        return JSONResponse(cfg)

    return app
