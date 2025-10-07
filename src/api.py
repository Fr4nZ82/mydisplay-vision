# -*- coding: utf-8 -*-
"""
FastAPI app with:
- /health, /stats                         (come prima)
- /debug (HTML), /debug/frame, /debug/stream  (come prima)
- /metrics/minute?last=N                  (NUOVO; se aggregator presente)
- /config                                 (NUOVO; se config presente)
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, Response, JSONResponse
from typing import Any, Optional, Callable
from datetime import datetime, timezone
import time


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
    """
    Prova a leggere la config dallo state:
    - metodo state.get_config() → dict o oggetto con .data
    - chiave state["config"]
    - attributo state.config
    Restituisce un dict pronto per JSON o None.
    """
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
        # oggetto con .data (es. Config dataclass)
        data = getattr(cfg, "data", None)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


def _get_aggregator_from_state(state):
    """
    Prova a ottenere l'aggregatore dallo state:
    - metodo state.get_aggregator()
    - chiave state["aggregator"]
    - attributo state.aggregator
    """
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
    def debug_page() -> str:
        return """
<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <title>MyDisplay Vision - Debug</title>
  <style>
    :root { color-scheme: dark; }
    body { font-family: system-ui, sans-serif; padding: 16px; background: #111; color: #ddd; }
    .wrap { max-width: none; margin: 0 auto; }
    .controls { margin: 8px 0 16px; display: flex; gap: 8px; align-items: center; }
    button { background:#222; color:#ddd; border:1px solid #444; padding:6px 10px; border-radius:6px; cursor:pointer; }
    button:hover { background:#2a2a2a; }
    img { border: 2px solid #444; border-radius: 8px; display:block; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>MyDisplay Vision — Debug</h1>
    <p>Stream MJPEG live con annotazioni (volti rilevati).</p>

    <div class="controls">
      <button onclick="fit()">Fit</button>
      <button onclick="one()">1:1</button>
      <span id="info" style="opacity:.8"></span>
    </div>

    <img id="stream" src="/debug/stream" alt="debug stream" />

    <p style="opacity:.8">Se lo stream risultasse lento, prova <code><a href="/debug/frame" style="color:#9cf">/debug/frame</a></code> (singolo frame).</p>
  </div>

  <script>
    const img = document.getElementById('stream');
    function fit(){ img.style.maxWidth = '95vw'; img.style.width = ''; }
    function one(){ img.style.maxWidth = 'none'; img.style.width = ''; }

    // mostra dimensione acquisita (dalla /health)
    fetch('/health').then(r=>r.json()).then(j=>{
      document.getElementById('info').textContent = `capture: ${j.size[0]}×${j.size[1]}`;
    }).catch(()=>{});

    // default: fit
    fit();
  </script>
</body>
</html>
    """.strip()

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
                yield (b"--" + boundary.encode() + b"\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                       + jpg + b"\r\n")
                # conserva lo stesso controllo FPS lato server
                try:
                    fps = max(state.get_stream_fps(), 0.1)
                except Exception:
                    fps = 5.0
                time.sleep(1.0 / fps)
        return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")

    # ---------------------- NUOVI Endpoints ----------------------

    @app.get("/metrics/minute")
    def metrics_minute(last: int = 10):
        """
        Ritorna gli ultimi N aggregati (se l'aggregatore è disponibile nello state).
        Se non disponibile, restituisce [].
        """
        agg = _get_aggregator_from_state(state)
        if agg and hasattr(agg, "get_last"):
            try:
                last_int = max(1, min(int(last), 500))
            except Exception:
                last_int = 10
            return agg.get_last(last_int)
        # fallback: niente aggregatore disponibile
        return []

    @app.get("/config")
    def get_config():
        """
        Restituisce la configurazione attiva se disponibile nello state.
        """
        cfg = _get_config_from_state(state)
        if cfg is None:
            return JSONResponse({"error": "config not available"}, status_code=404)
        return JSONResponse(cfg)

    return app
