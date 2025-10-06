from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from typing import Any
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

def build_app(state) -> FastAPI:
    app = FastAPI(title="MyDisplay Vision", version="0.1.0")

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
                time.sleep(1.0 / max(state.get_stream_fps(), 0.1))
        return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")

    return app
