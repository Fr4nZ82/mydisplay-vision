from fastapi import FastAPI
from typing import Any

def build_app(state: "HealthState") -> FastAPI:
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
            "version": "0.1.0"
        }

    return app
