# -*- coding: utf-8 -*-
"""
main.py (slim)
--------------
Bootstrap:
- parse args, carica AppConfig
- istanzia HealthState
- avvia API FastAPI (thread uvicorn) con il tuo api.py (build_app(state))
- avvia la pipeline (run_pipeline) che aggiorna lo State

Compatibile con il tuo api.py esistente (/debug HTML + stream) e con i nuovi moduli.
"""

from __future__ import annotations

import argparse
import threading
import time
import sys
import platform
import cv2
import numpy as np
import uvicorn

from src.config import AppConfig
from src.api import build_app
from src.state import HealthState
from src.runtime import run_pipeline


def _print_versions():
    py = platform.python_version()
    arch = platform.architecture()[0]
    try:
        import onnxruntime as ort
        ort_s = f"{ort.__version__} ({ort.get_device()})"
    except Exception as e:
        ort_s = f"n/a ({e})"
    print(f"Python: {py} ({arch})")
    print(f"OpenCV: {cv2.__version__} | NumPy: {np.__version__} | ONNXRuntime: {ort_s}")

def main():
    ap = argparse.ArgumentParser(description="MyDisplay Vision — service (API + pipeline)")
    ap.add_argument("--config", type=str, default="config.json", help="Path al file config.json")
    ap.add_argument("--camera", type=int, help="Indice camera (override)")
    ap.add_argument("--width",  type=int, help="Larghezza (override)")
    ap.add_argument("--height", type=int, help="Altezza (override)")
    ap.add_argument("--target-fps", type=float, help="FPS target (override)")
    args = ap.parse_args()

    cfg = AppConfig.load(args.config)
    if args.camera is not None:     cfg.camera = args.camera
    if args.width is not None:      cfg.width = args.width
    if args.height is not None:     cfg.height = args.height
    if args.target_fps is not None: cfg.target_fps = args.target_fps

    print("== MyDisplay Vision (API + Pipeline) ==")
    _print_versions()

    # Stato condiviso API <-> Pipeline
    state = HealthState(stream_fps=getattr(cfg, "debug_stream_fps", 5.0))

    # Avvio API FastAPI (thread separato)
    def _serve_api():
        app = build_app(state)
        uvicorn.run(app, host=getattr(cfg, "api_host", "127.0.0.1"), port=int(getattr(cfg, "api_port", 8765)), log_level="warning")

    api_thr = threading.Thread(target=_serve_api, name="api-server", daemon=True)
    api_thr.start()
    print(f"[API] /health, /stats, /debug su http://{getattr(cfg,'api_host','127.0.0.1')}:{getattr(cfg,'api_port',8765)}/debug")

    # Avvia la pipeline (blocking finché non esci)
    try:
        run_pipeline(state, cfg)
    except KeyboardInterrupt:
        print("\n[EXIT] Interrotto dall'utente.")
    finally:
        # La pipeline chiude la camera; l'API thread è daemon e si fermerà all'uscita del processo.
        pass


if __name__ == "__main__":
    main()
