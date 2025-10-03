import argparse
import time
import sys
import threading
from datetime import datetime, timezone
import cv2
import numpy as np
import uvicorn

from src.config import AppConfig
from src.api import build_app

class HealthState:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            "camera_ok": False,
            "fps": 0.0,
            "size": [0, 0],  # [w, h]
            "since": datetime.now(timezone.utc).isoformat(timespec="seconds")
        }

    def update(self, *, camera_ok: bool, fps: float, width: int, height: int):
        with self._lock:
            self._data["camera_ok"] = camera_ok
            self._data["fps"] = float(fps)
            self._data["size"] = [int(width), int(height)]

    def snapshot(self):
        with self._lock:
            return dict(self._data)

def open_camera(index: int, width: int, height: int):
    """Prova MSMF, fallback DirectShow."""
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW]
    last_err = None
    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if cap.isOpened():
            if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(width))
            if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            return cap
        last_err = f"Backend {be} failed"
    raise RuntimeError(f"Cannot open camera index={index}. Last: {last_err}")

def start_api_server(state: HealthState, host: str, port: int):
    app = build_app(state)
    uvicorn.run(app, host=host, port=port, log_level="warning")

def main():
    ap = argparse.ArgumentParser(description="MyDisplay Vision - MVP service")
    ap.add_argument("--config", type=str, default="config.json", help="Path al file config.json")
    # override opzionali da CLI
    ap.add_argument("--camera", type=int, help="Indice camera (override)")
    ap.add_argument("--width",  type=int, help="Larghezza (override)")
    ap.add_argument("--height", type=int, help="Altezza (override)")
    ap.add_argument("--target-fps", type=float, help="FPS target (override)")
    args = ap.parse_args()

    cfg = AppConfig.load(args.config)
    # override da CLI se forniti
    if args.camera is not None:     cfg.camera = args.camera
    if args.width is not None:      cfg.width = args.width
    if args.height is not None:     cfg.height = args.height
    if args.target_fps is not None: cfg.target_fps = args.target_fps

    print("== MyDisplay Vision (MVP) ==")
    try:
        import onnxruntime as ort
        print(f"OpenCV: {cv2.__version__} | NumPy: {np.__version__} | ONNXRuntime: {ort.__version__}")
    except Exception as e:
        print(f"Info: onnxruntime non importabile ora (ok per questo step). Dettagli: {e}")

    # Stato condiviso per /health
    state = HealthState()

    # Avvio API HTTP in thread separato
    api_thread = threading.Thread(
        target=start_api_server,
        args=(state, cfg.api_host, cfg.api_port),
        daemon=True
    )
    api_thread.start()
    print(f"[API] /health su http://{cfg.api_host}:{cfg.api_port}/health")

    # Apertura camera
    try:
        cap = open_camera(cfg.camera, cfg.width, cfg.height)
    except Exception as e:
        print(f"[FATAL] Impossibile aprire la camera: {e}")
        state.update(camera_ok=False, fps=0.0, width=0, height=0)
        sys.exit(1)

    print(f"[OK] Camera aperta (index={cfg.camera}). Ctrl+C per uscire.")
    frame_count = 0
    last_ts = time.time()
    last_report = time.time()
    delay = 1.0 / max(cfg.target_fps, 0.1)

    try:
        while True:
            ok, frame = cap.read()
            now = time.time()
            if not ok or frame is None:
                time.sleep(0.05)
                state.update(camera_ok=False, fps=0.0, width=0, height=0)
                continue

            frame_count += 1
            # throttle semplice
            elapsed = now - last_ts
            if elapsed < delay:
                time.sleep(delay - elapsed)
            last_ts = time.time()

            # report ~1s
            if (now - last_report) >= 1.0:
                h, w = frame.shape[:2]
                fps = frame_count / (now - last_report)
                print(f"[health] fps={fps:.1f} | size={w}x{h} | camera=OK")
                state.update(camera_ok=True, fps=fps, width=w, height=h)
                frame_count = 0
                last_report = now

    except KeyboardInterrupt:
        print("\n[EXIT] Interrotto dall'utente.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
