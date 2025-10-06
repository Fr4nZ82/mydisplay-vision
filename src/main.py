import argparse
import time
import sys
import threading
from datetime import datetime, timezone
import os
import cv2
import numpy as np
import uvicorn

from src.config import AppConfig
from src.api import build_app
from src.face_detector import YuNetDetector

class HealthState:
    def __init__(self, stream_fps: float = 5.0):
        self._lock = threading.Lock()
        self._data = {
            "camera_ok": False,
            "fps": 0.0,
            "size": [0, 0],  # [w, h]
            "since": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "frames_total": 0,
            "last_update": None,
        }
        self._debug_jpeg: bytes | None = None
        self._stream_fps: float = float(stream_fps)

    def update(self, *, camera_ok: bool, fps: float, width: int, height: int, frames_inc: int = 0):
        with self._lock:
            self._data["camera_ok"] = camera_ok
            self._data["fps"] = float(fps)
            self._data["size"] = [int(width), int(height)]
            self._data["frames_total"] += int(frames_inc)
            self._data["last_update"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def snapshot(self):
        with self._lock:
            return dict(self._data)

    def set_debug_jpeg(self, jpg: bytes | None):
        with self._lock:
            self._debug_jpeg = jpg

    def get_debug_jpeg(self) -> bytes | None:
        with self._lock:
            return self._debug_jpeg

    def set_stream_fps(self, fps: float):
        with self._lock:
            self._stream_fps = float(fps)

    def get_stream_fps(self) -> float:
        with self._lock:
            return self._stream_fps

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

def _resize_keep_aspect(img, target_w: int):
    if target_w <= 0:
        return img
    h, w = img.shape[:2]
    if w == 0:
        return img
    scale = target_w / float(w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def main():
    ap = argparse.ArgumentParser(description="MyDisplay Vision - MVP service (debug stream + YuNet)")
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

    print("== MyDisplay Vision (MVP) ==")
    try:
        import onnxruntime as ort
        print(f"OpenCV: {cv2.__version__} | NumPy: {np.__version__} | ONNXRuntime: {ort.__version__}")
    except Exception as e:
        print(f"Info: onnxruntime non importabile ora (ok per questo step). Dettagli: {e}")

    # Stato + API
    state = HealthState(stream_fps=cfg.debug_stream_fps)
    api_thread = threading.Thread(
        target=start_api_server,
        args=(state, cfg.api_host, cfg.api_port),
        daemon=True
    )
    api_thread.start()
    print(f"[API] /health, /stats, /debug su http://{cfg.api_host}:{cfg.api_port}/debug")

    # Apertura camera
    try:
        cap = open_camera(cfg.camera, cfg.width, cfg.height)
    except Exception as e:
        print(f"[FATAL] Impossibile aprire la camera: {e}")
        state.update(camera_ok=False, fps=0.0, width=0, height=0)
        sys.exit(1)
    print(f"[OK] Camera aperta (index={cfg.camera}). Ctrl+C per uscire.")

    # Inizializza YuNet
    detector = None
    try:
        detector = YuNetDetector(
            model_path=cfg.detector_model,
            score_th=cfg.detector_score_th,
            nms_iou=cfg.detector_nms_iou,
            top_k=cfg.detector_top_k,
            backend_id=cfg.detector_backend,
            target_id=cfg.detector_target,
        )
        print("[OK] YuNet caricato.")
    except FileNotFoundError as e:
        print(f"[WARN] {e}\nProcedo senza detection: lo stream mostrerà solo la camera.")

    frame_count = 0
    last_ts = time.time()
    last_report = time.time()
    delay = 1.0 / max(cfg.target_fps, 0.1)

    last_stream_t = 0.0
    stream_interval = 1.0 / max(cfg.debug_stream_fps, 0.1)

    try:
        while True:
            ok, frame = cap.read()
            now = time.time()
            if not ok or frame is None:
                time.sleep(0.05)
                state.update(camera_ok=False, fps=0.0, width=0, height=0)
                continue

            frame_count += 1

            # throttle semplice cattura
            elapsed = now - last_ts
            if elapsed < delay:
                time.sleep(delay - elapsed)
            last_ts = time.time()

            # report ~1s
            if (now - last_report) >= 1.0:
                h, w = frame.shape[:2]
                fps = frame_count / (now - last_report)
                print(f"[health] fps={fps:.1f} | size={w}x{h} | camera=OK")
                state.update(camera_ok=True, fps=fps, width=w, height=h, frames_inc=frame_count)
                frame_count = 0
                last_report = now

            # --- Debug stream con detection ---
            if cfg.debug_enabled and (now - last_stream_t) >= stream_interval:
                vis = _resize_keep_aspect(frame, cfg.debug_resize_width)

                if detector is not None:
                    # YuNet lavora a dimensione corrente dell'immagine vis
                    dets = detector.detect(vis)
                else:
                    dets = []

                # Disegna rettangoli verdi e score
                for (x, y, w, h), score in dets:
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"{score:.2f}"
                    cv2.putText(vis, label, (x, max(y - 6, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                ok_jpg, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok_jpg:
                    state.set_debug_jpeg(buf.tobytes())
                last_stream_t = now

    except KeyboardInterrupt:
        print("\n[EXIT] Interrotto dall'utente.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
