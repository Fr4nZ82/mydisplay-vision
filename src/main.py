import argparse
import time
import sys
import cv2
import numpy as np

def open_camera(index: int, width: int, height: int):
    """Prova ad aprire la camera su Windows con MSMF, fallback a DirectShow."""
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW]
    last_err = None
    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if cap.isOpened():
            # Prova a impostare risoluzione desiderata (best-effort)
            if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(width))
            if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            return cap
        else:
            last_err = f"Backend {be} failed"
    raise RuntimeError(f"Cannot open camera index={index}. Last: {last_err}")

def main():
    ap = argparse.ArgumentParser(description="MyDisplay Vision - MVP service (capture smoke test)")
    ap.add_argument("--camera", type=int, default=0, help="Indice camera (default 0)")
    ap.add_argument("--width",  type=int, default=640, help="Larghezza desiderata (best-effort)")
    ap.add_argument("--height", type=int, default=480, help="Altezza desiderata (best-effort)")
    ap.add_argument("--target-fps", type=float, default=10.0, help="FPS target per throttling semplice")
    args = ap.parse_args()

    print("== MyDisplay Vision (MVP) ==")
    # Versioni librerie (utile per diagnostica)
    try:
        import onnxruntime as ort
        print(f"OpenCV: {cv2.__version__} | NumPy: {np.__version__} | ONNXRuntime: {ort.__version__}")
    except Exception as e:
        print(f"Info: onnxruntime non importabile ora (ok per questo step). Dettagli: {e}")

    # Apertura camera
    try:
        cap = open_camera(args.camera, args.width, args.height)
    except Exception as e:
        print(f"[FATAL] Impossibile aprire la camera: {e}")
        sys.exit(1)

    print(f"[OK] Camera aperta (index={args.camera}). Ctrl+C per uscire.")

    # Loop di cattura senza render GUI: stampa uno "health line" ogni ~1s
    frame_count = 0
    last_ts = time.time()
    last_report = time.time()
    delay = 1.0 / max(args.target_fps, 0.1)

    try:
        while True:
            ok, frame = cap.read()
            now = time.time()
            if not ok or frame is None:
                # Piccolo backoff se la lettura fallisce
                time.sleep(0.05)
                continue

            frame_count += 1
            # Throttle semplice per non saturare CPU
            elapsed = now - last_ts
            if elapsed < delay:
                time.sleep(delay - elapsed)
            last_ts = time.time()

            # Report ogni ~1s
            if (now - last_report) >= 1.0:
                h, w = frame.shape[:2]
                fps = frame_count / (now - last_report)
                print(f"[health] fps={fps:.1f} | size={w}x{h} | camera=OK")
                frame_count = 0
                last_report = now

    except KeyboardInterrupt:
        print("\n[EXIT] Interrotto dall'utente.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
