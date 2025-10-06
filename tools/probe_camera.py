import argparse, time
import cv2

MODES = [
    (3840,2160),(2560,1440),(1920,1080),
    (1600,1200),(1280,960),(1280,720),
    (1024,768),(800,600),(640,480)
]
FPS_CANDIDATES = [30, 25, 20, 15, 10, 5]

def be_name(be):
    return {cv2.CAP_MSMF:"MSMF", cv2.CAP_DSHOW:"DShow"}.get(be, str(be))

def measure_fps(cap, seconds=1.2):
    start = time.time(); frames = 0
    while time.time() - start < seconds:
        ok, _ = cap.read()
        if not ok: break
        frames += 1
    return frames / max(time.time() - start, 1e-6)

def probe_backend(index, backend, seconds):
    print(f"\n=== Backend {be_name(backend)} ===")
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        print("  ! impossibile aprire la camera con questo backend")
        return
    try:
        for (w,h) in MODES:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(w))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
            ok, frame = cap.read()
            rw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            rh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            drv_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

            # prova qualche fps
            best_meas = 0.0; best_set = None
            for f in FPS_CANDIDATES:
                cap.set(cv2.CAP_PROP_FPS, float(f))
                _ = cap.read()  # 1 frame di “assestamento”
                meas = measure_fps(cap, seconds=seconds/3)
                if meas > best_meas:
                    best_meas, best_set = meas, f

            status = "OK" if ok else "FAIL"
            print(f"  req {w}x{h:>4} -> got {rw}x{rh:>4} | {status} | "
                  f"driver_fps≈{drv_fps:.1f} | measured≈{best_meas:.1f} (set:{best_set})")
    finally:
        cap.release()

def main():
    ap = argparse.ArgumentParser("Probe camera modes")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=1.5)
    args = ap.parse_args()
    for be in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
        probe_backend(args.camera, be, args.seconds)

if __name__ == "__main__":
    main()
