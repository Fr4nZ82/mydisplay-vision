# -*- coding: utf-8 -*-
"""
runtime.py
-----------
Loop della pipeline:
capture → (resize debug) → detect (YuNet) → track (SORT-lite)
→ classify (Age/Gender + cache) → tripwire (aggregator) → overlay → publish JPEG allo State.

Questo modulo NON avvia l'API: esegue solo la pipeline e aggiorna `HealthState`.
"""

from __future__ import annotations

import time
import sys
from typing import List, Tuple, Optional

import cv2
import numpy as np

from .state import HealthState
from .utils_vis import resize_keep_aspect, draw_box_with_label
from .tracker import SortLiteTracker
from .age_gender import AgeGenderClassifier
from .aggregator import MinuteAggregator
from .face_detector import YuNetDetector  # wrapper già usato nel tuo main precedente


# -------------------- Camera helpers --------------------

def open_camera(index: int, width: int, height: int):
    """Prova MSMF, fallback DirectShow (come nel main originale)."""
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW]
    last_err = None
    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if cap.isOpened():
            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            return cap
        last_err = f"Backend {be} failed"
    raise RuntimeError(f"Cannot open camera index={index}. Last: {last_err}")


# -------------------- Pipeline --------------------

def run_pipeline(state: HealthState, cfg) -> None:
    """
    Esegue la pipeline finché il processo è vivo.
    Richiede:
      - state: HealthState
      - cfg: AppConfig (o oggetto equivalente con gli attributi usati nel tuo main)
    """
    # Salva config/aggregator nello state per gli endpoint nuovi (/config, /metrics/minute)
    state.set_config_obj(cfg)

    # Aggregatore (con default se il campo manca)
    aggregator = MinuteAggregator(
        window_sec=int(getattr(cfg, "metrics_window_sec", 60)),
        retention_min=int(getattr(cfg, "metrics_retention_min", 120)),
    )
    state.set_aggregator(aggregator)

    # Tracker
    tracker = SortLiteTracker(
        max_age=int(getattr(cfg, "tracker_max_age", 15)),
        min_hits=int(getattr(cfg, "tracker_min_hits", 2)),
        iou_th=float(getattr(cfg, "tracker_iou_th", 0.35)),
        smooth_win=8,
        smooth_alpha=0.5,
    )

    # Classifier (safe: se modelli mancanti → enabled False e unknown)
    classifier = AgeGenderClassifier(
        # separati (restano compatibili e opzionali)
        age_model_path=getattr(cfg, "age_model_path", "models/age.onnx"),
        gender_model_path=getattr(cfg, "gender_model_path", "models/gender.onnx"),

        # COMBINATO (se presente lo usa in automatico)
        combined_model_path=getattr(cfg, "combined_model_path", "models/age-gender-recognition-retail-0013.onnx"),
        combined_input_size=tuple(getattr(cfg, "combined_input_size", (62, 62))),
        combined_bgr_input=bool(getattr(cfg, "combined_bgr_input", True)),
        combined_scale01=bool(getattr(cfg, "combined_scale01", False)),
        combined_age_scale=float(getattr(cfg, "combined_age_scale", 100.0)),
        combined_gender_order=tuple(getattr(cfg, "combined_gender_order", ("female", "male"))),

        # comuni
        age_buckets=tuple(getattr(cfg, "age_buckets", ("0-13","14-24","25-34","35-44","45-54","55-64","65+"))),
        input_size=(224, 224),
        cls_min_face_px=int(getattr(cfg, "cls_min_face_px", 64)),
        cls_min_conf=float(getattr(cfg, "cls_min_conf", 0.35)),
    )

    # Tripwire config (normalizzato 0..1)
    roi_tripwire = getattr(cfg, "roi_tripwire", [[0.1, 0.5], [0.9, 0.5]])
    roi_direction = getattr(cfg, "roi_direction", "both")
    roi_band_px = int(getattr(cfg, "roi_band_px", 12))

    # Detector (YuNet) come nel main originale
    detector = None
    try:
        detector = YuNetDetector(
            model_path=getattr(cfg, "detector_model", "models/face_detection_yunet_2023mar.onnx"),
            score_th=getattr(cfg, "detector_score_th", 0.8),
            nms_iou=getattr(cfg, "detector_nms_iou", 0.3),
            top_k=getattr(cfg, "detector_top_k", 5000),
            backend_id=getattr(cfg, "detector_backend", 0),
            target_id=getattr(cfg, "detector_target", 0),
        )
        print("[OK] YuNet caricato.")
    except FileNotFoundError as e:
        print(f"[WARN] {e}\nProcedo senza detection: lo stream mostrerà solo la camera.")

    # Apertura camera
    try:
        cap = open_camera(getattr(cfg, "camera", 0), getattr(cfg, "width", 1920), getattr(cfg, "height", 1080))
    except Exception as e:
        print(f"[FATAL] Impossibile aprire la camera: {e}")
        state.update(camera_ok=False, fps=0.0, width=0, height=0)
        sys.exit(1)
    print(f"[OK] Camera aperta (index={getattr(cfg,'camera',0)}). Ctrl+C per uscire.")

    # Loop vars
    frame_count = 0
    last_ts = time.time()
    last_report = time.time()
    delay = 1.0 / max(float(getattr(cfg, "target_fps", 15.0)), 0.1)

    last_stream_t = 0.0
    stream_interval = 1.0 / max(float(getattr(cfg, "debug_stream_fps", 5.0)), 0.1)

    # Classifier throttle per track
    cls_interval_ms = int(getattr(cfg, "cls_interval_ms", 300))
    last_cls_ts_per_track = {}
    cls_cache_per_track = {}

    try:
        while True:
            ok, frame = cap.read()
            now = time.time()
            if not ok or frame is None:
                time.sleep(0.05)
                state.update(camera_ok=False, fps=0.0, width=0, height=0)
                continue

            frame_count += 1

            # throttle cattura
            elapsed = now - last_ts
            if elapsed < delay:
                time.sleep(delay - elapsed)
            last_ts = time.time()

            # report ~1s
            if (now - last_report) >= 1.0:
                h, w = frame.shape[:2]
                fps = frame_count / (now - last_report)
                print(
                    f"[health] fps={fps:.1f} | size={w}x{h} | camera=OK | tracks={len(tracker.tracks)} | cls={'ON' if classifier.enabled else 'OFF'}"
                )
                state.update(camera_ok=True, fps=fps, width=w, height=h, frames_inc=frame_count)
                frame_count = 0
                last_report = now

            # --- Debug stream + pipeline ---
            if getattr(cfg, "debug_enabled", True) and (now - last_stream_t) >= stream_interval:
                # Usiamo 'vis' per detection e overlay (come nel main originale)
                vis = resize_keep_aspect(frame, getattr(cfg, "debug_resize_width", 960))
                vh, vw = vis.shape[:2]

                # Detection (YuNet su vis)
                if detector is not None:
                    dets_ori = detector.detect(vis)  # [((x,y,w,h), score), ...]
                    detections = [[float(x), float(y), float(w), float(h), float(score)] for (x, y, w, h), score in dets_ori]
                else:
                    detections = []

                # Tracking
                tracks = tracker.update(detections)

                # Classificazione (cache ogni cls_interval_ms, solo se volto abbastanza grande)
                for t in tracks:
                    tid = t["track_id"]
                    x, y, w, h = map(int, t["bbox"])
                    if min(w, h) >= int(getattr(cfg, "cls_min_face_px", 64)):
                        last_ts_t = last_cls_ts_per_track.get(tid, 0.0)
                        if (now - last_ts_t) * 1000.0 >= cls_interval_ms:
                            face_roi = vis[max(0, y): y + h, max(0, x): x + w]
                            res = classifier.infer(face_roi)
                            cls_cache_per_track[tid] = res
                            last_cls_ts_per_track[tid] = now
                    # smoothing nel tracker (anche se cache assente → unknown)
                    cached = cls_cache_per_track.get(tid, {"gender": "unknown", "ageBucket": "unknown", "confidence": 0.0})
                    tracker.apply_labels(tid, cached.get("gender", "unknown"), cached.get("ageBucket", "unknown"), cached.get("confidence", 0.0))

                # Tripwire crossing
                if isinstance(roi_tripwire, (list, tuple)) and len(roi_tripwire) == 2:
                    for t in tracks:
                        tid = t["track_id"]
                        tb = t["bbox"]
                        curr_c = (tb[0] + tb[2] * 0.5, tb[1] + tb[3] * 0.5)
                        prev_c = tracker.tracks.get(tid, {}).get("prev_center", None)
                        if prev_c is not None:
                            dir_tag = MinuteAggregator.check_crossing(
                                frame_w=vw,
                                frame_h=vh,
                                roi_tripwire=(tuple(roi_tripwire[0]), tuple(roi_tripwire[1])),
                                roi_band_px=roi_band_px,
                                roi_direction=roi_direction,
                                prev_center=prev_c,
                                curr_center=curr_c,
                            )
                            if dir_tag:
                                tr_state = tracker.tracks.get(tid, {})
                                gender = tr_state.get("gender", "unknown")
                                age_bucket = tr_state.get("ageBucket", "unknown")
                                aggregator.add_cross_event(gender=gender, age_bucket=age_bucket, direction_tag=dir_tag, track_id=tid, now=now)
                        # aggiorna prev_center dopo il check
                        tracker.update_prev_center(tid)

                # Overlay (box + ID + gender/age/conf + tripwire)
                if isinstance(roi_tripwire, (list, tuple)) and len(roi_tripwire) == 2:
                    ax, ay = int(roi_tripwire[0][0] * vw), int(roi_tripwire[0][1] * vh)
                    bx, by = int(roi_tripwire[1][0] * vw), int(roi_tripwire[1][1] * vh)
                    cv2.line(vis, (ax, ay), (bx, by), (255, 0, 0), max(1, int(roi_band_px)))

                for t in tracks:
                    tid = t["track_id"]
                    g = t.get("gender", "unknown")
                    a = t.get("ageBucket", "unknown")
                    c = t.get("conf", 0.0)
                    lbl = f"#{tid} {g[:1].upper() if g!='unknown' else '?'} / {a if a!='unknown' else '--'} ({c:.2f})"
                    draw_box_with_label(vis, t["bbox"], lbl, (0, 180, 0))

                # Pubblica JPEG allo state
                ok_jpg, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok_jpg:
                    state.set_debug_jpeg(buf.tobytes())
                last_stream_t = now

    except KeyboardInterrupt:
        print("\n[EXIT] Interrotto dall'utente.")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
