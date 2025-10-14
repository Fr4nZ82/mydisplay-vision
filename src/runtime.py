# -*- coding: utf-8 -*-
"""
runtime.py
-----------
Loop della pipeline:
capture → (resize debug) → detect (YuNet) → track (SORT-lite)
→ classify (Age/Gender + cache) → (tripwire opzionale) → overlay → publish JPEG allo State.

Questo modulo NON avvia l'API: esegue solo la pipeline e aggiorna `HealthState`.
"""

from __future__ import annotations
import time
import sys
import os
from typing import Optional
import cv2
import numpy as np
from .utils_vis import (
    resize_keep_aspect, draw_box_with_label,
    draw_tripwire, associate_faces_to_tracks
)
from .state import HealthState
from .tracker import SortLiteTracker
from .age_gender import AgeGenderClassifier
from .aggregator import MinuteAggregator
from .face_detector import YuNetDetector
from .person_detector import PersonDetector
from .reid_memory import FaceReID
from .body_reid import BodyReID
from .model_resolver import resolve_all
from .logs import log_event

# == Helpers comuni ==
def _pick_path(info: dict) -> str:
    """Ritorna il path effettivo in base al kind (xml per openvino, onnx per onnx)."""
    if not info or not info.get("exists"):
        return ""
    return info["xml"] if info["kind"] == "openvino" else info.get("onnx", "")

def _row_auto(label: str, info: dict) -> str:
    if not info.get("exists"):
        return f"  - {label:<22}: (not found)"
    kind = info["kind"].upper()
    path = _pick_path(info)
    try:
        sz = os.path.getsize(path) / 1024 / 1024
        szs = f"{sz:.1f} MB" + (" + bin" if info["kind"] == "openvino" else "")
    except Exception:
        szs = "-"
    return f"  - {label:<22}: [{kind}] {path}  [OK | {szs}]"

def _resolve_and_log_models() -> dict:
    models = resolve_all("models")
    print("[MODEL CHECK]")
    print(_row_auto("Face detector", models["face"]))
    print(_row_auto("Person detector", models["person"]))
    print(_row_auto("Age/Gender", models["genderage"]))
    print(_row_auto("Face ReID", models["reid_face"]))
    print(_row_auto("Body ReID", models["reid_body"]))
    try:
        import onnxruntime as _ort  # noqa
        print("[ONNXRUNTIME] AVAILABLE")
    except Exception:
        print("[ONNXRUNTIME] NOT AVAILABLE")
    # Log modelli risolti (percorsi sintetici)
    try:
        log_event(
            "MODEL_RESOLVE",
            face=models.get("face"),
            person=models.get("person"),
            genderage=models.get("genderage"),
            reid_face=models.get("reid_face"),
            reid_body=models.get("reid_body"),
        )
    except Exception:
        pass
    return models

def _log_init(name: str, info: dict, path: str, ok: bool, extra: str = "") -> None:
    kind = (info.get("kind", "none") if info else "none").upper()
    path_eff = path or "-"
    status = "OK" if ok else "DISABLED"
    tail = f" {extra}" if extra else ""
    print(f"[INIT] {name:<12} kind={kind:<8} path={path_eff} status={status}{tail}")

# == Init modulari ==
def init_tracker(cfg):
    return SortLiteTracker(
        max_age=int(getattr(cfg, "tracker_max_age", 8)),
        min_hits=int(getattr(cfg, "tracker_min_hits", 4)),
        iou_th=float(getattr(cfg, "tracker_iou_th", 0.35)),
        smooth_win=8,
        smooth_alpha=0.5,
    )

def init_age_gender(cfg, models):
    mp = ""
    if models["genderage"]["exists"]:
        mp = _pick_path(models["genderage"])
    clsf = AgeGenderClassifier(
        combined_model_path=mp,
        combined_input_size=tuple(getattr(cfg, "combined_input_size", (62, 62))),
        combined_bgr_input=bool(getattr(cfg, "combined_bgr_input", True)),
        combined_scale01=bool(getattr(cfg, "combined_scale01", False)),
        combined_age_scale=float(getattr(cfg, "combined_age_scale", 100.0)),
        combined_gender_order=tuple(getattr(cfg, "combined_gender_order", ("female", "male"))),
        age_buckets=tuple(getattr(cfg, "age_buckets", ("0-13","14-24","25-34","35-44","45-54","55-64","65+"))),
        cls_min_face_px=int(getattr(cfg, "cls_min_face_px", 64)),
        cls_min_conf=float(getattr(cfg, "cls_min_conf", 0.35)),
    )
    _log_init("AgeGender", models["genderage"], mp, bool(getattr(clsf, "enabled", False)))
    try:
        log_event("INIT_AGE_GENDER", enabled=bool(getattr(clsf, "enabled", False)), path=mp)
    except Exception:
        pass
    return clsf

def init_face_detector(cfg, models):
    mp = _pick_path(models["face"]) if models["face"]["exists"] else ""
    det = None
    if models["face"]["exists"] and models["face"]["kind"] == "onnx":
        det = YuNetDetector(
            model_path=mp,
            score_th=getattr(cfg, "detector_score_th", 0.8),
            nms_iou=getattr(cfg, "detector_nms_iou", 0.3),
            top_k=getattr(cfg, "detector_top_k", 5000),
            backend_id=getattr(cfg, "detector_backend", 0),
            target_id=getattr(cfg, "detector_target", 0),
        )
    _log_init(
        "FaceDetector",
        models["face"],
        mp,
        det is not None,
        extra="(note: OpenVINO YuNet non supportato)" if (models["face"]["exists"] and models["face"]["kind"] == "openvino") else "",
    )
    try:
        log_event("INIT_FACE_DET", enabled=det is not None, kind=models["face"].get("kind"), path=mp)
    except Exception:
        pass
    return det

def init_person_detector(cfg, models):
    mp = _pick_path(models["person"]) if models["person"]["exists"] else ""
    det = None
    if models["person"]["exists"]:
        det = PersonDetector(
            model_path=mp,
            img_size=int(getattr(cfg, "person_img_size", 640)),
            score_th=float(getattr(cfg, "person_score_th", 0.26)),
            iou_th=float(getattr(cfg, "person_iou_th", 0.45)),
            max_det=int(getattr(cfg, "person_max_det", 200)),
            backend_id=int(getattr(cfg, "person_backend", 0)),
            target_id=int(getattr(cfg, "person_target", 0)),
            ov_device=str(getattr(cfg, "person_ov_device", "CPU")),
        )
    auto_kind = "OV" if (mp.lower().endswith(".xml")) else (models["person"].get("kind", "?").upper() if mp else "-")
    _log_init("PersonDet", models["person"], mp, det is not None, extra=f"(auto kind={auto_kind})")
    try:
        log_event("INIT_PERSON_DET", enabled=det is not None, kind=auto_kind, path=mp)
    except Exception:
        pass
    return det

def init_face_reid(cfg, models):
    # Accetta ONNX o OpenVINO IR
    reid_enabled = bool(getattr(cfg, "reid_enabled", True))
    mp = _pick_path(models["reid_face"]) if models["reid_face"]["exists"] else ""
    if not reid_enabled or not mp:
        reid = FaceReID("", 1.0, 1, 1)
    else:
        reid = FaceReID(
            model_path=mp,
            similarity_th=float(getattr(cfg, "reid_similarity_th", 0.365)),
            cache_size=int(getattr(cfg, "reid_cache_size", 1000)),
            memory_ttl_sec=int(getattr(cfg, "reid_memory_ttl_sec", 600)),
            bank_size=int(getattr(cfg, "reid_bank_size", 10)),
        )
    _log_init("FaceReID", models["reid_face"], mp, bool(getattr(reid, "enabled", False)), extra=f"backend={getattr(reid, 'backend_name', '?')}")
    try:
        log_event("INIT_FACE_REID", enabled=bool(getattr(reid, "enabled", False)), path=mp, backend=getattr(reid, 'backend_name', '?'))
    except Exception:
        pass
    return reid

def init_body_reid(cfg, models, reid):
    if models["reid_body"]["exists"]:
        mp = _pick_path(models["reid_body"])  # preferisce OpenVINO se presente
        backend = BodyReID(
            model_path=mp,
            backend_id=int(getattr(cfg, "body_reid_backend", 0)),
            target_id=int(getattr(cfg, "body_reid_target", 0)),
            input_size=(int(getattr(cfg, "body_reid_input_w", 128)), int(getattr(cfg, "body_reid_input_h", 256))),
        )
        if hasattr(reid, "set_body_backend"):
            reid.set_body_backend(backend)
        _log_init("BodyReID", models["reid_body"], mp, bool(getattr(backend, "enabled", True)), extra=f"mode={getattr(backend, 'mode', '?')}")
        try:
            log_event("INIT_BODY_REID", enabled=True, path=mp, mode=getattr(backend, 'mode', '?'))
        except Exception:
            pass
        return backend
    _log_init("BodyReID", models["reid_body"], "", False)
    try:
        log_event("INIT_BODY_REID", enabled=False)
    except Exception:
        pass
    return None

# -------------------- Runtime libs --------------------
try:
    import onnxruntime as _ort  # noqa
    _ORT_OK = True
except Exception:
    _ORT_OK = False

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

def open_rtsp(url: str, cfg):
    """
    Apre un flusso RTSP via FFmpeg (OpenCV). Supporta opzioni base da config.
    """
    transport = getattr(cfg, "rtsp_transport", "tcp")
    url_eff = url
    if url.lower().startswith("rtsp://") and transport and "rtsp_transport=" not in url:
        sep = "&" if "?" in url else "?"
        url_eff = f"{url}{sep}rtsp_transport={transport}"

    cap = cv2.VideoCapture(url_eff, cv2.CAP_FFMPEG)

    # Timeout (best-effort)
    try:
        ot = float(getattr(cfg, "rtsp_open_timeout_ms", 4000))
        rt = float(getattr(cfg, "rtsp_read_timeout_ms", 4000))
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, ot)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, rt)
    except Exception:
        pass

    # Buffer interno (in frame)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, float(getattr(cfg, "rtsp_buffer_frames", 2)))
    except Exception:
        pass

    return cap

# -------------------- Pipeline --------------------
def run_pipeline(state: HealthState, cfg) -> None:
    """
    Esegue la pipeline finché il processo è vivo.
    Richiede:
      - state: HealthState
      - cfg: AppConfig (o oggetto equivalente con gli attributi usati nel tuo main)
    """
    # Salva config/aggregator nello state per gli endpoint (/config, /metrics/minute)
    state.set_config_obj(cfg)

    # Aggregatore (metrics)
    aggregator = MinuteAggregator(
        window_sec=int(getattr(cfg, "metrics_window_sec", 60)),
        retention_min=int(getattr(cfg, "metrics_retention_min", 120)),
    )
    state.set_aggregator(aggregator)

    # Pipeline start log
    try:
        log_event(
            "PIPELINE_START",
            count_mode=str(getattr(cfg, "count_mode", "presence")),
            tripwire=getattr(cfg, "roi_tripwire", None),
            roi_direction=getattr(cfg, "roi_direction", None),
            roi_band_px=int(getattr(cfg, "roi_band_px", 12)),
        )
    except Exception:
        pass

    # Tracker
    tracker = init_tracker(cfg)

    # Risoluzione modelli + log
    models = _resolve_and_log_models()
    if not _ORT_OK:
        print("[WARN] ONNXRuntime non disponibile: i moduli ONNX potrebbero non inizializzarsi.")

    # Classifier Age/Gender (combined; preferisce OpenVINO se presente)
    classifier = init_age_gender(cfg, models)

    # Tripwire config
    roi_tripwire = getattr(cfg, "roi_tripwire", [[0.1, 0.5], [0.9, 0.5]])
    roi_direction = getattr(cfg, "roi_direction", "both")
    roi_band_px = int(getattr(cfg, "roi_band_px", 12))

    # Face ReID (memoria) + policy
    reid = init_face_reid(cfg, models)
    try:
        reid.set_id_policy(
            require_face_if_available=bool(getattr(cfg, "reid_require_face_if_available", True)),
            body_only_th=float(getattr(cfg, "body_only_th", 0.80)),
            allow_body_seed=bool(getattr(cfg, "reid_allow_body_seed", True)),
        )
        if hasattr(reid, "set_debug"):
            reid.set_debug(bool(getattr(cfg, "debug_reid_verbose", False)))
    except Exception:
        pass

    # Body ReID (collega al FaceReID se presente)
    _ = init_body_reid(cfg, models, reid)

    # Dedup counting & eviction callback
    count_mode = str(getattr(cfg, "count_mode", "presence")).lower()
    presence_ttl = int(getattr(cfg, "presence_ttl_sec", getattr(cfg, "reid_memory_ttl_sec", 600)))
    reid_ttl = int(getattr(cfg, "reid_memory_ttl_sec", 600))
    try:
        reid.ttl = presence_ttl if count_mode == "presence" else reid_ttl
    except Exception:
        pass

    def _on_evict(gid: int, meta: dict, last_ts: float):
        if count_mode != "presence":
            return
        try:
            gh = (meta or {}).get('gender_hist', {}) or {}
            ah = (meta or {}).get('age_hist', {}) or {}
            gender = max(gh.items(), key=lambda kv: kv[1])[0] if gh else "unknown"
            age_bucket = max(ah.items(), key=lambda kv: kv[1])[0] if ah else "unknown"
        except Exception:
            gender, age_bucket = "unknown", "unknown"
        aggregator.add_presence_event(gender=gender, age_bucket=age_bucket, global_id=gid, now=last_ts)
        try:
            log_event("PRESENCE", gid=int(gid), gender=gender, age=age_bucket)
        except Exception:
            pass

    if hasattr(reid, "on_evict"):
        try:
            reid.on_evict = _on_evict
        except Exception:
            pass

    # Dedup per i conteggi (tripwire): non riconteggiare lo stesso global_id entro TTL
    from collections import OrderedDict
    count_seen = OrderedDict()

    def should_count(global_id: int, now_ts: float) -> bool:
        ttl = int(getattr(cfg, "count_dedup_ttl_sec", 600))
        last = count_seen.get(global_id)
        if (last is not None) and ((now_ts - last) < ttl):
            return False
        count_seen[global_id] = now_ts
        # tenue pruning per non far crescere la mappa all’infinito
        if len(count_seen) > 5000:
            while len(count_seen) > 4000:
                count_seen.popitem(last=False)
        return True

    # Detector volti/persona
    face_detector = init_face_detector(cfg, models)
    if face_detector is not None:
        print("[OK] YuNet caricato.")
    person_detector = init_person_detector(cfg, models)
    if person_detector is not None:
        print("[OK] Person detector caricato.")

    # Apertura camera
    try:
        cam_cfg = getattr(cfg, "camera", 0)
        is_rtsp = isinstance(cam_cfg, str) and cam_cfg.lower().startswith("rtsp://")
        if is_rtsp:
            cap = open_rtsp(cam_cfg, cfg)
        else:
            cap = open_camera(int(cam_cfg), getattr(cfg, "width", 1920), getattr(cfg, "height", 1080))
        if not cap or not cap.isOpened():
            raise RuntimeError(f"Cannot open {'RTSP' if is_rtsp else 'camera'}: {cam_cfg}")
    except Exception as e:
        print(f"[FATAL] Impossibile aprire la camera: {e}")
        try:
            log_event("CAMERA_OPEN_FAIL", camera=str(getattr(cfg, 'camera', cam_cfg)), error=str(e))
        except Exception:
            pass
        state.update(camera_ok=False, fps=0.0, width=0, height=0)
        sys.exit(1)

    print(f"[OK] Camera aperta (index={getattr(cfg,'camera',0)}). Ctrl+C per uscire.")
    try:
        log_event("CAMERA_OPEN_OK", camera=str(getattr(cfg,'camera',0)), is_rtsp=bool(is_rtsp))
    except Exception:
        pass

    # Loop vars
    frame_count = 0
    last_ts = time.time()
    last_report = time.time()
    delay = 1.0 / max(float(getattr(cfg, "target_fps", 15.0)), 0.1)
    last_stream_t = 0.0
    stream_interval = 1.0 / max(float(getattr(cfg, "debug_stream_fps", 5.0)), 0.1)

    # ---- Helpers per post-filtri persona ----
    def _build_ignore_polys_px(cfg_obj, w_px: int, h_px: int):
        zones = getattr(cfg_obj, "person_ignore_zone", []) or []
        polys = []
        for poly in zones:
            try:
                if not isinstance(poly, (list, tuple)) or len(poly) < 3:
                    continue
                pts = []
                for p in poly:
                    if not isinstance(p, (list, tuple)) or len(p) != 2:
                        continue
                    x = int(float(p[0]) * float(w_px))
                    y = int(float(p[1]) * float(h_px))
                    pts.append([x, y])
                if len(pts) >= 3:
                    polys.append(np.array(pts, dtype=np.int32))
            except Exception:
                continue
        return polys

    def _point_in_any(polys, x: float, y: float) -> bool:
        """Ritorna True se il punto (x,y) è dentro uno qualsiasi dei poligoni (lista di np.array)."""
        if not polys:
            return False
        try:
            for poly in polys:
                if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                    return True
        except Exception:
            return False
        return False

    def _apply_person_filters(dets, w_px: int, h_px: int, cfg_obj):
        # dets: [((x,y,w,h), score), ...]
        min_area = int(getattr(cfg_obj, "person_min_box_area", 0) or 0)
        polys = _build_ignore_polys_px(cfg_obj, w_px, h_px)
        if min_area <= 0 and not polys:
            return dets
        out = []
        for (bx, by, bw, bh), sc in dets:
            try:
                if min_area > 0 and (float(bw) * float(bh)) < float(min_area):
                    continue
                if polys:
                    cx = float(bx) + float(bw) * 0.5
                    cy = float(by) + float(bh) * 0.5
                    ignore = False
                    for poly in polys:
                        if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                            ignore = True
                            break
                    if ignore:
                        continue
                out.append(((float(bx), float(by), float(bw), float(bh)), float(sc)))
            except Exception:
                out.append(((float(bx), float(by), float(bw), float(bh)), float(sc)))
        return out

    # Classifier throttle per track
    cls_interval_ms = int(getattr(cfg, "cls_interval_ms", 300))
    last_cls_ts_per_track = {}
    cls_cache_per_track = {}

    # Stato riconnessioni RTSP
    rtsp_fail_count = 0
    rtsp_max_fail = int(getattr(cfg, "rtsp_max_failures", 60))
    rtsp_reconnect = float(getattr(cfg, "rtsp_reconnect_sec", 2.0))
    is_rtsp_mode = isinstance(getattr(cfg, "camera", 0), str) and str(getattr(cfg, "camera")).lower().startswith("rtsp://")

    try:
        while True:
            ok, frame = cap.read()
            now = time.time()

            # Avanza le finestre metriche anche senza eventi
            try:
                aggregator.tick(now)
            except Exception:
                pass

            if not ok or frame is None:
                state.update(camera_ok=False, fps=0.0, width=0, height=0)
                if is_rtsp_mode:
                    rtsp_fail_count += 1
                    if rtsp_fail_count >= rtsp_max_fail:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        try:
                            log_event("RTSP_RECONNECT", url=str(getattr(cfg, "camera", "")), after_fail=rtsp_fail_count)
                        except Exception:
                            pass
                        time.sleep(rtsp_reconnect)
                        cap = open_rtsp(getattr(cfg, "camera", ""), cfg)
                        rtsp_fail_count = 0
                    else:
                        time.sleep(0.05)
                else:
                    # webcam locale: logga un warning sporadico
                    if frame_count % 30 == 0:
                        try:
                            log_event("CAMERA_FRAME_MISS")
                        except Exception:
                            pass
                    time.sleep(0.05)
                continue
            else:
                if is_rtsp_mode:
                    rtsp_fail_count = 0

            frame_count += 1

            # Throttle cattura
            elapsed = now - last_ts
            if elapsed < delay:
                time.sleep(delay - elapsed)
            last_ts = time.time()

            # Report ~1s
            if (now - last_report) >= 1.0:
                h, w = frame.shape[:2]
                fps = frame_count / (now - last_report)
                print(f"[health] fps={fps:.1f} | size={w}x{h} | camera=OK | tracks={len(tracker.tracks)} | cls={'ON' if classifier.enabled else 'OFF'}")
                try:
                    log_event("HEALTH", fps=round(fps, 2), size=[w, h], tracks=len(tracker.tracks), cls=bool(getattr(classifier, 'enabled', False)))
                except Exception:
                    pass
                state.update(camera_ok=True, fps=fps, width=w, height=h, frames_inc=frame_count)
                frame_count = 0
                last_report = now

            # --- PREPARA FRAME DI LAVORO (sempre, non solo in debug) ---
            proc_w = int(getattr(cfg, "proc_resize_width", 0) or 0)
            proc = resize_keep_aspect(frame, proc_w) if (proc_w > 0 and proc_w < frame.shape[1]) else frame
            ph, pw = proc.shape[:2]

            # Detection persone
            person_dets = []
            if person_detector is not None:
                try:
                    person_dets = person_detector.detect(proc)  # [((x,y,w,h),score), ...]
                except Exception:
                    person_dets = []

            # Filtri post-process (area minima, ignore zones)
            try:
                person_dets = _apply_person_filters(person_dets, pw, ph, cfg)
            except Exception:
                pass
            
            # Detection volto
            face_dets = []
            if face_detector is not None:
                try:
                    det_w = int(getattr(cfg, "detector_resize_width", 0) or 0)
                    if det_w > 0 and det_w < pw:
                        det_img = resize_keep_aspect(proc, det_w)
                        s = det_img.shape[1] / float(pw)
                    else:
                        det_img = proc
                        s = 1.0
                    dets_ori = face_detector.detect(det_img)
                    if s != 1.0:
                        face_dets = [((float(x)/s, float(y)/s, float(w)/s, float(h)/s), float(score)) for (x, y, w, h), score in dets_ori]
                    else:
                        face_dets = [((float(x), float(y), float(w), float(h)), float(score)) for (x, y, w, h), score in dets_ori]
                except Exception:
                    face_dets = []

            # Log periodico conteggio detection (opzionale)
            try:
                dbg_mod = int(max(1, float(getattr(cfg, "debug_stream_fps", 5))))
            except Exception:
                dbg_mod = 5
            if dbg_mod > 0 and (frame_count % dbg_mod) == 0:
                print(f"[dbg] person_dets={len(person_dets)} face_dets={len(face_dets)}")
                try:
                    log_event("DETECT", persons=len(person_dets), faces=len(face_dets))
                except Exception:
                    pass

            # Tracking
            track_src = "person" if person_dets else "face"
            detections = [[bx, by, bw, bh, sc] for (bx, by, bw, bh), sc in (person_dets if person_dets else face_dets)]
            try:
                log_event("TRACK_INPUT", src=track_src, det=len(detections))
            except Exception:
                pass

            tracks = tracker.update(detections)
            try:
                log_event("TRACK", active=len(tracks))
            except Exception:
                pass

            # Associazione volti e ReID
            face_assoc = associate_faces_to_tracks(
                face_dets, tracks,
                iou_th=float(getattr(cfg, "face_assoc_iou_th", 0.20)),
                use_center_in=bool(getattr(cfg, "face_assoc_center_in", True))
            )

            # Assegna/ReID
            for t in tracks:
                tid = t["track_id"]
                tstate = tracker.tracks.get(tid, {})
                x, y, w, h = map(int, t["bbox"])
                person_crop = proc[max(0, y): y + h, max(0, x): x + w]
                matched_face = face_assoc.get(tid)

                if "global_id" not in tstate:
                    if matched_face is not None:
                        fx, fy, fw, fh = map(int, matched_face)
                        face_crop = proc[max(0, fy): fy + fh, max(0, fx): fx + fw]
                    else:
                        face_crop = None
                        
                    body_crop_to_pass = person_crop if track_src == "person" else None
                    try:
                        gid = reid.assign_global_id(
                            face_bgr_crop=face_crop,
                            kps5=None,
                            body_bgr_crop=body_crop_to_pass
                        )
                    except TypeError:
                        # fallback: chiamate legacy
                        try:
                            gid = reid.assign_global_id(face_bgr_crop=face_crop, kps5=None)
                        except TypeError:
                            gid = reid.assign_global_id(face_crop, None)
                    gid = reid.canon(gid)
                    tstate["global_id"] = gid
                    tstate["assigned_with_face"] = (matched_face is not None)
                    try:
                        log_event("REID_ASSIGN", tid=int(tid), gid=int(gid), withFace=bool(matched_face is not None))
                    except Exception:
                        pass

                # salva anche il face bbox corrente (per classificazione)
                tstate["face_bbox"] = matched_face  # None o (x,y,w,h)
                tracker.tracks[tid] = tstate

            # --- BUILD SNAPSHOTS FOR DEBUG ---
            # 1) Active tracks
            active = []
            for t in tracks:
                tid = t["track_id"]
                tstate = tracker.tracks.get(tid, {})
                gid = tstate.get("global_id", tid)
                x, y, w, h = map(int, t["bbox"])
                active.append({'gid': int(gid), 'tid': int(tid), 'bbox': [x, y, w, h]})
            state.set_active_tracks(active)

            # 2) ReID memory snapshot
            try:
                mem_items = []
                now_s = time.time()
                for pid, info in getattr(reid, 'mem', {}).items():
                    canon   = reid.canon(pid) if hasattr(reid, 'canon') else pid
                    last    = float(info.get('last', 0.0))
                    created = float(info.get('created', last))
                    feats   = list(info.get('feats', []) or [])
                    body    = list(info.get('body', []) or [])
                    meta    = info.get('meta', {}) or {}
                    gh      = meta.get('gender_hist', {}) or {}
                    ah      = meta.get('age_hist', {}) or {}
                    genderMajor = max(gh.items(), key=lambda kv: kv[1])[0] if gh else "unknown"
                    ageMajor    = max(ah.items(), key=lambda kv: kv[1])[0] if ah else "unknown"
                    ttl = float(getattr(reid, 'ttl', 600.0) or 0.0)
                    ttlRem = max(0.0, ttl - (now_s - last)) if ttl > 0 else None

                    mem_items.append({
                        'id': int(canon),
                        'rawId': int(pid),
                        'hits': int(info.get('hits', 1)),
                        'last': last,
                        'created': created,
                        'ageSec': max(0, now_s - created),
                        'hasFace': len(feats) > 0,
                        'hasSilhouette': len(body) > 0,
                        'faceCount': len(feats),
                        'silCount': len(body),
                        'genderMajor': genderMajor,
                        'ageMajor': ageMajor,
                        'genderHist': gh,
                        'ageHist': ah,
                        'ttlRemSec': ttlRem,
                    })

                mem_items.sort(key=lambda r: r['id'])
                state.set_reid_debug(mem_items)
            except Exception:
                pass

            # Classificazione (cache ogni cls_interval_ms, solo se il volto è abbastanza grande)
            for t in tracks:
                tid = t["track_id"]
                tstate = tracker.tracks.get(tid, {})
                fb = tstate.get("face_bbox", None)
                if fb is None:
                    cached = cls_cache_per_track.get(tid, {"gender": "unknown", "ageBucket": "unknown", "confidence": 0.0})
                    tracker.apply_labels(tid, cached.get("gender", "unknown"), cached.get("ageBucket", "unknown"), cached.get("confidence", 0.0))
                    continue

                fx, fy, fw, fh = map(int, fb)
                if min(fw, fh) >= int(getattr(cfg, "cls_min_face_px", 64)):
                    last_ts_t = last_cls_ts_per_track.get(tid, 0.0)
                    if (now - last_ts_t) * 1000.0 >= cls_interval_ms:
                        face_roi = proc[max(0, fy): fy + fh, max(0, fx): fx + fw]
                        res = classifier.infer(face_roi)
                        # Se l'ID era stato assegnato senza volto, prova a correggere con volto+corpo
                        try:
                            if not tstate.get("assigned_with_face", False):
                                try:
                                    body_crop2 = (
                                        proc[max(0, t["bbox"][1]): t["bbox"][1] + t["bbox"][3],
                                             max(0, t["bbox"][0]): t["bbox"][0] + t["bbox"][2]]
                                    ) if track_src == "person" else None
                                    new_gid = reid.assign_global_id(
                                        face_bgr_crop=face_roi,
                                        kps5=None,
                                        body_bgr_crop=body_crop2,
                                    )
                                except TypeError:
                                    new_gid = reid.assign_global_id(face_bgr_crop=face_roi, kps5=None)
                                new_gid = reid.canon(new_gid)
                                if new_gid != tstate.get("global_id", tid):
                                    tstate["global_id"] = new_gid
                                    try:
                                        log_event("REID_CORRECT", tid=int(tid), new_gid=int(new_gid))
                                    except Exception:
                                        pass
                                tstate["assigned_with_face"] = True
                                tracker.tracks[tid] = tstate
                        except Exception:
                            pass
                        cls_cache_per_track[tid] = res
                        last_cls_ts_per_track[tid] = now
                        # aggiorna meta del global id
                        try:
                            gid = reid.canon(tstate.get("global_id", tid))
                            if hasattr(reid, "update_meta"):
                                reid.update_meta(gid, res.get("gender","unknown"), res.get("ageBucket","unknown"))
                        except Exception:
                            pass
                cached = cls_cache_per_track.get(tid, {"gender": "unknown", "ageBucket": "unknown", "confidence": 0.0})
                tracker.apply_labels(tid, cached.get("gender", "unknown"), cached.get("ageBucket", "unknown"), cached.get("confidence", 0.0))

            # Tripwire crossing (solo se count_mode == "tripwire")
            if count_mode == "tripwire" and isinstance(roi_tripwire, (list, tuple)) and len(roi_tripwire) == 2:
                for t in tracks:
                    tid = t["track_id"]
                    tb = t["bbox"]
                    curr_c = (tb[0] + tb[2] * 0.5, tb[1] + tb[3] * 0.5)
                    prev_c = tracker.tracks.get(tid, {}).get("prev_center", None)
                    if prev_c is not None:
                        dir_tag = MinuteAggregator.check_crossing(
                            frame_w=pw,
                            frame_h=ph,
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
                            gid = tr_state.get("global_id", tid)
                            gid = reid.canon(gid)
                            now_ts = now
                            if should_count(gid, now_ts):
                                aggregator.add_cross_event(
                                    gender=gender,
                                    age_bucket=age_bucket,
                                    direction_tag=dir_tag,
                                    track_id=gid,
                                    now=now_ts,
                                )
                                try:
                                    log_event("TRIPWIRE_CROSS", gid=int(gid), dir=dir_tag, gender=gender, age=age_bucket)
                                except Exception:
                                    pass
                            try:
                                if hasattr(reid, "touch"):
                                    reid.touch(gid, now_ts)
                            except Exception:
                                pass
                        # aggiorna prev_center dopo il check
                        tracker.update_prev_center(tid)

            # --- OVERLAY E STREAM SOLO SE DEBUG ON E AD INTERVALLO ---
            vis = proc.copy()

            if count_mode == "tripwire" and isinstance(roi_tripwire, (list, tuple)) and len(roi_tripwire) == 2:
                draw_tripwire(vis, (tuple(roi_tripwire[0]), tuple(roi_tripwire[1])), roi_band_px, (255, 0, 0))

            # opzionale: non mostrare in debug i track dentro le ignore-zone
            debug_hide_ignored = bool(getattr(cfg, "debug_hide_ignored", True))
            debug_mark_centers = bool(getattr(cfg, "debug_mark_centers", False))
            debug_show_ignore_rects = bool(getattr(cfg, "debug_show_ignore_rects", True))
            dbg_polys = _build_ignore_polys_px(cfg, pw, ph)
            if dbg_polys and debug_show_ignore_rects:
                try:
                    overlay = vis.copy()
                    # riempimento semi-trasparente rosso
                    cv2.fillPoly(overlay, dbg_polys, (0, 0, 255))
                    vis = cv2.addWeighted(overlay, 0.12, vis, 0.88, 0)
                    # contorno rosso più sottile
                    cv2.polylines(vis, dbg_polys, isClosed=True, color=(0, 0, 255), thickness=1)
                    # etichetta al baricentro del primo poligono
                    M = cv2.moments(dbg_polys[0])
                    if M["m00"] != 0:
                        cx_lbl = int(M["m10"] / M["m00"])
                        cy_lbl = int(M["m01"] / M["m00"])
                    else:
                        cx_lbl, cy_lbl = int(dbg_polys[0][0][0]), int(dbg_polys[0][0][1])
                    label = "person_ignore1"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    x1 = max(0, cx_lbl - tw // 2 - 4)
                    y1 = max(0, cy_lbl - th // 2 - 4)
                    cv2.rectangle(vis, (x1, y1), (x1 + tw + 8, y1 + th + 8), (0, 0, 0), -1)  # sfondo nero sempre
                    cv2.putText(vis, label, (x1 + 4, y1 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                except Exception:
                    pass


            for t in tracks:
                tid = t["track_id"]
                tstate = tracker.tracks.get(tid, {})
                gid = tstate.get("global_id", tid)

                # centro del box per debug/ignore
                cx = float(t["bbox"][0]) + float(t["bbox"][2]) * 0.5
                cy = float(t["bbox"][1]) + float(t["bbox"][3]) * 0.5
                inside_ignore = False
                if dbg_polys:
                    inside_ignore = _point_in_any(dbg_polys, cx, cy)

                # marker del centro: rosso se inside_ignore, grigio altrimenti
                if debug_mark_centers:
                    cv2.circle(vis, (int(cx), int(cy)), 2, (0, 0, 255) if inside_ignore else (180, 180, 180), -1)

                # se richiesto, non disegnare track il cui centro è dentro l'ignore
                if debug_hide_ignored and inside_ignore:
                    continue

                # face (lime) se presente
                fb = tstate.get("face_bbox", None)
                if fb is not None:
                    fx, fy, fw, fh = map(int, fb)
                    cv2.rectangle(vis, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 1)

                g = t.get("gender", "unknown")
                a = t.get("ageBucket", "unknown")
                c = t.get("conf", 0.0)
                lbl = f"#G{gid} {g[:1].upper() if g!='unknown' else '?'} / {a if a!='unknown' else '--'} ({c:.2f})"
                # sfondo nero sempre (draw_box_with_label già usa uno sfondo nero pieno)
                draw_box_with_label(vis, t["bbox"], lbl, (255, 255, 0))


            if getattr(cfg, 'debug_enabled', True) and (now - last_stream_t) >= stream_interval:
                ok_jpg, buf = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok_jpg:
                    state.set_debug_jpeg(buf.tobytes())
                last_stream_t = now
                try:
                    log_event('FRAME_OUT', jpeg=len(buf) if ok_jpg else 0)
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\n[EXIT] Interrotto dall'utente.")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
