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
from typing import Optional
import cv2
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
    # Opzioni RTSP
    transport = getattr(cfg, "rtsp_transport", "tcp")
    url_eff = url
    if url.lower().startswith("rtsp://") and transport and "rtsp_transport=" not in url:
        sep = "&" if "?" in url else "?"
        url_eff = f"{url}{sep}rtsp_transport={transport}"

    cap = cv2.VideoCapture(url_eff, cv2.CAP_FFMPEG)

    # Timeout (best-effort: proprietà disponibili solo su alcune build)
    try:
        ot = float(getattr(cfg, "rtsp_open_timeout_ms", 4000))
        rt = float(getattr(cfg, "rtsp_read_timeout_ms", 4000))
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, ot)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, rt)
    except Exception:
        pass

    # Buffer interno (in frame; non tutte le build lo rispettano)
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

    # --- Re-Identification (memoria facce/aspetto) + dedup conteggio ---
    try:
        reid_enabled = bool(getattr(cfg, "reid_enabled", True))
        if reid_enabled:
            reid = FaceReID(
                model_path=getattr(cfg, "reid_model_path", "models/face_recognition_sface_2021dec.onnx"),
                similarity_th=float(getattr(cfg, "reid_similarity_th", 0.365)),
                cache_size=int(getattr(cfg, "reid_cache_size", 1000)),
                memory_ttl_sec=int(getattr(cfg, "reid_memory_ttl_sec", 600)),
                bank_size=int(getattr(cfg, "reid_bank_size", 10)),
                merge_sim=float(getattr(cfg, "reid_merge_sim", 0.55)),
                prefer_oldest=bool(getattr(cfg, "reid_prefer_oldest", True)),
            )
        else:
            reid = FaceReID("", 1.0, 1, 1)  # istanza "spenta" safe
        print(f"[OK] ReID enabled={getattr(reid,'enabled',False)}")
    except Exception as e:
        print(f"[WARN] ReID init failed: {e}")
        reid = FaceReID("", 1.0, 1, 1)

    # Parametri appearance (se il metodo non esiste ancora, salta)
    if hasattr(reid, "set_appearance_params"):
        try:
            reid.set_appearance_params(
                bins=int(getattr(cfg, "appearance_hist_bins", 24)),
                min_area_px=int(getattr(cfg, "appearance_min_area_px", 900)),
                weight=float(getattr(cfg, "appearance_weight", 0.35)),
                app_th=float(getattr(cfg, "reid_app_similarity_th", 0.82)),
            )
            # NEW: estendi policy con body_only_th / allow_body_seed in modo retro-compatibile
            try:
                reid.set_id_policy(
                    appearance_weight=float(getattr(cfg, "appearance_weight", 0.35)),
                    app_only_min_th=float(getattr(cfg, "reid_app_only_th", 0.65)),
                    require_face_if_available=bool(getattr(cfg, "reid_require_face_if_available", True)),
                    face_gate=float(getattr(cfg, "reid_face_gate", max(getattr(cfg, "reid_similarity_th", 0.35), 0.42))),
                    body_only_th=float(getattr(cfg, "body_only_th", 0.80)),
                    allow_body_seed=bool(getattr(cfg, "reid_allow_body_seed", True)),
                )
            except TypeError:
                reid.set_id_policy(
                    appearance_weight=float(getattr(cfg, "appearance_weight", 0.35)),
                    app_only_min_th=float(getattr(cfg, "reid_app_only_th", 0.65)),
                    require_face_if_available=bool(getattr(cfg, "reid_require_face_if_available", True)),
                    face_gate=float(getattr(cfg, "reid_face_gate", max(getattr(cfg, "reid_similarity_th", 0.35), 0.42))),
                )
            # NEW: verbose debug decisioni ReID (se supportato)
            if hasattr(reid, "set_debug"):
                reid.set_debug(bool(getattr(cfg, "debug_reid_verbose", False)))
        except Exception:
            pass
    
    # NEW: Backend Body ReID opzionale (OSNet/Intel OMZ) – safe se non configurato o non disponibile
    try:
        body_model = getattr(cfg, "body_reid_model_path", "")
        if BodyReID is not None and isinstance(body_model, str) and len(body_model.strip()) > 0:
            body_backend = BodyReID(
                model_path=body_model,
                backend_id=int(getattr(cfg, "body_reid_backend", 0)),
                target_id=int(getattr(cfg, "body_reid_target", 0)),
                input_size=(
                    int(getattr(cfg, "body_reid_input_w", 128)),
                    int(getattr(cfg, "body_reid_input_h", 256)),
                ),
            )
            if hasattr(reid, "set_body_backend"):
                reid.set_body_backend(body_backend)
            print("[OK] Body ReID backend caricato.")
        else:
            print("[INFO] Body ReID non configurato.")
    except Exception as e:
        print(f"[WARN] Body ReID init failed: {e}")

    # Counting mode & eviction callback (fuori dal try di init reid)
    count_mode = str(getattr(cfg, "count_mode", "presence")).lower()
    presence_ttl = int(getattr(cfg, "presence_ttl_sec", getattr(cfg, "reid_memory_ttl_sec", 600)))
    reid_ttl     = int(getattr(cfg, "reid_memory_ttl_sec", 600))

    try:
        reid.ttl = presence_ttl if count_mode == "presence" else reid_ttl
    except Exception:
        pass

    def _on_evict(gid: int, meta: dict, last_ts: float):
        if count_mode != "presence":
            return
        # scegli il genere prevalente e l'età prevalente se disponibili
        try:
            gh = meta.get('gender_hist', {}) or {}
            ah = meta.get('age_hist', {}) or {}
            gender = max(gh.items(), key=lambda kv: kv[1])[0] if gh else "unknown"
            age_bucket = max(ah.items(), key=lambda kv: kv[1])[0] if ah else "unknown"
        except Exception:
            gender, age_bucket = "unknown", "unknown"
        aggregator.add_presence_event(
            gender=gender,
            age_bucket=age_bucket,
            global_id=gid,
            now=last_ts
        )

    # collega la callback solo se l'attributo esiste
    if hasattr(reid, "on_evict"):
        try:
            reid.on_evict = _on_evict
        except Exception:
            pass

    # mappa per dedup dei conteggi: global_id -> last_count_ts (usata in modalità tripwire)
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

    # Detector persone (primary)
    person_det = None
    try:
        p_model = getattr(cfg, "person_model_path", "")
        if isinstance(p_model, str) and len(p_model.strip()) > 0:
            person_det = PersonDetector(
                model_path=p_model,
                img_size=int(getattr(cfg, "person_img_size", 640)),
                score_th=float(getattr(cfg, "person_score_th", 0.35)),
                iou_th=float(getattr(cfg, "person_iou_th", 0.45)),
                max_det=int(getattr(cfg, "person_max_det", 200)),
                backend_id=int(getattr(cfg, "person_backend", 0)),
                target_id=int(getattr(cfg, "person_target", 0)),
            )
            print("[OK] Person detector caricato.")
        else:
            print("[INFO] Person detector non configurato (fallback: tracking su volto).")
    except Exception as e:
        print(f"[WARN] Person detector init failed: {e}")
        person_det = None

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

            # Avanza le finestre metriche anche senza eventi
            try:
                aggregator.tick(now)
            except Exception:
                pass

            # (ri)leggi frame (alcune webcam/rtsp beneficiano del doppio read)
            ok, frame = cap.read()
            now = time.time()

            # init contatori fallimenti (una sola volta)
            if 'rtsp_fail_count' not in locals():
                rtsp_fail_count = 0
                rtsp_max_fail   = int(getattr(cfg, "rtsp_max_failures", 60))  # ~60*50ms = 3s
                rtsp_reconnect  = float(getattr(cfg, "rtsp_reconnect_sec", 2.0))

            if not ok or frame is None:
                state.update(camera_ok=False, fps=0.0, width=0, height=0)

                if isinstance(getattr(cfg, "camera", 0), str) and str(getattr(cfg, "camera")).lower().startswith("rtsp://"):
                    rtsp_fail_count += 1
                    if rtsp_fail_count >= rtsp_max_fail:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        time.sleep(rtsp_reconnect)
                        cap = open_rtsp(cam_cfg, cfg)
                        rtsp_fail_count = 0
                    else:
                        time.sleep(0.05)
                else:
                    time.sleep(0.05)
                continue
            else:
                # reset fallimenti su frame valido
                if isinstance(getattr(cfg, "camera", 0), str) and str(getattr(cfg, "camera")).lower().startswith("rtsp://"):
                    rtsp_fail_count = 0

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
                print(f"[health] fps={fps:.1f} | size={w}x{h} | camera=OK | tracks={len(tracker.tracks)} | cls={'ON' if classifier.enabled else 'OFF'}")
                state.update(camera_ok=True, fps=fps, width=w, height=h, frames_inc=frame_count)
                frame_count = 0
                last_report = now

            # --- Debug stream + pipeline ---
            if getattr(cfg, "debug_enabled", True) and (now - last_stream_t) >= stream_interval:
                # Usiamo 'vis' per detection e overlay (come nel main originale)
                vis = resize_keep_aspect(frame, getattr(cfg, "debug_resize_width", 960))
                vh, vw = vis.shape[:2]

                # Detection persone (primary per tracking)
                person_dets = []
                if person_det is not None:
                    try:
                        person_dets = person_det.detect(vis)  # [((x,y,w,h),score), ...]
                    except Exception:
                        person_dets = []

                # Detection volto (serve per età/genere e per arricchire ReID)
                face_dets = []
                if detector is not None:
                    try:
                        # Larghezza dedicata alla face detection (se 0 o >= vw, usa 'vis' senza ridurre)
                        det_w = int(getattr(cfg, "detector_resize_width", 0) or 0)
                        if det_w > 0 and det_w < vw:
                            det_img = resize_keep_aspect(vis, det_w)
                            s = det_img.shape[1] / float(vw)  # rapporto larghezza det_img/vis
                        else:
                            det_img = vis
                            s = 1.0

                        # YuNet sulla versione ridotta
                        dets_ori = detector.detect(det_img)  # [((x,y,w,h), score), ...]

                        # Rimappa le bbox al sistema di coordinate di 'vis'
                        if s != 1.0:
                            face_dets = [
                                ((float(x)/s, float(y)/s, float(w)/s, float(h)/s), float(score))
                                for (x, y, w, h), score in dets_ori
                            ]
                        else:
                            face_dets = [
                                ((float(x), float(y), float(w), float(h)), float(score))
                                for (x, y, w, h), score in dets_ori
                            ]
                    except Exception:
                        face_dets = []

                # subito dopo aver settato person_dets e face_dets
                if frame_count % int(max(1, 1*float(getattr(cfg, "debug_stream_fps", 5)))) == 0:
                    print(f"[dbg] person_dets={len(person_dets)} face_dets={len(face_dets)}")

                # Scegli input del tracker: persone se disponibili, altrimenti fallback ai volti
                if person_dets:
                    detections = [[bx, by, bw, bh, sc] for (bx, by, bw, bh), sc in person_dets]
                else:
                    detections = [[bx, by, bw, bh, sc] for (bx, by, bw, bh), sc in face_dets]

                # Tracking (su persone o volti)
                tracks = tracker.update(detections)

                # Associa volti ai track persona
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
                    person_crop = vis[max(0, y): y + h, max(0, x): x + w]
                    matched_face = face_assoc.get(tid)

                    if "global_id" not in tstate:
                        if matched_face is not None:
                            fx, fy, fw, fh = map(int, matched_face)
                            face_crop = vis[max(0, fy): fy + fh, max(0, fx): fx + fw]
                        else:
                            face_crop = None
                        # Usa persona come 'appearance', volto (se c'è) come face
                        try:
                            gid = reid.assign_global_id(
                                face_bgr_crop=face_crop,
                                kps5=None,
                                appearance_bgr_crop=person_crop
                            )
                        except TypeError:
                            try:
                                gid = reid.assign_global_id(
                                    face_bgr_crop=face_crop,
                                    kps5=None,
                                    appearance_bgr_crop=person_crop,
                                )
                            except TypeError:
                                gid = reid.assign_global_id(face_crop, None)
                        gid = reid.canon(gid)
                        tstate["global_id"] = gid
                        tstate["assigned_with_face"] = (matched_face is not None)

                    # salva anche il face bbox corrente (per classificazione)
                    tstate["face_bbox"] = matched_face  # None o (x,y,w,h)
                    tracker.tracks[tid] = tstate

                # --- BUILD SNAPSHOTS FOR DEBUG ---
                # 1) Active tracks (per evidenziare in pagina gli ID visibili ora)
                active = []
                for t in tracks:
                    tid = t["track_id"]
                    tstate = tracker.tracks.get(tid, {})
                    gid = tstate.get("global_id", tid)
                    x, y, w, h = map(int, t["bbox"])
                    active.append({'gid': int(gid), 'tid': int(tid), 'bbox': [x, y, w, h]})
                state.set_active_tracks(active)   # <-- passa la LISTA, non una mappa

                # 2) ReID memory snapshot (lista ordinata per id, con meta estesi)
                try:
                    mem_items = []
                    now_s = time.time()
                    for pid, info in getattr(reid, 'mem', {}).items():
                        canon   = reid.canon(pid) if hasattr(reid, 'canon') else pid
                        last    = float(info.get('last', 0.0))
                        created = float(info.get('created', last))
                        feats   = list(info.get('feats', []) or [])
                        apps    = list(info.get('app', []) or [])
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
                            'hasSilhouette': len(apps) > 0,
                            'faceCount': len(feats),
                            'silCount': len(apps),
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

                # Classificazione (cache ogni cls_interval_ms, solo se volto abbastanza grande)
                for t in tracks:
                    tid = t["track_id"]
                    tstate = tracker.tracks.get(tid, {})
                    fb = tstate.get("face_bbox", None)
                    if fb is None:
                        # niente volto associato → mantieni le label correnti (unknown)
                        cached = cls_cache_per_track.get(tid, {"gender": "unknown", "ageBucket": "unknown", "confidence": 0.0})
                        tracker.apply_labels(tid, cached.get("gender", "unknown"), cached.get("ageBucket", "unknown"), cached.get("confidence", 0.0))
                        continue

                    fx, fy, fw, fh = map(int, fb)
                    if min(fw, fh) >= int(getattr(cfg, "cls_min_face_px", 64)):
                        last_ts_t = last_cls_ts_per_track.get(tid, 0.0)
                        if (now - last_ts_t) * 1000.0 >= cls_interval_ms:
                            face_roi = vis[max(0, fy): fy + fh, max(0, fx): fx + fw]
                            res = classifier.infer(face_roi)
                            # Se l'ID era stato assegnato senza volto, ora che il volto c'è riconsidera l'assegnazione
                            try:
                                if not tstate.get("assigned_with_face", False):
                                    # prova a riassegnare usando volto + persona
                                    try:
                                        new_gid = reid.assign_global_id(
                                            face_bgr_crop=face_roi,
                                            kps5=None,
                                            body_bgr_crop=vis[max(0, t["bbox"][1]): t["bbox"][1] + t["bbox"][3],
                                                              max(0, t["bbox"][0]): t["bbox"][0] + t["bbox"][2]],
                                        )
                                    except TypeError:
                                        new_gid = reid.assign_global_id(
                                            face_bgr_crop=face_roi,
                                            kps5=None,
                                            appearance_bgr_crop=vis[max(0, t["bbox"][1]): t["bbox"][1] + t["bbox"][3],
                                                                    max(0, t["bbox"][0]): t["bbox"][0] + t["bbox"][2]],
                                        )
                                    new_gid = reid.canon(new_gid)
                                    if new_gid != tstate.get("global_id", tid):
                                        tstate["global_id"] = new_gid
                                    tstate["assigned_with_face"] = True
                                    tracker.tracks[tid] = tstate
                            except Exception:
                                pass
                            cls_cache_per_track[tid] = res
                            last_cls_ts_per_track[tid] = now
                            # aggiorna meta del global id (se disponibile)
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
                                gid = tr_state.get("global_id", tid)
                                gid = reid.canon(gid)
                                now_ts = now
                                # dedup: non riconteggiare lo stesso global_id entro TTL
                                if should_count(gid, now_ts):
                                    aggregator.add_cross_event(
                                        gender=gender,
                                        age_bucket=age_bucket,
                                        direction_tag=dir_tag,
                                        track_id=gid,
                                        now=now_ts,
                                    )
                                # mantieni "viva" la memoria reid (touch se disponibile)
                                try:
                                    if hasattr(reid, "touch"):
                                        reid.touch(gid, now_ts)
                                except Exception:
                                    pass
                        # aggiorna prev_center dopo il check
                        tracker.update_prev_center(tid)

                if count_mode == "tripwire" and isinstance(roi_tripwire, (list, tuple)) and len(roi_tripwire) == 2:
                  draw_tripwire(vis, (tuple(roi_tripwire[0]), tuple(roi_tripwire[1])), roi_band_px, (255,0,0))

                for t in tracks:
                    tid = t["track_id"]
                    tstate = tracker.tracks.get(tid, {})
                    gid = tstate.get("global_id", tid)

                    # box persona (cyan)
                    px, py, pw, ph = map(int, t["bbox"])
                    cv2.rectangle(vis, (px, py), (px+pw, py+ph), (255, 255, 0), 1)

                    # face (lime) se presente
                    fb = tstate.get("face_bbox", None)
                    if fb is not None:
                        fx, fy, fw, fh = map(int, fb)
                        cv2.rectangle(vis, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 1)

                    g = t.get("gender", "unknown")
                    a = t.get("ageBucket", "unknown")
                    c = t.get("conf", 0.0)
                    lbl = f"#G{gid} {g[:1].upper() if g!='unknown' else '?'} / {a if a!='unknown' else '--'} ({c:.2f})"
                    draw_box_with_label(vis, t["bbox"], lbl, (255, 255, 0))

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
