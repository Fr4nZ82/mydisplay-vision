# MyDisplay Vision – Configurazione (guida completa)

Questo documento elenca e spiega tutte le proprietà utilizzabili in config.json.  

- Indice
  - [🎥 Capture / Stream](#-capture--stream)
  - [🌐 API](#-api)
  - [📡 RTSP (sorgenti IP)](#-rtsp-sorgenti-ip)
  - [🧭 Tracker (ID stabili)](#-tracker-id-stabili)
  - [🔎 Detector](#-detector)
    - [Person detector (YOLO ONNX)](#person-detector-yolo-onnx)
    - [Face detector (YuNet)](#face-detector-yunet)
    - [Associazione volto→persona](#associazione-voltopersona)
  - [🧠 Classificatore Età/Genere](#-classificatore-etàgenere)
    - [Modello combinato (consigliato)](#modello-combinato-consigliato)
    - [Modelli separati](#modelli-separati)
    - [Throttle / caching classificazione](#throttle--caching-classificazione)
  - [🚶 ROI / Tripwire](#-roi--tripwire)
  - [🔁 Re-Identification (ReID)](#-re-identification-reid)
    - [Obiettivi e panoramica](#obiettivi-e-panoramica)
    - [Face ReID (SFace/ArcFace)](#face-reid-sfacearcface)
    - [Body ReID (OSNet / Intel OMZ)](#body-reid-osnet--intel-omz)
    - [Firma di aspetto legacy (colore vestiti)](#firma-di-aspetto-legacy-colore-vestiti)
    - [Politiche di fusione e soglie](#politiche-di-fusione-e-soglie)
    - [Memoria, TTL e banca di feature](#memoria-ttl-e-banca-di-feature)
    - [Diagnostica ReID](#diagnostica-reid)
  - [🧮 Modalità di conteggio e deduplica](#-modalità-di-conteggio-e-deduplica)
  - [📊 Metriche / Aggregazione](#-metriche--aggregazione)
  - [⚙️ Suggerimenti di performance](#️-suggerimenti-di-performance)
  - [🧪 Troubleshooting](#-troubleshooting)


## 🎥 Capture / Stream

Ambito: impostazioni di acquisizione e stream di debug (dimensioni, fps, overlay).

- camera (default: 0): indice webcam o stringa RTSP.
- width (default: 1920), height (default: 1080): risoluzione richiesta (px).
- target_fps (default: 10.0): FPS desiderati per il loop (throttle).
- debug_enabled (default: true): abilita /debug e stream MJPEG.
- debug_stream_fps (default: 5): FPS dello stream MJPEG.
- debug_resize_width (default: 960): larghezza frame per /debug; 0=nessun resize.


## 🌐 API

Ambito: server HTTP locale per diagnostica, stream e metriche.

- api_host (default: "127.0.0.1"): bind address (usa "0.0.0.0" per LAN).
- api_port (default: 8080): porta del server.


## 📡 RTSP (sorgenti IP)

Ambito: tuning best‑effort per flussi RTSP con OpenCV/FFmpeg.

- rtsp_transport (default: "tcp"): "tcp" o "udp".
- rtsp_buffer_frames (default: 2): dimensione buffer interno (frame).
- rtsp_open_timeout_ms (default: 4000): timeout apertura (ms).
- rtsp_read_timeout_ms (default: 4000): timeout lettura (ms).
- rtsp_reconnect_sec (default: 2.0): attesa prima del reopen.
- rtsp_max_failures (default: 60): read fallite prima di riaprire.


## 🧭 Tracker (ID stabili)

Ambito: mantenere un ID coerente per persona/volto tra frame consecutivi (SORT‑lite).

- tracker_max_age (default: 12): frame tollerati senza update.
- tracker_min_hits (default: 3): hit minimi per attivare un track.
- tracker_iou_th (default: 0.35): soglia IoU per matching.


## 🔎 Detector

### Person detector (YOLO ONNX)

Ambito: detection primaria delle persone. Se presente, il tracker usa queste bbox; altrimenti fallback sui volti.

- person_model_path (default: ""): path ONNX (vuoto = disattivato).
- person_img_size (default: 640): lato input del modello.
- person_score_th (default: 0.26): soglia confidenza minima.
- person_iou_th (default: 0.45): soglia IoU per NMS.
- person_max_det (default: 200): massimo numero di box in output.
- person_backend (default: 0), person_target (default: 0): backend/target DNN.

### Face detector (YuNet)

Ambito: detection volti per età/genere e ancoraggio ReID via embedding facciale.

- detector_model (default: "").
- detector_score_th (default: 0.8).
- detector_nms_iou (default: 0.3).
- detector_top_k (default: 5000).
- detector_backend (default: 0), detector_target (default: 0).
- detector_resize_width (default: 640): resize solo per detection.
- Compat: blocco opzionale "yunet": { onnx_path, score_th, nms_th, top_k } viene mappato su detector_* se non impostati.

### Associazione volto→persona

Ambito: collegare un volto alla bbox persona più plausibile per usare volto nel classifier/ReID.

- face_assoc_iou_th (default: 0.20): IoU minima per associare volto→persona.
- face_assoc_center_in (default: true): consente criterio “centro volto dentro bbox persona”.


## 🧠 Classificatore Età/Genere

Ambito: stima genere/età da crop volto via ONNX Runtime.

- age_model_path (default: ""), gender_model_path (default: "").
- age_buckets (default: ["0-13","14-24","25-34","35-44","45-54","55-64","65+"]).
- cls_min_face_px (default: 64): lato minimo volto per inferenza.
- cls_min_conf (default: 0.35): soglia confidenza genere.
- cls_interval_ms (default: 300): throttle per track.

### Modello combinato (consigliato)

Ambito: un solo ONNX che predice età+genere.

- combined_model_path (default: "").
- combined_input_size (default: [96, 96]).
- combined_bgr_input (default: true).
- combined_scale01 (default: false).
- combined_age_scale (default: 100.0).
- combined_gender_order (default: ["female","male"]).

### Throttle / caching classificazione

Ambito: prestazioni e stabilità label.

- I risultati sono memorizzati per track e riutilizzati fino a cls_interval_ms; lo smoothing del tracker riduce fluttuazioni.


## 🚶 ROI / Tripwire

Ambito: conteggio direzionale di attraversamenti su linea virtuale normalizzata.

- roi_tripwire (default: [[0.1,0.5],[0.9,0.5]]): punti normalizzati A→B.
- roi_direction (default: "both"): direzione valida (both|a2b|b2a).
- roi_band_px (default: 12): spessore banda di tolleranza (px).

Funzionamento: registra un evento quando il centro del box attraversa la tripwire; l’aggregatore crea metriche per finestra temporale.


## 🔁 Re-Identification (ReID)

### Obiettivi e panoramica

Ambito: riassociare la stessa persona su uscite/rientri entro TTL; ridurre duplicati e conteggi spuri.

La pipeline usa embedding di volto e corpo, più una firma di aspetto (colore) come prior debole. Una policy di soglie/fusione assegna l’ID più plausibile.

### Face ReID (SFace/ArcFace)

- reid_enabled (default: true): abilita ReID volto.
- reid_model_path (default: "").
- reid_similarity_th (default: 0.365): soglia match volto.
- reid_face_gate (default: 0.42): gate minimo per considerare affidabile il volto.
- reid_require_face_if_available (default: true): preferisci ID già ancorati da volto.
- reid_cache_size (default: 1000): dimensione cache ID.
- reid_memory_ttl_sec (default: 600): TTL memoria (eviction/presence).
- reid_bank_size (default: 10): max feature per banca/ID.
- reid_merge_sim (default: 0.55): soglia merge alias simili.
- reid_prefer_oldest (default: true): tie‑break verso ID più vecchio.
- reid_app_only_th (default: 0.65): soglia severa per match “solo aspetto”.

### Body ReID (OSNet / Intel OMZ)

- body_reid_model_path (default: ""): path modello corpo (vuoto = disattivo).
- body_reid_input_w (default: 128), body_reid_input_h (default: 256): input W×H.
- body_reid_backend (default: 0), body_reid_target (default: 0): backend/target DNN.
- body_only_th (default: 0.80): soglia match solo-corpo.
- reid_allow_body_seed (default: true): consenti creare ID con sola feature corpo quando nessun match è affidabile.

### Firma di aspetto legacy (colore vestiti)

- appearance_hist_bins (default: 24): bins istogramma HSV.
- appearance_min_area_px (default: 900): area minima del crop per calcolo firma.
- appearance_weight (default: 0.35): peso nella fusione (prior debole).

### Politiche di fusione e soglie

Priorità: volto > corpo > aspetto.  
- Se face_sim ≥ reid_similarity_th → match per volto.
- Altrimenti se body_sim ≥ body_only_th → match per corpo (con gate verso ID con volto se reid_require_face_if_available = true).
- In alternativa, aspetto se ≥ reid_app_only_th (prudenza).
- Nessun match → nuovo ID; se reid_allow_body_seed = true, semina banca corpo.

### Memoria, TTL e banca di feature

- reid_cache_size (default: 1000), reid_memory_ttl_sec (default: 600), reid_bank_size (default: 10): controllo memoria/TTL e rotazione feature per ID.

### Diagnostica ReID

- debug_reid_verbose (default: false): stampa decisioni (ID scelto, face/body/app, top‑3).


## 🧮 Modalità di conteggio e deduplica

Ambito: generazione eventi per metriche.

- count_mode (default: "presence"): "presence" | "tripwire".
  - presence: conteggio all’eviction (TTL) con genere/età prevalenti osservati.
  - tripwire: conteggio al passaggio oltre la linea A→B/B→A.
- presence_ttl_sec (default: 600): TTL presenza (usato in presence).
- count_dedup_ttl_sec (default: 600): dedup per stessa persona in tripwire.


## 📊 Metriche / Aggregazione

Ambito: raccolta eventi su finestre temporali per reporting.

- metrics_window_sec (default: 60): durata finestra (s).
- metrics_retention_min (default: 120): retention dati (minuti).

Output per finestra: counts per sesso (male/female/unknown) e per fascia d’età (0‑13 … 65+ / unknown), con ts ISO e windowSec.


## ⚙️ Suggerimenti di performance

- Riduci detector_resize_width (face) a 480–640 per accelerare la detection.
- Aumenta cls_interval_ms per ridurre inferenze ripetute sullo stesso volto.
- tracker_iou_th ~ 0.3–0.4 è un buon compromesso.
- In ReID, aumenta body_only_th (0.82+) in contesti con abbigliamento simile.
- Per RTSP instabile, regola rtsp_* (timeout/buffer) e valuta tcp vs udp.


## 🧪 Troubleshooting

- JSON non supporta commenti: non usare // o /* */ in config.json.
- Età/genere sempre unknown: verifica combined_model_path o modelli separati; controlla cls_min_face_px e illuminazione.
- ReID che collassa su un unico ID: alza body_only_th (es. 0.85), mantieni reid_require_face_if_available=true, calibra reid_similarity_th.
- RTSP instabile: alza timeout/buffer, riduci person_img_size, verifica rete.
