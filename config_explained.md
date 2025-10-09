# MyDisplay Vision – Configurazione (guida completa)

Questo documento elenca e spiega tutte le proprietà utilizzabili in config.json.  
Nota: JSON non supporta commenti; tieni i commenti in un file di esempio (es. config.example.json) oppure nel README.

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

- camera: indice della webcam (0 predefinito) oppure stringa RTSP.
- width, height: risoluzione richiesta alla camera (pixel).
- target_fps: FPS desiderati per l’elaborazione (throttle del loop).
- debug_enabled: abilita la pagina /debug e lo stream MJPEG.
- debug_stream_fps: frequenza dei frame nello stream MJPEG (non influenza il loop).
- debug_resize_width: larghezza per lo stream di debug, mantenendo aspect ratio (0 = nessun resize).


## 🌐 API

Ambito: server HTTP locale per diagnostica, stream e metriche.

- api_host: indirizzo bind (es. "127.0.0.1" o "0.0.0.0").
- api_port: porta del server (es. 8080).


## 📡 RTSP (sorgenti IP)

Ambito: tuning best‑effort per flussi RTSP letti da OpenCV/FFmpeg.

- rtsp_transport: "tcp" o "udp" (default tcp).
- rtsp_buffer_frames: dimensione buffer interno (frame).
- rtsp_open_timeout_ms: timeout di apertura (millisecondi).
- rtsp_read_timeout_ms: timeout di lettura (millisecondi).
- rtsp_reconnect_sec: attesa prima di riaprire dopo errori.
- rtsp_max_failures: quante read fallite prima di tentare un reopen.


## 🧭 Tracker (ID stabili)

Ambito: mantenere un ID coerente per persona/volto nei frame consecutivi (SORT‑lite).

- tracker_max_age: quanti frame tollerare senza update prima di eliminare il track.
- tracker_min_hits: frammenti minimi per considerare “valido” un track (filtra falsi positivi).
- tracker_iou_th: soglia IoU per associare una detection al track esistente (0.3–0.4 tipico).


## 🔎 Detector

### Person detector (YOLO ONNX)

Ambito: detection “primaria” delle persone. Se configurato, il tracker usa queste bbox; altrimenti fa fallback ai volti.

- person_model_path: percorso del modello ONNX (es. models/yolov8n.onnx). Vuoto = disattivato.
- person_img_size: lato dell’input (es. 640).
- person_score_th: soglia confidenza minima.
- person_iou_th: soglia IoU per NMS interno.
- person_max_det: numero massimo di detection in output.
- person_backend, person_target: backend/target per OpenCV DNN.

### Face detector (YuNet)

Ambito: detection volti per età/genere e per ancorare il ReID via face embedding.

- detector_model: percorso ONNX di YuNet (es. models/face_detection_yunet_2023mar.onnx).
- detector_score_th: soglia confidenza minima.
- detector_nms_iou: soglia NMS (IoU).
- detector_top_k: massimo numero di box processati.
- detector_backend, detector_target: backend/target OpenCV DNN.
- detector_resize_width: ridimensiona solo per il face detector (accelera senza toccare lo stream di debug).
- Compat: se nel JSON è presente un blocco "yunet": { onnx_path, score_th, nms_th, top_k }, viene mappato automaticamente sui detector_* se non già impostati.

### Associazione volto→persona

Ambito: collegare un volto alla bbox persona più plausibile per usare volto nel classifier e nel ReID.

- face_assoc_iou_th: IoU minima per associare volto→persona (0.2 tipico).
- face_assoc_center_in: se true, accetta anche “centro volto dentro bbox persona” come criterio.


## 🧠 Classificatore Età/Genere

Ambito: stima genere ed età da crop volto, via ONNX Runtime.

- age_model_path, gender_model_path: modelli separati (fallback se non c’è il combinato).
- age_buckets: etichette delle fasce d’età (devono allinearsi a report/metriche).
- cls_min_face_px: lato minimo del volto per tentare la classificazione (evita input troppo piccoli).
- cls_min_conf: confidenza minima per genere (sotto soglia → "unknown").
- cls_interval_ms: intervallo minimo tra inferenze sullo stesso track (cache/throttle).

### Modello combinato (consigliato)

Ambito: un solo ONNX che predice età e genere insieme (più veloce).

- combined_model_path: percorso del modello combinato (es. Intel age-gender-recognition-retail-0013.onnx, InsightFace genderage.onnx). Se presente, ha priorità.
- combined_input_size: dimensione input (es. [62,62] Intel, [96,96] InsightFace).
- combined_bgr_input: true se il modello attende BGR (tipico Intel/InsightFace), false per RGB.
- combined_scale01: scala a [0..1] se necessario (Intel solitamente lavora 0..255 → false).
- combined_age_scale: fattore per riportare l’età da output normalizzato (es. ×100).
- combined_gender_order: ordine classi nella predizione di genere (es. ["female","male"]).

### Throttle / caching classificazione

Ambito: prestazioni e stabilità label.

- Il sistema memorizza l’ultimo risultato per track e lo riutilizza fino a cls_interval_ms.
- Il tracker applica smoothing (finestra + EMA) per ridurre fluttuazioni.


## 🚶 ROI / Tripwire

Ambito: conteggio direzionale di attraversamenti su una linea virtuale normalizzata.

- roi_tripwire: due punti normalizzati [[x1,y1],[x2,y2]] (0..1 rispetto a frame) che definiscono la linea A→B.
- roi_direction: "both" | "a2b" | "b2a" (filtra la direzione valida).
- roi_band_px: spessore (pixel) della banda di tolleranza attorno alla linea.

Funzionamento: il sistema registra un evento quando il centro del box (persona/volto) attraversa la tripwire; l’aggregatore trasforma eventi in metriche per finestra temporale.


## 🔁 Re-Identification (ReID)

### Obiettivi e panoramica

Ambito: riassociare la stessa persona su uscite/rientri entro un TTL, anche senza volto visibile; ridurre duplicazioni e conteggi spuri.

La pipeline usa:
- Face ReID (embedding del volto) quando disponibile;
- Body ReID (embedding corpo) per agganciare persone senza volto;
- Firma di aspetto legacy (istogramma colore) come prior debole.
Una politica di fusione e soglie decide l’ID più plausibile.

### Face ReID (SFace/ArcFace)

- reid_enabled: abilita/disabilita il ReID.
- reid_model_path: modello SFace (OpenCV contrib) o ArcFace ONNX.
- reid_similarity_th: soglia di similarità (cosine) per match volto (0.35–0.40 tipico SFace).
- reid_face_gate: soglia minima per considerare affidabile il volto (gate interno).
- reid_require_face_if_available: se true, preferisce match verso ID già ancorati da un volto.
- reid_bank_size: numero max di embedding conservati per ID (banca rotante).
- reid_merge_sim: soglia per fondere alias molto simili (se supportato).
- reid_prefer_oldest: tie‑break verso l’ID più “anziano”.

### Body ReID (OSNet / Intel OMZ)

- body_reid_model_path: percorso modello corpo (es. models/osnet_x0_25_msmt17.onnx o person-reidentification-retail-0288.onnx). Vuoto = disattivato.
- body_reid_input_w, body_reid_input_h: dimensione input modello corpo (tipicamente 128x256 W×H).
- body_reid_backend, body_reid_target: backend/target DNN per OpenCV.
- body_only_th: soglia per match “solo corpo” (alza a 0.82–0.85 per essere conservativo).
- reid_allow_body_seed: consenti creare un nuovo ID “ancorato” inizialmente solo sul corpo (utile quando non si vedono volti).

### Firma di aspetto legacy (colore vestiti)

- appearance_hist_bins: bins istogramma HSV (maggiore = più fine).
- appearance_min_area_px: area minima crop per calcolare la firma.
- appearance_weight: peso della firma nel calcolo di prior (usata nella fusione come segnale debole).
- reid_app_only_th: soglia severa per match “solo aspetto” (usare con prudenza).

### Politiche di fusione e soglie

Priorità: volto > corpo > aspetto.  
- Se face_sim ≥ reid_similarity_th → match per volto.
- Altrimenti, se body_sim ≥ body_only_th → match per corpo (con gate verso ID ancorati al volto se reid_require_face_if_available = true).
- In alternativa, aspetto legacy se ≥ reid_app_only_th (più severo).
- Nessun match → nuovo ID; se reid_allow_body_seed = true, viene “seminata” anche la banca corpo.

### Memoria, TTL e banca di feature

- reid_cache_size: massimo numero di ID memorizzati.
- reid_memory_ttl_sec: durata della memoria ReID; scaduto il TTL, l’ID viene “evicted” (in modalità presence questo genera il conteggio).
- reid_bank_size: numero max di feature per ogni banca (face/body/app) per ID.

### Diagnostica ReID

- debug_reid_verbose: se true, stampa decisioni di assegnazione (ID scelto, similitudini face/body/app, top‑3 candidati). Utile per tuning delle soglie in campo.


## 🧮 Modalità di conteggio e deduplica

Ambito: come generare eventi da trasformare in metriche.

- count_mode: "presence" | "tripwire"
  - presence: il conteggio avviene alla “scadenza” (eviction) della presenza in memoria (TTL), associando genere/età prevalenti osservati.
  - tripwire: il conteggio avviene al passaggio oltre la linea A→B/B→A, opzionalmente con direzione filtrata.
- presence_ttl_sec: TTL di presenza (usato in modalità presence).
- count_dedup_ttl_sec: intervallo minimo prima di poter ricontare la stessa persona (dedup), in modalità tripwire.


## 📊 Metriche / Aggregazione

Ambito: raccolta eventi su finestre temporali regolari per reporting.

Ogni evento (presence o tripwire) viene passato al MinuteAggregator, che mantiene finestre e retention.

- metrics_window_sec: durata finestra (secondi). Es. 60 = per minuto.
- metrics_retention_min: retention (minuti) dei record aggregati (in memoria).
- Output per finestra: counts per sesso (male/female/unknown) e per fascia d’età (0‑13 … 65+ / unknown), ts ISO e windowSec.


## ⚙️ Suggerimenti di performance

- Riduci detector_resize_width (face) a 480–640 per accelerare la detection mantenendo stabilità.
- Aumenta cls_interval_ms per ridurre inferenze ripetute sullo stesso volto.
- Scegli tracker_iou_th ~ 0.3–0.4 per buon compromesso tra stabilità e distinzione persone vicine.
- In ReID, aumenta body_only_th (0.82+) in ambienti con abbigliamento simile per ridurre merge falsi.
- RTSP: regola rtsp_* secondo la stabilità della rete/camera; riduci person_img_size per ridurre latenza.


## 🧪 Troubleshooting

- “Non posso commentare JSON”: lo standard JSON non supporta commenti; tieni i commenti in file separati o usa un parser dedicato se vuoi supportarli.
- Età/genere sempre unknown: verifica combined_model_path o modelli separati; alza cls_min_face_px se i volti sono troppo piccoli; controlla luminosità.
- ReID “collassa” su un solo ID: alza body_only_th (es. 0.85), reid_similarity_th (se serve), mantieni reid_require_face_if_available = true.
- RTSP instabile: aumenta rtsp_open_timeout_ms/read_timeout_ms e rtsp_reconnect_sec; verifica banda; valuta tcp vs udp.