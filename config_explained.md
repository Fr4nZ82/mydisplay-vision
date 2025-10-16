# MyDisplay Vision – Configurazione (guida completa)

Questo documento è la guida ufficiale per il tuning di config.json. È stato allineato ai parametri realmente supportati dal codice (vedi src/config.py) e copre ogni chiave disponibile, con default, tipo, valori ammessi e linee guida pratiche.

Come leggere questo file
- I modelli non hanno path nel config: sono caricati automaticamente se presenti in models/<categoria>/(openvino|onnx)/.
  - models/person/(openvino|onnx)/ → detector persone (YOLO ONNX preferito)
  - models/face/(openvino|onnx)/ → face detector (YuNet ONNX)
  - models/genderage/(openvino|onnx)/ → classificatore combinato età+genere
  - models/reid_face/onnx/ → face ReID (SFace/ArcFace)
  - models/reid_body/(openvino|onnx)/ → body ReID (OSNet/OMZ)
- JSON non supporta commenti: non inserire // o /* */ in config.json.
- Le note concettuali (IoU/NMS/DNN) sono qui sopra e in fondo; l’elenco delle proprietà è contiguo e completo.

Concetti chiave (riassunto)
- IoU (Intersection over Union): misura la sovrapposizione tra due box, usata per tracking e NMS. IoU = area(intersezione) / area(unione), in [0..1].
- NMS (Non‑Maximum Suppression): elimina box duplicate mantenendo le più forti e sopprimendo quelle con IoU oltre soglia.
- DNN backend/target (OpenCV DNN): scegli motore e dispositivo di inferenza.
  - backend: 0=OPENCV, 5=CUDA, 2=OpenVINO
  - target: 0=CPU, 6=CUDA, 7=CUDA_FP16, 1=OPENCL, 2=OPENCL_FP16, 3=MYRIAD
  - Tipico: CPU 0/0; NVIDIA 5/6 o 5/7; OpenVINO 2/0 (CPU) o 2/3 (MYRIAD).

—

Elenco proprietà config.json (contiguo)
- camera (default: 0) tipo: int|string. Indice webcam (0,1,…) oppure URL RTSP/HTTP/FILE. Con RTSP valgono i parametri rtsp_*.
- width (default: 1920) tipo: int. Larghezza richiesta del frame di acquisizione.
- height (default: 1080) tipo: int. Altezza richiesta del frame di acquisizione.
- target_fps (default: 10.0) tipo: float. Limite FPS del loop (throttle). Aumentarlo riduce latenza ma aumenta carico.
- debug_enabled (default: false) tipo: bool. Abilita endpoint /debug e stream MJPEG con overlay.
- debug_stream_fps (default: 5) tipo: int. FPS dello stream MJPEG di /debug.
- api_host (default: "127.0.0.1") tipo: string. Bind address per le API; usare "0.0.0.0" per esposizione in LAN.
- api_port (default: 8080) tipo: int. Porta HTTP del server locale.
- rtsp_transport (default: "tcp") tipo: string. "tcp" o "udp"; tcp più affidabile, udp più reattivo ma fragile.
- rtsp_buffer_frames (default: 2) tipo: int. Buffer interno (frame) lato demux/decoding.
- rtsp_open_timeout_ms (default: 4000) tipo: int. Timeout apertura RTSP in millisecondi.
- rtsp_read_timeout_ms (default: 4000) tipo: int. Timeout lettura pacchetti/frames in millisecondi.
- rtsp_reconnect_sec (default: 2.0) tipo: float. Attesa tra tentativi di riapertura flusso.
- rtsp_max_failures (default: 60) tipo: int. Letture fallite prima del reopen forzato.
- log_enabled (default: true) tipo: bool. Abilita logging strutturato su file in log_dir.
- log_dir (default: "logs") tipo: string. Cartella per i file di log.
- log_level (default: "INFO") tipo: string. Livello: DEBUG | INFO | WARNING | ERROR.
- log_rotate_mb (default: 10) tipo: int. Rotazione log al superamento di N MB per file.
- log_keep (default: 5) tipo: int. Quanti file ruotati conservare per ciascun log.
- proc_resize_width (default: 0) tipo: int. Larghezza di lavoro del frame elaborato; 0=usa frame pieno. Valori >0 riducono il carico ma peggiorano la sensibilità su oggetti piccoli.
- count_mode (default: "presence") tipo: string. "presence" per conteggio a fine presenza; "tripwire" per conteggio su attraversamento.
- presence_ttl_sec (default: 600) tipo: int. TTL di una presenza prima dell’eviction (solo in presence).
- person_img_size (default: 640) tipo: int. Lato input del detector persone (letterbox). Multiplo di 32. Più alto = più recall su piccoli, ma più costo.
- person_score_th (default: 0.30) tipo: float [0..1]. Soglia confidenza minima YOLO. Più bassa = più recall, più falsi; regolare con person_iou_th.
- person_iou_th (default: 0.45) tipo: float [0..1]. Soglia IoU per NMS persone. Più bassa = NMS più aggressivo, meno duplicati.
- person_max_det (default: 200) tipo: int. Numero massimo di bbox persona restituite per frame dopo NMS.
- person_backend (default: 0) tipo: int. Backend DNN per detector persone. Vedi mappa backend sopra.
- person_target (default: 0) tipo: int. Target dispositivo/precisione per il detector persone. Vedi mappa target sopra.
- person_min_box_area (default: 0) tipo: int. Area minima (w*h in px) del box persona; 0 disabilita. Utile per filtrare soggetti troppo lontani.
- person_ignore_zone (default: []) tipo: lista di poligoni. Ognuno è una lista di punti normalizzati [x,y] in [0..1] rispetto a (width,height). Le persone il cui centro cade dentro un poligono vengono ignorate. Esempio rettangolo: [[0.6,0.0],[1.0,0.0],[1.0,1.0],[0.6,1.0]].
- face_assoc_iou_th (default: 0.25) tipo: float [0..1]. IoU minima volto↔persona per associare il volto a una bbox persona.
- face_assoc_center_in (default: true) tipo: bool. Se true, se il centro del volto è dentro una persona candidata, l’associazione è valida anche a IoU < soglia, scegliendo quella con IoU più alta.
- detector_score_th (default: 0.75) tipo: float [0..1]. Soglia confidenza per YuNet (volti). Valori più bassi aumentano recall ma anche falsi.
- detector_nms_iou (default: 0.3) tipo: float [0..1]. Soglia IoU per NMS sui volti.
- detector_top_k (default: 5000) tipo: int. Limite candidati pre‑NMS; raramente da modificare.
- detector_backend (default: 0) tipo: int. Backend DNN per face detector. Vedi mappa backend.
- detector_target (default: 0) tipo: int. Target per face detector. Vedi mappa target.
- detector_resize_width (default: 800) tipo: int. Ridimensiona il frame solo per la detection volti; più basso = più veloce, meno sensibile sui volti piccoli.
- debug_hide_ignored (default: true) tipo: bool. In /debug non mostra volti/detections ignorati da filtri.
- debug_mark_centers (default: false) tipo: bool. Disegna un “+” sul centro dei volti in /debug (utile per associazione volto→persona).
- debug_show_ignore_rects (default: true) tipo: bool. Mostra i poligoni/zone di ignore nell’overlay di /debug.
- debug_reid_verbose (default: false) tipo: bool. Log verboso delle decisioni di ReID (match, candidati, motivi).
- debug_hide_uncommitted (default: true) tipo: bool. Nasconde in overlay i track non ancora confermati/committati (niente GID/attributi).
- debug_log_frame_out (default: false) tipo: bool. Logga ogni FRAME_OUT; molto verboso, solo per diagnostica puntuale.
- debug_log_detect (default: true) tipo: bool. Logga gli eventi di detection.
- debug_log_detect_zero (default: false) tipo: bool. Logga anche quando non si rileva nessuna persona/volto.
- debug_log_health (default: true) tipo: bool. Abilita log periodici di health/throughput.
- combined_input_size (default: [62, 62]) tipo: [int,int]. Dimensione input del modello combinato età+genere (Intel 0013).
- combined_bgr_input (default: true) tipo: bool. True se il modello si aspetta BGR; false per RGB.
- combined_scale01 (default: false) tipo: bool. True per normalizzare input a [0..1]; per Intel 0013 lasciare false (usa 0..255 float).
- combined_age_scale (default: 100.0) tipo: float. Fattore per convertire l’uscita età normalizzata in anni (Intel 0013 usa age/100).
- combined_gender_order (default: ["female","male"]) tipo: lista string. Ordine logit/probabilità di genere atteso dal modello.
- age_buckets (default: ["0-13","14-24","25-34","35-44","45-54","55-64","65+"]) tipo: lista string. Etichette per le fasce d’età in output.
- cls_min_face_px (default: 64) tipo: int. Lato minimo del volto (px) per avviare la classificazione età/genere.
- cls_min_conf (default: 0.35) tipo: float [0..1]. Soglia minima di confidenza per accettare la previsione.
- cls_interval_ms (default: 300) tipo: int. Minimo intervallo per ripetere la classificazione sullo stesso track (cache/throttle).
- tracker_max_age (default: 8) tipo: int. Frame massimi senza aggiornamento prima di marcare un track come perso. Aumenta per fps bassi/occlusioni brevi.
- tracker_min_hits (default: 4) tipo: int. Hit consecutivi richiesti per confermare un track; valori alti filtrano rumore ma aumentano latenza.
- tracker_iou_th (default: 0.35) tipo: float [0..1]. Soglia IoU per l’associazione detection↔track; più alta = più severo.
- roi_tripwire (default: [[0.1, 0.5], [0.9, 0.5]]) tipo: [[float,float],[float,float]]. Linea A→B con coordinate normalizzate.
- roi_direction (default: "both") tipo: string. "both" | "a2b" | "b2a"; direzione considerata valida per il conteggio.
- roi_band_px (default: 12) tipo: int. Spessore banda di tolleranza attorno alla tripwire in pixel frame.
- reid_enabled (default: true) tipo: bool. Abilita la Re‑Identification basata su volto.
- reid_similarity_th (default: 0.38) tipo: float. Soglia di similitudine coseno per match volto (embedding L2‑normalizzati).
- reid_cache_size (default: 1000) tipo: int. Dimensione massima della cache ID in memoria.
- reid_memory_ttl_sec (default: 600) tipo: int. TTL memoria degli ID per eviction/statistiche.
- reid_bank_size (default: 10) tipo: int. Numero massimo di feature per banca/ID (rotazione FIFO).
- reid_require_face_if_available (default: false) tipo: bool. Se true, quando un ID in memoria possiede embedding volto, preferisce/impone conferma via volto prima di accettare solo‑corpo; aiuta a evitare merge spuri ma può aumentare frammentazioni.
- reid_min_face_px (default: 56) tipo: int. Lato minimo del crop volto per estrarre embedding ReID; sotto la soglia si evita di usare feature rumorose.
- reid_face_body_bias (default: 0.02) tipo: float. Bias a favore del match volto nella fusione con il corpo (valori >0 privilegiano il volto in casi borderline). 0 per disabilitare.
- reid_min_body_h_px (default: 100) tipo: int. Altezza minima (px) del box corpo per calcolare/accettare embedding di corpo.
- body_reid_input_w (default: 128) tipo: int. Larghezza input del modello body ReID.
- body_reid_input_h (default: 256) tipo: int. Altezza input del modello body ReID.
- body_reid_backend (default: 0) tipo: int. Backend DNN per body ReID (solo modelli ONNX). Ignorato per OpenVINO .xml.
- body_reid_target (default: 0) tipo: int. Target DNN per body ReID (solo ONNX). Vedi mappa target.
- body_only_th (default: 0.65) tipo: float. Soglia di similitudine coseno per accettare un match solo‑corpo quando il volto non matcha.
- reid_allow_body_seed (default: true) tipo: bool. Consente di creare nuovi ID “seminati” solo con feature corpo quando nessun match è affidabile.
- body_commit_min_hits (default: 5) tipo: int. Hit minimi prima di committare un nuovo GID basato solo‑corpo; se non impostato userebbe tracker_min_hits.
- count_dedup_ttl_sec (default: 600) tipo: int. Deduplica in modalità tripwire: evita doppi conteggi della stessa persona entro N secondi.
- metrics_window_sec (default: 60) tipo: int. Durata finestra di aggregazione metriche (secondi).
- metrics_retention_min (default: 120) tipo: int. Retention dei dati aggregati in memoria (minuti).

—

Suggerimenti di performance
- Preferisci detector_resize_width 640–800 per compromesso velocità/accuratezza sui volti.
- Su CPU, person_img_size 416–576; su GPU 640–768; scene ampie/soggetti piccoli 736–768.
- Aumenta cls_interval_ms per ridurre inferenze ripetute su stessi ID.
- tracker_iou_th 0.30–0.40 regge jitter/RTSP; alza con stream stabili.
- In ReID, aumenta body_only_th (0.80+) in contesti con abbigliamento simile o rischio di merge.
- Per RTSP instabile: alza rtsp_open_timeout_ms/rtsp_read_timeout_ms, incrementa rtsp_buffer_frames, valuta rtsp_transport="tcp".

Troubleshooting
- Face/age‑gender sempre unknown: verifica modello combinato in models/genderage/(openvino|onnx), luce sufficiente, cls_min_face_px coerente.
- Face detector non attivo: metti un YuNet ONNX in models/face/onnx/.
- ReID non effettiva: verifica modelli in models/reid_face/onnx e models/reid_body/(onnx|openvino). Regola reid_similarity_th/body_only_th.
- Collasso ReID su un unico ID: alza body_only_th (0.80–0.85), tieni reid_require_face_if_available=true, verifica reid_min_face_px.
- Overlay troppo affollato: usa debug_hide_uncommitted=true, debug_hide_ignored=true e riduci person_max_det.
- Log troppo verbosi: imposta log_level="INFO" e disattiva debug_log_frame_out/debug_reid_verbose.

