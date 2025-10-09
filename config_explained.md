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

- camera (default: 0): indice webcam (USB/integrata) o stringa RTSP.
- width (default: 1920), height (default: 1080): risoluzione richiesta (px).
- target_fps (default: 10.0): FPS desiderati per il loop (throttle).
- debug_enabled (default: false): abilita /debug e stream MJPEG.
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

## Nota su IoU (Intersection over Union)
IoU misura quanta parte di due bounding box A e B si sovrappone rispetto alla loro unione. Si calcola come IoU = area(A ∩ B) / area(A ∪ B) e vale tra 0 e 1 (1 = box completamente sovrapposti, 0 = nessuna sovrapposizione). Nel tracking viene usata per associare una detection a un track esistente tra frame consecutivi: soglie più alte richiedono box molto coerenti, soglie più basse tollerano spostamenti rapidi, blur o variazioni di scala.


## Nota su NMS (Non‑Maximum Suppression)
L’NMS serve a eliminare i duplicati quando il detector produce più bounding box per lo stesso oggetto. Conserva le box con punteggio più alto e sopprime le altre troppo sovrapposte.

Come funziona (hard NMS):
1) ordina le box per score decrescente;
2) seleziona la migliore e aggiungila all’output;
3) sopprimi le altre con IoU superiore alla soglia rispetto a quella appena scelta;
4) ripeti finché non restano candidati o si raggiunge il limite massimo.

Linee guida pratiche:
- Vedi duplicati sulla stessa persona → abbassa leggermente la soglia IoU dell’NMS.
- Persone molto vicine “collassano” in una sola box → alza la soglia IoU dell’NMS e/o riduci leggermente lo score_th per far entrare più candidate.
- Scene molto affollate → valuta di alzare person_iou_th e person_max_det; in alternativa mantieni una soglia più alta ma aumenta gli score_th per contenere il costo in risorse.

Nota: la pipeline usa NMS “hard” (soppressione netta oltre soglia). Varianti come Soft‑NMS riducono progressivamente gli score anziché sopprimere, ma non sono attive qui.


## Nota su DNN (Deep Neural Network) e backend/target
Per DNN si intende il motore di inferenza che esegue i modelli di rete neurale. In questa pipeline, per i blocchi basati su OpenCV DNN (YOLO persone, YuNet face, Body ReID) si configurano due campi: backend e target.

- backend: seleziona il framework di esecuzione.
  - 0 = OPENCV (default, CPU)
  - 5 = CUDA (OpenCV DNN con CUDA – GPU NVIDIA)
  - 2 = OpenVINO/Inference Engine (Intel)
- target: seleziona dispositivo/precisione.
  - 0 = CPU
  - 6 = CUDA (FP32)
  - 7 = CUDA_FP16 (half precision)
  - 1 = OPENCL, 2 = OPENCL_FP16 (GPU generiche via OpenCL)
  - 3 = MYRIAD (Intel NCS2)

Combinazioni tipiche:
- CPU: backend=0, target=0
- GPU NVIDIA: backend=5, target=6 (FP32) oppure 7 (FP16)
- Intel/OpenVINO: backend=2, target=0 (CPU) oppure 3 (MYRIAD)

Suggerimenti:
- FP16 (target=7 o OPENCL_FP16) aumenta il throughput riducendo l’uso di memoria, con lieve perdita di precisione; ideale su GPU.
- Mantieni backend/target coerenti tra i blocchi più pesanti per evitare copie di memoria tra dispositivi.
- CUDA richiede OpenCV compilato con supporto DNN CUDA e driver NVIDIA compatibili; in caso contrario usa 0/0.
- OpenVINO richiede il runtime installato e modelli compatibili.
- I campi backend/target non si applicano al classificatore età/genere se eseguito con ONNX Runtime (CPU per default).

Diagnostica:
- Se compaiono errori di inizializzazione o fallback a CPU, prova backend=0/target=0 e verifica versione di OpenCV, driver e build con supporto DNN desiderato.


## 🧭 Tracker (ID stabili)

Ambito: mantenere un ID coerente per persona/volto tra frame consecutivi (SORT‑lite).

- tracker_max_age (default: 8): numero di frame consecutivi consentiti senza aggiornamento prima di marcare il track come perso ed eliminarlo. Valori alti aiutano con fps bassi, blur, occlusioni brevi o RTSP instabile (meno drop di ID), ma aumentano il rischio di “zombie”/agganci errati persistenti; valori bassi rendono il tracking più reattivo ma spezzano più facilmente gli ID. È espresso in frame (non in secondi): a 10 FPS, 12 ≈ 1.2 s. Linee guida: 8–16 per 8–12 FPS/RTSP; 3–6 per 20–30 FPS stabili. Aumenta se il detector è poco sensibile o la scala varia molto; riduci in scene affollate per limitare swap/merge.
- tracker_min_hits (default: 4): numero di associazioni consecutive (hit) richieste per promuovere un track da “tentativo” a “confermato”. Prima della conferma il track non viene esposto/contato, così da filtrare detezioni spurie o lampi singoli. Valori alti riducono falsi positivi ma aumentano la latenza di comparsa e possono perdere soggetti molto brevi; valori bassi rendono il sistema più reattivo ma possono generare ID effimeri. È espresso in frame: a 10 FPS, 3 ≈ 0.2–0.3 s di latenza prima della conferma. Linee guida: 2–3 per 8–15 FPS; 1 se il detector è molto pulito e serve bassa latenza; 4–5 se il detector è rumoroso o la sorgente RTSP è instabile (burst). Regolare in coppia con tracker_max_age: con fps bassi o occlusioni brevi, riduci min_hits o aumenta max_age per facilitare la conferma.
- tracker_iou_th (default: 0.35): soglia IoU per associare una detection a un track esistente. Valori alti richiedono più sovrapposizione (tracking più severo: meno swap di ID ma più drop con movimenti rapidi o camera instabile); valori bassi sono più tolleranti (reggono low‑fps/blur/RTSP instabile ma aumentano match errati). Range consigliato 0.30–0.50; alza con camera/fps stabili e detector preciso, abbassa con soggetti veloci o variazioni di scala marcate.


## 🔎 Detector

### Person detector (YOLO ONNX)

Ambito: detection primaria delle persone. Se presente, il tracker usa queste bbox; altrimenti fallback sui volti.

- person_model_path (default: ""): path ONNX (vuoto = disattivato).
- person_img_size (default: 640): lato di input (px) usato per il resize “letterbox” prima dell’inferenza YOLO. Valori maggiori rilevano soggetti più piccoli e lontani ma aumentano latenza/uso di CPU/GPU e RAM (costo ~quadratico); valori minori velocizzano ma possono perdere oggetti piccoli. Deve essere multiplo di 32 (es. 320, 416, 512, 576, 640, 736, 768). Non modifica la risoluzione del frame di acquisizione; le bbox sono riportate alla dimensione originale. Linee guida: CPU/RTSP instabile 416–576; GPU 640–768; soggetti molto piccoli/scene ampie 736–768.
- person_score_th (default: 0.26): soglia di confidenza minima per mantenere una detection YOLO (objectness × class). Più bassa = più recall (rileva anche soggetti piccoli/lontani) ma più falsi positivi e costo NMS; più alta = più precisione ma rischio di perdere volti/persona deboli o parziali. Linee guida: 0.22–0.30 su CPU/RTSP instabile; 0.30–0.40 su scene pulite/GPU. Regolare insieme a person_iou_th.
- person_iou_th (default: 0.45): soglia IoU per NMS (sopprime box con IoU > soglia). Valori più bassi rendono l’NMS più aggressivo (meno duplicati ma possibile soppressione di persone molto vicine/overlap); valori più alti conservano più box (utile in folle/overlap, ma aumentano duplicati e costo). Range tipico 0.40–0.55; 0.45 è un buon compromesso.
- person_max_det (default: 200): limite massimo di bbox persona restituite per frame dopo NMS. Ridurlo limita il carico su tracker/classificatore in scene affollate; aumentarlo evita “tagli” in crowd densi. Linee guida: 100–200 per retail standard; 300–500 per scene molto affollate; 50–100 su CPU deboli.
- person_backend (default: 0): backend DNN OpenCV per l’inferenza YOLO. Valori comuni: 0=DEFAULT/OPENCV (CPU), 5=CUDA (NVIDIA), 2=OpenVINO/Inference Engine (Intel). Deve essere coerente con person_target.
- person_target (default: 0): target di esecuzione per il backend scelto. Valori comuni: 0=CPU, 6=CUDA, 7=CUDA_FP16, 1=OPENCL, 2=OPENCL_FP16, 3=MYRIAD (NCS2). Esempi: 0/0 per CPU; 5/6 o 5/7 per GPU NVIDIA; 2/0 (CPU) o 2/3 (MYRIAD) con OpenVINO.

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
- face_assoc_center_in (default: true): abilita un criterio complementare all’IoU per associare volto→persona. Se il punto centrale del box volto cade all’interno della bbox persona candidata, l’associazione è considerata valida anche quando l’IoU è bassa (viso molto piccolo dentro un box corpo grande).
  - Regola primaria: associa se IoU ≥ face_assoc_iou_th.
  - Fallback: se IoU < soglia ma il centro volto è contenuto in una o più bbox persona, scegli quella con IoU più alta; se nessuna lo contiene, non associare.
  - Quando usarlo: tienilo attivo per robustezza a scale diverse e jitter del detector; valuta di disattivarlo in scene molto affollate o con forti occlusioni, dove più bbox persona si sovrappongono e il centro volto può cadere nel box sbagliato.


## 🧠 Classificatore Età/Genere

Ambito: stima genere/età da crop volto via ONNX Runtime.

### Proprietà comuni (sia modello combinato che doppio modello)
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

### Modelli separati

Ambito: due ONNX distinti, uno per età e uno per genere.

- age_model_path (default: "").
- gender_model_path (default: "").

## 🚶 ROI / Tripwire

Ambito: conteggio direzionale di attraversamenti su linea virtuale normalizzata.

- roi_tripwire (default: [[0.1,0.5],[0.9,0.5]]): punti normalizzati A→B.
- roi_direction (default: "both"): direzione valida (both|a2b|b2a).
- roi_band_px (default: 12): spessore banda di tolleranza (px).

Funzionamento: registra un evento quando il centro del box attraversa la tripwire; l’aggregatore crea metriche per finestra temporale.


## 🔁 Re-Identification (ReID)

### Obiettivi e panoramica

Ambito: riassociare la stessa persona su uscite/rientri entro TTL; ridurre duplicati e conteggi spuri.

La pipeline usa embedding di volto e corpo; la firma di aspetto (colore) è legacy/opzionale e, in presenza di Body ReID, si consiglia di disattivarla (appearance_weight=0). Una policy di soglie/fusione assegna l’ID più plausibile con priorità volto > corpo > (eventuale) aspetto.

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
