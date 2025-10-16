# MyDisplay Vision

Sistema di visione computerizzata on‑device per display interattivi MyDisplay. Rileva persone, associa ID stabili nel tempo, stima età/genere in modo anonimo, rileva attraversamenti di una tripwire e aggrega metriche esposte via API locali.

Tutto avviene in locale (nessun frame o video inviato a server), nel rispetto della privacy.


## Caratteristiche principali

- Rilevamento volto e persona (YuNet + YOLO ONNX opzionale)
- Tracking leggero con ID stabili (SORT‑lite)
- Classificazione età/genere (modello combinato, ONNX/OpenVINO)
- Re‑Identification opzionale (SFace/ArcFace + OSNet/OMZ) con memoria e TTL per deduplicare passaggi
- Tripwire normalizzata con conteggio direzionale a2b/b2a
- Aggregazione per finestre temporali (minute windows) e API HTTP di lettura
- Pagina debug con stream MJPEG, snapshot ReID e pagina /reports
- Logging strutturato su file JSONL con marcatori custom via API


## Come funziona (pipeline)

capture → detect (person/face) → track (SORT‑lite) → age/gender → re‑id (face/body) → tripwire/presence → aggregate → debug stream

- Se presente, il person detector (YOLO) fornisce le bbox primarie per il tracker; il face detector (YuNet) gira in parallelo per età/genere e per associare il volto alla persona.
- Se il person detector non è disponibile o non produce detezioni in un frame, il tracker usa i volti come input di fallback.
- Entrambi i moduli sono indipendenti: disattivati in assenza dei relativi modelli.
- Associazione volto→persona per usare il volto nel classificatore e nel Re‑ID.
- Due modalità di conteggio: presence (alla scadenza della presenza) e tripwire (all’attraversamento A→B/B→A).


## Requisiti

- Python 3.10+
- Windows 10/11 x64 o Linux x64
- CPU; ONNX Runtime e OpenCV DNN (CPUExecutionProvider)

Pacchetti principali (vedi requirements.txt):
- opencv‑contrib‑python 4.10.0.84
- onnxruntime 1.23.1
- numpy 1.26.4
- fastapi 0.118.3, uvicorn 0.37.0
- openvino 2024.6 (opzionale; principalmente per body ReID OMZ)


## Installazione

1) Clona il repository

```bash
git clone https://github.com/Fr4nZ82/mydisplay-vision.git
cd mydisplay-vision
```

2) Crea un ambiente virtuale

- Linux/macOS

```bash
python3.12 -m venv venv  # richiede Python 3.12.10
source venv/bin/activate
```

- Windows (PowerShell)

```powershell
py -3.12 -m venv venv  # richiede Python 3.12.10
venv\Scripts\Activate.ps1
```

3) Installa le dipendenze

```bash
pip install -r requirements.txt
```


## Modelli

I modelli non si configurano via path nel config. Copiali nelle cartelle corrette sotto models/ e il sistema li risolve automaticamente (precedenza: OpenVINO → ONNX):

Struttura attesa:
- models/face/(openvino|onnx)/           → face detector (YuNet ONNX supportato; OpenVINO non usato per YuNet)
- models/person/(openvino|onnx)/         → person detector (ONNX supportato; OpenVINO supportato solo se previsto dal wrapper corrente)
- models/genderage/(openvino|onnx)/      → classificatore età/genere combinato (ONNX o OpenVINO)
- models/reid_face/(openvino|onnx)/      → ReID volto (ONNX/OpenVINO)
- models/reid_body/(openvino|onnx)/      → ReID corpo (OSNet ONNX o Intel OMZ .xml)

Note pratiche:
- Face detector: YuNet ONNX in models/face/onnx/.
- Person detector: YOLO ONNX in models/person/onnx/ (classe "person").
- Età/Genere: combinato in models/genderage/(openvino|onnx)/ (es. Intel 0013 62x62).
- ReID volto: SFace/ArcFace in models/reid_face/onnx/ (OpenVINO accettato dal backend corrente).
- ReID corpo: OSNet (osnet_x0_25_msmt17.onnx) o Intel OMZ (person-reidentification-retail-0288.xml).

Se un modello non è presente nella cartella attesa, la relativa funzionalità viene disattivata e le etichette restano "unknown".


## Configurazione

Il servizio legge un file JSON (default: config.json). Tutte le opzioni sono documentate in:
- config_explained.md

Punti chiave:
- camera, width/height, target_fps
- detector_* (YuNet), person_* (YOLO)
- Modelli: auto-discovery in models/<categoria>/(openvino|onnx); nessun path in config
- tracker_*, roi_tripwire/roi_direction/roi_band_px
- count_mode: "presence" oppure "tripwire"
- reid_* e body_* (soglie e policy), filtri person_* (min area, ignore zones)
- api_host, api_port, debug_* e logging (log_*)

Esempio minimo di override:

```jsonc
{
  "camera": 0,
  "api_host": "127.0.0.1",
  "api_port": 8080,
  "count_mode": "tripwire",
  "roi_tripwire": [[0.1, 0.5], [0.9, 0.5]],
  "roi_direction": "both"
}
```


## Avvio rapido

```bash
python src/main.py --config config.json
```

Opzioni da CLI (override dei campi config):
- --camera, --width, --height, --target-fps, --help

All’avvio:
- API FastAPI su http://127.0.0.1:8080 (valori configurabili da api_host/api_port)
- Pagina debug: /debug con stream MJPEG annotato


## API HTTP

- GET /health → stato camera, fps, size, since, version
- GET /stats → framesTotal, uptimeSec, lastFps, size, cameraOk
- GET /debug → pagina HTML di debug
- GET /debug/stream → stream MJPEG
- GET /debug/frame → ultimo frame JPEG
- GET /debug/data → snapshot ReID (memoria + active tracks)
- GET /reports → pagina HTML di report (se presente)
- GET /metrics/minute?last=N&includeCurrent=1 → finestre aggregate
- GET /config → configurazione effettiva
- GET /log/mark?msg=...&tag=... → inserisce un marcatore nel log strutturato


## Logging

Il servizio scrive un log JSONL (rotazione e retention configurabili) se log_enabled=true. La cartella è log_dir (default logs). È disponibile l’endpoint /log/mark per inserire marcatori manuali.


## Modalità di conteggio

- presence: la memoria Re‑ID mantiene presenze con TTL; alla scadenza (eviction) si registra un evento con genere/età prevalenti.
- tripwire: quando il centro di un track attraversa la linea A→B/B→A entro una banda (roi_band_px), si registra un evento, con direzione filtrabile via roi_direction.

Dedup: count_dedup_ttl_sec evita doppi conteggi della stessa persona entro l’intervallo specificato.


## Struttura del progetto

- src/main.py: bootstrap, parsing argomenti, avvio API + pipeline
- src/runtime.py: ciclo principale, orchestrazione detector → tracker → classifier → re‑id → aggregator
- src/api.py: FastAPI e endpoint di health/debug/reports/metrics/config/log
- src/face_detector.py: YuNet wrapper (OpenCV FaceDetectorYN)
- src/person_detector.py: YOLO(ONNX) per classe "person" (opzionale)
- src/body_reid.py: backend per ReID corpo (OSNet ONNX / OMZ OpenVINO)
- src/tracker.py: SORT‑lite con smoothing etichette
- src/age_gender.py: classificatore età/genere (modello combinato)
- src/reid_memory.py: memoria Re‑ID (SFace/ArcFace) con policy apparenza
- src/aggregator.py: finestre minute, conteggio per genere/età e direzione
- src/model_resolver.py: auto-discovery modelli in models/
- src/logs.py: setup logging JSONL + log_event
- src/utils_vis.py: utility di visualizzazione (overlay, tripwire, IoU)
- src/state.py: stato condiviso API↔pipeline (jpeg, health, config, metrics)
- src/web/debug.html, src/web/reports.html: pagine HTML statiche
- config_explained.md: guida completa ai parametri


## Performance e suggerimenti

- Usa proc_resize_width per ridurre la risoluzione di lavoro e aumentare il throughput generale.
- Riduci detector_resize_width (es. 640→480) per accelerare la face detection.
- Tuning YOLO: su CPU person_img_size 416–576; su GPU 640–768; scene ampie/soggetti piccoli 736–768.
- Aumenta cls_interval_ms per ridurre inferenze ripetute sullo stesso ID.
- tracker_iou_th 0.30–0.40 regge jitter/RTSP; alzalo con stream stabili/detector accurato.
- Re‑ID: alza body_only_th (0.80+) in contesti con abbigliamento uniforme; usa reid_require_face_if_available=true per evitare merge spuri.
- RTSP: alza rtsp_open_timeout_ms/rtsp_read_timeout_ms e rtsp_buffer_frames; usa rtsp_transport="tcp" per stabilità.


## Limiti e roadmap

- Tracker semplice (IoU greedy): in scene affollate può generare ID instabili.
- Re‑ID dipende da posa/luci; le soglie vanno calibrate sul campo.
- Supporto RTSP best‑effort (timeout/buffer dipendono dalla build OpenCV).

Roadmap (prossime evoluzioni):
- Integrazione con ecosistema MyDisplay:
  - mydisplay-node: conteggio centralizzato e aggregazione dei dati provenienti da più istanze MyDisplay Vision.
  - mydisplay-player-electron e mydisplay-player-tizen: definizione/implementazione del canale di scambio dati e telemetria (offline e near‑realtime).
  - Gestione remota della configurazione via app.mydisplay.it (web‑app) con lettura/scrittura della config, validazione e sincronizzazione.
- Miglioramenti tracking multi‑oggetto e associazione volto→persona
- Ottimizzazioni su hardware a bassa potenza


## Privacy

- Nessuna immagine/video salvata o inviata. Elaborazione locale.
- Non si effettua riconoscimento identitario: gli ID sono locali/temporanei.
- Le metriche esposte sono aggregate e anonime.


## Troubleshooting

- Face detector non attivo → assicurati di avere YuNet ONNX in models/face/onnx/ e che detector_score_th non sia troppo alto.
- Età/genere sempre unknown → verifica modello combinato in models/genderage/(openvino|onnx) e coerenza dei parametri combined_* (es. input 62x62); controlla cls_min_face_px.
- Stream /debug lento → riduci debug_stream_fps e/o usa proc_resize_width; per la detection volti abbassa detector_resize_width.
- RTSP instabile → regola rtsp_* (timeout/reconnect/buffer) e riduci person_img_size.
- Overlay affollato → abilita debug_hide_uncommitted e debug_hide_ignored; usa person_ignore_zone e person_min_box_area.


## Contribuire

- Fork, branch feature/nome_feature, PR con descrizione e test
- Stile: codice Python tipizzato dove sensato, log puliti, commenti essenziali


## Licenza

Da definire (inserire licenza scelta e file LICENSE).
