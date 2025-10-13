# MyDisplay Vision

Sistema di visione computerizzata on‑device per display interattivi MyDisplay. Rileva persone, associa ID stabili nel tempo, stima età/genere in modo anonimo, rileva attraversamenti di una tripwire e aggrega metriche esposte via API locali.

Tutto avviene in locale (nessun frame o video inviato a server), nel rispetto della privacy.


## Caratteristiche principali

- Rilevamento volto e persona (YuNet + YOLO ONNX opzionale)
- Tracking leggero con ID stabili (SORT‑lite)
- Classificazione età/genere (modello combinato, ONNX/OpenVINO)
- Re‑Identification opzionale (SFace/ArcFace) con memoria e TTL per deduplicare passaggi
- Tripwire normalizzata con conteggio direzionale a2b/b2a
- Aggregazione per finestre temporali (minute windows) e API HTTP di lettura
- Pagina debug con stream MJPEG e info ReID attive


## Come funziona (pipeline)

capture → detect (person/face) → track (SORT‑lite) → age/gender → re‑id → tripwire/presence → aggregate → debug stream

- Se abilitato, il person detector (YOLO) fornisce le bbox primarie per il tracker; il face detector (YuNet) gira in parallelo per età/genere e per associare il volto alla persona.
- Se il person detector non è disponibile o non produce detezioni in un frame, il tracker usa i volti come input di fallback.
- Entrambi i moduli sono indipendenti: puoi disattivarli non indicando i rispettivi modelli nel config.
- Associazione volto→persona per usare il volto nel classificatore e nel Re‑ID
- Due modalità di conteggio:
  - presence: si conta alla “uscita” dalla memoria (eviction) di una presenza
  - tripwire: si conta al passaggio del centro del box oltre la linea A→B/B→A


## Requisiti

- Python 3.10+
- Windows 10/11 x64 o Linux x64
- CPU; ONNX Runtime e OpenCV DNN (CPUExecutionProvider)

Versioni principali (requirements.txt):
- opencv‑contrib‑python 4.12.x
- onnxruntime 1.23.x
- numpy 2.2.x


## Installazione

1) Clona il repository

```bash
git clone https://github.com/Fr4nZ82/mydisplay-vision.git
cd mydisplay-vision
```

2) Crea un ambiente virtuale

- Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

- Windows (PowerShell)

```powershell
py -3 -m venv venv
venv\Scripts\Activate.ps1
```

3) Installa le dipendenze

```bash
pip install -r requirements.txt
```


## Modelli

I modelli non si configurano più tramite path nel config. Basta copiarli nelle cartelle corrette sotto models/ e il sistema li risolve automaticamente (precedenza: OpenVINO → ONNX):

Struttura attesa:
- models/face/(openvino|onnx)/           → face detector (YuNet ONNX supportato; OpenVINO segnalato ma non usato per YuNet)
- models/person/(openvino|onnx)/         → person detector (ONNX supportato; OpenVINO segnalato ma non usato)
- models/genderage/(openvino|onnx)/      → classificatore età/genere combinato (ONNX o OpenVINO)
- models/reid_face/(openvino|onnx)/      → ReID volto (ONNX supportato; OpenVINO non usato dal backend corrente)
- models/reid_body/(openvino|onnx)/      → ReID corpo (ONNX o OpenVINO)

Note di supporto attuali:
- Face detector: serve un ONNX YuNet in models/face/onnx/.
- Person detector: serve un ONNX YOLO in models/person/onnx/ (classe person).
- Età/Genere: modello combinato in models/genderage/(openvino|onnx)/; es. Intel age-gender-recognition-retail-0013 (62x62) o InsightFace genderage.
- ReID volto: ONNX in models/reid_face/onnx/ (SFace/ArcFace).
- ReID corpo: ONNX/OpenVINO in models/reid_body/(onnx|openvino)/. Supportati e testati: OSNet (osnet_x0_25_msmt17.onnx) e Intel OMZ (person-reidentification-retail-0288.xml).

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
- reid_* e parametri appearance_*
- api_host, api_port, debug_*

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
- API FastAPI su http://127.0.0.1:8080 (valori configurabili)
- Pagina debug: /debug con stream MJPEG annotato


## API HTTP

- GET /health → stato camera, fps, size, since, version
- GET /stats → framesTotal, uptimeSec, lastFps, size, cameraOk
- GET /debug → pagina HTML di debug
- GET /debug/stream → stream MJPEG
- GET /debug/frame → ultimo frame JPEG
- GET /debug/data → snapshot ReID (memoria + active tracks)
- GET /metrics/minute?last=N&includeCurrent=1 → finestre aggregate
- GET /config → configurazione effettiva


## Modalità di conteggio

- presence: Re‑ID mantiene una memoria con TTL; quando una presenza scade, l’aggregatore registra un evento associando il genere/età prevalenti osservati.
- tripwire: quando il centro di un track attraversa la linea normalizzata A→B/B→A entro una banda di tolleranza (roi_band_px), si registra un evento. Direzioni filtrabili con roi_direction.

Dedup: count_dedup_ttl_sec evita doppi conteggi della stessa persona entro un intervallo.


## Struttura del progetto

- src/main.py: bootstrap, parsing argomenti, avvio API + pipeline
- src/runtime.py: ciclo principale, orchestrazione detector → tracker → classifier → re‑id → aggregator
- src/api.py: FastAPI e endpoint di health/debug/metrics/config
- src/face_detector.py: YuNet wrapper (OpenCV FaceDetectorYN)
- src/person_detector.py: YOLO(ONNX) per classe "person" (opzionale)
- src/tracker.py: SORT‑lite con smoothing etichette
- src/age_gender.py: classificatore età/genere (modello combinato)
- src/reid_memory.py: memoria Re‑ID (SFace/ArcFace) con policy apparenza
- src/aggregator.py: finestre minute, conteggio per genere/età e direzione
- src/utils_vis.py: utility di visualizzazione (overlay, tripwire, IoU)
- src/state.py: stato condiviso API↔pipeline (jpeg, health, config, metrics)
- src/web/debug.html: pagina di debug
- config_explained.md: guida completa ai parametri


## Performance e suggerimenti

- Riduci detector_resize_width (es. 480–640) per accelerare la face detection
- Aumenta cls_interval_ms per ridurre inferenze ripetute
- Imposta count_mode in base all’uso (presence vs tripwire)
- Tieni tracker_iou_th intorno a 0.3–0.4 per un buon compromesso
- Re‑ID SFace richiede opencv‑contrib‑python; in alternativa usa ArcFace ONNX


## Limiti e roadmap

- Tracker semplice (IoU greedy): in scene affollate può generare ID instabili
- Re‑ID dipendente da condizioni di posa/luci; soglie da calibrare sul campo
- Supporto RTSP best‑effort (timeout/buffer dipendono dalla build OpenCV)

Planned:
- Miglioramenti tracking multi‑oggetto e associazione volto→persona
- Ottimizzazioni su hardware a bassa potenza
- Integrazione dati con MyDisplayPlayer


## Privacy

- Nessuna immagine/video salvata o inviata. Elaborazione locale.
- Non si effettua riconoscimento identitario: gli ID sono locali/temporanei.
- Le metriche esposte sono aggregate e anonime.


## Troubleshooting

- Face detector non attivo → verifica che un modello YuNet ONNX sia in models/face/onnx/ e che detector_score_th non sia troppo alto
- Età/genere sempre unknown → verifica che un modello combinato sia in models/genderage/(openvino|onnx)/ e che i parametri combined_* siano coerenti (es. input 62x62 per Intel 0013); controlla cls_min_face_px
- Stream /debug lento → abbassa debug_resize_width e/o debug_stream_fps
- RTSP instabile → regola rtsp_* (timeout, reconnect) e riduci person_img_size


## Contribuire

- Fork, branch feature/nome_feature, PR con descrizione e test
- Stile: codice Python tipizzato dove sensato, log puliti, commenti essenziali


## Licenza

Da definire (inserire licenza scelta e file LICENSE).
