# MyDisplay Vision (MVP)
Servizio di **analisi video on-device** per i player MyDisplay. Rileva il **passaggio di persone** davanti al display e stima **sesso** e **fascia d’età** in modo **anonimo**, inviando **metriche aggregate** al server MyDisplay.

> **Stato**: MVP sperimentale — prima implementazione su **Windows**, estendibile a Linux; Tizen valutato tramite offload verso un nodo locale.

---

## Obiettivi
- **Conteggio** persone (footfall) senza identificazione.
- **Stima sesso** \{male, female, unknown\} con soglia di confidenza.
- **Stima età** in **fasce** (0–13, 14–24, 25–34, 35–44, 45–54, 55–64, 65+, unknown).
- **Aggregazione locale** per finestre di 1 minuto e invio periodico al server.
- **Privacy-first**: niente upload di frame/video; tutto **on-device**.

---

## Come funziona (alto livello)
```
[Webcam] → [Capture] → [Detect+Track] → [Age/Gender Classify] → [Tripwire/ROI]
                                                │
                                                └─→ [Smoothing per ID] → [Aggregator/min]
                                                                │
                                                                └─→ [Buffer locale] → [API server]
```
- **Capture**: acquisizione camera (Media Foundation/DirectShow su Windows).
- **Detect+Track**: detection leggera + tracking (SORT/OC-SORT) per evitare doppi conteggi.
- **Classify**: modelli ONNX piccoli per sesso ed età.
- **Tripwire/ROI**: linea/area virtuale; conta al primo attraversamento per ogni ID.
- **Aggregator**: comprime in **metriche per minuto**; invio con retry/backoff.

---

## Stack tecnico (MVP)
- **Python 3.13** con **virtualenv (venv)**
- **OpenCV** (`opencv-contrib-python`) per capture, pre/post-processing, tracking
- **NumPy** per calcolo matriciale
- **ONNX Runtime (CPU)** per inferenza dei modelli (age/gender, detector opzionale)
- API locale (in una fase successiva): **HTTP /health**, **/stats**, **/config**

---

## Requisiti minimi consigliati
- **CPU**: Intel i3/i5 8th+ (ok anche Celeron con fps ridotti)
- **RAM**: 8 GB
- **Storage**: 64 GB
- **Camera**: 720p/1080p; montaggio sopra display, ROI 1–4 m
- **OS**: Windows 10/11 (Linux in roadmap)

---

## Setup rapido (Windows)
1. **Python & venv**
   ```powershell
   cd E:\projects\mydisplay-vision
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip setuptools wheel
   ```
2. **Dipendenze base**
   ```powershell
   pip install numpy opencv-contrib-python onnxruntime
   ```
3. **Smoke test**
   ```powershell
   python test_env.py   # verrà aggiunto con un esempio di capture
   ```

---

## Struttura (iniziale)
```
mydisplay-vision/
├─ .venv/                  # ambiente virtuale (non versionato)
├─ src/
│  ├─ main.py              # entrypoint del servizio (loop + IPC/API)
│  ├─ detector.py          # rilevamento volti/persone (ONNX o cv.dnn)
│  ├─ tracker.py           # tracking (SORT/OC-SORT)
│  ├─ classifier.py        # età/sesso (ONNX)
│  ├─ aggregator.py        # aggregati per finestra temporale
│  └─ config.py            # gestione configurazione/ROI
├─ tests/
│  └─ test_env.py          # test ambiente/librerie + capture
├─ requirements.txt        # elenco pacchetti (verrà popolato)
└─ README.md
```

---

## Privacy & Compliance (GDPR-first)
- **Elaborazione locale** (on-device), niente video/frame salvati di default.
- **Data minimization**: il server riceve **solo aggregati**.
- **Debug opzionale** con blur e retention limitata.
- **Cartellonistica informativa** e **DPIA** raccomandata prima del rollout.

---

## Roadmap
- **Fase A (MVP)**: loop di cattura + detection/tracking + classificazione + aggregati/min locale.
- **Fase B**: API locale `/health` `/stats` `/config`, tool di calibrazione ROI.
- **Fase C**: integrazione con MyDisplayPlayer3 (Electron), endpoint server e dashboard, auto-update, hardening.

---

## Licenza
TBD.

---

### Autori
MyDisplay Team — progettazione e sviluppo edge analytics per digital signage.
