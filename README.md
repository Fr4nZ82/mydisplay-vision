# MyDisplay Vision

**mydisplay-vision** è un modulo di visione computerizzata progettato per i device MyDisplay.  
Permette di rilevare persone che passano davanti al display, stimare sesso ed età in modo anonimo, tracciare i volti, conteggiare attraversamenti, e aggregare metriche. L’elaborazione è completamente **on-device**, senza invio di video o immagini al server, per garantire privacy.

---

## Funzionalità

- Rilevamento del volto (face detection)  
- Stima del sesso (male / female / unknown)  
- Stima della fascia d’età (es. 0–13, 14–24, 25–34, …, 65+, unknown)  
- Tracking dei volti con assegnazione di **track_id**  
- Conteggio degli attraversamenti di una “tripwire” (linea virtuale)  
- Aggregazione delle metriche (conteggi, sesso, età) in finestre temporali (es. ogni minuto)  
- Server HTTP locale per esposizione di metriche e debug  
- Visualizzazione su frame annotato (rettangoli, ID, età/sesso)  

---

## Stato attuale & limiti

- La stima di **età e sesso** è attiva e funziona in modo soddisfacente.  
- Il **tracker** (basato su SortLite) funziona, ma non è ancora robusto: spesso assegna un nuovo `track_id` a uno stesso volto, causando duplicati nei conteggi.  
- La tripwire è disegnata per default come linea orizzontale centrale (colore blu nel flusso di debug).  
- Comportamenti da migliorare in future iterazioni sono indicati nella sezione *Da migliorare*.

---

## Architettura / file principali

- `src/main.py` — punto d’ingresso, orchestrazione del servizio  
- `src/runtime.py` — loop principale di acquisizione, elaborazione, classificazione, tracking  
- `src/face_detector.py` — wrapper per il rilevatore di volti (es. YuNet)  
- `src/age_gender.py` — moduli di inferenza per età e sesso (modelli ONNX)  
- `src/tracker.py` — tracker (SortLite) che assegna ID persistenti  
- `src/aggregator.py` — aggregazione degli eventi in finestre temporali  
- `src/api.py` — server HTTP (FastAPI) con endpoint per health, metriche, debug  
- `src/config.py` — gestione configurazione / parametri  
- `config_explained.md` — descrizione dettagliata dei parametri configurabili in config.json

---

## Installazione

1. Clona il repository:

   ```bash
   git clone https://github.com/Fr4nZ82/mydisplay-vision.git
   cd mydisplay-vision
   ```

2. (Consigliato) crea e attiva un ambiente virtuale Python:

   ```bash
   python3 -m venv venv
   source venv/bin/activate       # Linux / macOS
   venv\Scripts\activate          # Windows
   ```

3. Installa le dipendenze:

   ```bash
   pip install -r requirements.txt
   ```

---

## Avvio & uso

Esegui il modulo da riga di comando:

```bash
python src/main.py --config config.json
```

Opzioni utili (override su `config.json`):

- `--camera`: indice della camera o path sorgente video  
- `--width`, `--height`: dimensioni di elaborazione  
- `--target-fps`: frame al secondo target  
- `--help`: visualizza tutte le opzioni disponibili  

Durante l’esecuzione, il servizio:

- apre la webcam (o altra sorgente)  
- elabora i frame per rilevamento, tracking, classificazione  
- disegna rettangoli, ID, età, sesso sui volti  
- traccia attraversamenti della tripwire  
- mantiene contatori aggregati  
- rende disponibili API HTTP per diagnostica, metriche e debug  

---

## API HTTP esposte

- `GET /health` — restituisce stato della camera, FPS, dimensioni frame, uptime  
- `GET /stats` — statistiche sull’esecuzione (frame totali, FPS medio, timestamp ultimo frame)  
- `GET /debug` — pagina con stream MJPEG annotato (e vari sottoroute, es. `/debug/stream`, `/debug/frame`)  
- `GET /metrics/minute` — restituisce gli aggregati al minuto (conteggi totali, suddivisi per età/sesso)  
- `GET /config` — restituisce la configurazione attiva  

---

## Da migliorare / roadmap

- Migliorare la robustezza del tracker per mantenere ID coerenti su un volto tra i frame  
- Ridurre falsi duplicati nei conteggi dovuti al tracker  
- Supporto a sorgenti video alternative (RTSP, file video, videocamere multiple)  
- Ottimizzazione delle prestazioni su hardware limitato  
- Integrazione diretta con **MyDisplayPlayer**, per invio automatico dei dati aggregati  
- Logging migliorato e dashboard visuale locale  
- (Futuro) supporto multipiattaforma, modelli aggiornati, calibrazione automatica  

---

## Note legali & privacy

- Nessun video o immagine viene salvato o inviato al server: tutto avviene localmente.  
- Le persone non sono riconosciute con identità: si usano ID locali temporanei.  
- I dati inviati al server (aggregati) non contengono immagini né informazioni personali identificabili.

---

## Contribuire

Se vuoi contribuire:

1. Fai un fork del repository  
2. Crea un branch (`feature/nome_feature`)  
3. Apporta modifiche, documenta e testa  
4. Apri una pull request  

---

## Licenza

(da specificare — al momento non è definita nel repository)
