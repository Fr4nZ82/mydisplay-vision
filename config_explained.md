# MyDisplay Vision – config_explained.md (complete)

Questo documento elenca **tutte le proprietà** utilizzabili in `config.json`, con spiegazioni pratiche su cosa fanno e quando modificarle.

---

## 🎥 Capture / Stream

### camera
Indice della webcam da aprire (0 = predefinita).  
Imposta 1, 2, ... se hai più dispositivi. Se assente, il software tenta 0 come default.

### width / height
Risoluzione richiesta alla camera (pixel).  
Più alta = più qualità ma più carico (più memoria, più lavoro per detector/classifier).

### target_fps
FPS **desiderati** per cattura+pipeline.  
Abbassalo per ridurre carico CPU, alzalo per tracking più fluido (se la macchina regge).

### debug_enabled
Abilita/disabilita la pagina `/debug` (HTML + stream MJPEG).  
Utile disattivarlo su host headless o quando non serve osservare il video.

### debug_stream_fps
Frequenza dei frame nello stream MJPEG (non influisce sul loop interno).  
Valori bassi riducono banda e CPU lato server/client.

### debug_resize_width
Larghezza del frame nella pagina `/debug` (mantiene aspect ratio).  
0 o negativo = nessun resize. Ridurlo accelera encoding JPEG e render nel browser.

---

## 🌐 API

### host / port
Indirizzo e porta del server FastAPI.  
`host: "127.0.0.1"` per accesso locale; usa `"0.0.0.0"` per esporre nella LAN.

---

## 🔎 Rilevatore Volti (YuNet)

### detector_resize_width
Ridimensiona **solo per la detection** (non tocca lo stream di debug).  
Valori tipici 480–640: riduce molto il carico mantenendo buona qualità su volti medi.

### yunet.onnx_path
Percorso del modello ONNX per YuNet face detector.  
Se mancante o invalido, la detection è disabilitata → vedrai solo il video senza riquadri.

### yunet.score_th
Soglia minima di confidenza (0–1) per accettare un volto.  
Alzala per ridurre falsi positivi; abbassala per cogliere volti lontani/difficili.

### yunet.nms_th
Soglia IoU per la Non‑Maximum Suppression.  
Bassa = elimina più box sovrapposti; alta = conserva più box vicini.

### yunet.top_k
Numero massimo di box gestiti a valle del detector.  
Aumenta solo in scene molto affollate; impatta la latenza.

---

## 🧠 Classificatore Età/Genere (ONNX)

Hai due modalità: **modelli separati** o **modello combinato**.

### (Modelli separati) age_model_path / gender_model_path
Percorsi dei modelli ONNX indipendenti per età e genere.  
Se non presenti o non caricabili, si usa il fallback: etichette `unknown` e confidenza `0.0`.

### age_buckets
Elenco delle fasce d’età in output (stringhe mostrate in overlay e metriche).  
Adattale alla tassonomia aziendale (es. 0‑13, 14‑24, ... 65+).

### cls_min_face_px
Lato minimo (pixel) del volto per tentare la classificazione.  
Evita inferenze su volti troppo piccoli (rumorosi) e risparmia CPU.

### cls_min_conf
Confidenza minima per **genere** (sotto soglia → `unknown`).  
Utile per essere conservativi in condizioni di luce/angolo difficili.

### cls_interval_ms
Intervallo minimo tra due inferenze sullo **stesso track** (caching).  
Riduce il carico quando una persona resta in scena per più secondi.

---

## 🧠 Classificatore Combinato (consigliato)

### combined_model_path
Percorso del modello ONNX che predice **età e genere insieme** (es. Intel retail‑0013, InsightFace genderage.onnx).  
Se presente, ha priorità sui modelli separati (più veloce, una sola inferenza).

### combined_input_size
Dimensione di input del volto per il modello combinato, es. `[62, 62]` (Intel) o `[96, 96]` (InsightFace).  
Deve corrispondere a quanto atteso dal modello.

### combined_bgr_input
Se `true`, il modello attende BGR (tipico OpenCV, Intel/InsightFace); se `false`, attende RGB.  
Consente di evitare conversioni inutili e piccoli errori di colore.

### combined_scale01
Se `true`, i pixel sono scalati a `[0..1]`; se `false`, si lasciano a `[0..255]`.  
Intel retail‑0013 lavora correttamente con 0..255.

### combined_age_scale
Fattore per riportare l’età alla scala reale quando l’output è **normalizzato** (es. `age_norm * 100`).  
Intel e InsightFace usano spesso età/100 → lascia 100.0.

### combined_gender_order
Ordine delle classi nell’output di genere del modello combinato, tipicamente `["female","male"]`.  
Serve per interpretare correttamente il vettore probabilità/logit a 2 elementi.

---

## 🧭 Tracker (ID stabili)

Il **tracker** mantiene un ID coerente per ogni volto rilevato tra più frame consecutivi.  
Ogni persona rilevata diventa un **track**, cioè un oggetto con un numero identificativo (`track_id`) che rimane stabile mentre quella persona è visibile.

Il tracker associa le nuove detection (YuNet) con i track già esistenti confrontando la **IoU (Intersection over Union)** tra riquadri.  
Un’alta IoU significa che i box si sovrappongono molto → probabilmente è lo stesso volto.  
Un’IoU bassa → volto diverso o spostamento troppo ampio.

### tracker_max_age
Numero massimo di frame consecutivi senza corrispondenza prima che un track venga eliminato.  
Serve per tollerare “buchi” momentanei della detection (es. occlusioni, blur, cambi luce).  
- Valore basso (es. 5): più reattivo, ma perde ID se il volto sparisce per un attimo.  
- Valore alto (es. 20–30): mantiene l’ID anche se il volto manca per qualche frame.  
Utile quando si vogliono ID persistenti o quando YuNet ha piccoli drop.

### tracker_min_hits
Numero minimo di frame consecutivi necessari prima che un track venga considerato “valido”.  
Serve a filtrare detections casuali o falsi positivi.  
- Valore basso (1–2): crea subito ID, ma genera più falsi positivi.  
- Valore alto (3–5): più stabile, ma introduce ritardo nel riconoscimento iniziale.

### tracker_iou_th
Soglia di **IoU** (0–1) per considerare una detection come la stessa persona di un track precedente.  
- Più bassa (0.2): più permissiva → mantiene ID anche con spostamenti ampi.  
- Più alta (0.6): più rigida → può perdere ID se il volto si muove molto.  
Valori comuni: **0.3–0.4**.  
Un buon equilibrio evita di confondere persone vicine, ma mantiene l’ID coerente.

---

## 🚶 ROI / Tripwire

La **tripwire** è una linea virtuale tracciata nell’immagine per rilevare **attraversamenti** (entrate/uscite).  
Quando il **centro del volto** di un track attraversa questa linea, viene registrato un evento `CROSS` nel sistema di aggregazione.  
È il principio base del conteggio di passaggi.

### Come funziona
- La linea è definita da due punti normalizzati `[[x1, y1], [x2, y2]]`, dove 0..1 rappresentano la proporzione rispetto alla larghezza/altezza del frame.  
- Il tracker salva la posizione del centro del volto (prev e curr).  
- Se la linea viene attraversata, viene emesso un evento con direzione `a2b` o `b2a`.  
- Ogni evento include il `track_id`, genere, fascia d’età e direzione.

### roi_tripwire
Coppia di punti normalizzati `[ [x1, y1], [x2, y2] ]` che rappresentano la linea A→B.  
Esempi:
- `[ [0.1, 0.5], [0.9, 0.5] ]` → linea orizzontale al centro, da sinistra a destra.  
- `[ [0.5, 0.2], [0.5, 0.8] ]` → linea verticale al centro, dall’alto in basso.  
Adatta questi valori per posizionare la linea sopra soglie, porte, ingressi, corridoi, ecc.

### roi_direction
Direzione di conteggio consentita:
- `"both"` → conta in entrambe le direzioni (entrate + uscite).  
- `"a2b"` → conta solo se il movimento è da A verso B.  
- `"b2a"` → conta solo da B verso A.  
Imposta `"a2b"` o `"b2a"` per contare solo entrate o solo uscite (utile per varchi unidirezionali).

### roi_band_px
Spessore della **banda** attorno alla tripwire (in pixel).  
È la tolleranza verticale/orizzontale per il rilevamento dell’attraversamento.  
- Più basso (5): linea sottile → preciso ma sensibile al jitter.  
- Più alto (20): fascia larga → più robusta ma può contare passaggi laterali.  
In pratica, la tripwire non è una linea infinitamente sottile, ma una fascia che riconosce attraversamenti con margine.

---

## Re-Identification (memoria facce)

### reid_enabled
Abilita o disabilita il riconoscimento ricorrente (Re-ID) dei volti.
Quando true, il sistema calcola un’impronta (embedding) del volto e prova a riconoscerlo se riappare dopo essere uscito dall’inquadratura.
Se il modello face_recognition_sface_2021dec.onnx non è presente o OpenCV non supporta SFace, la funzione viene ignorata automaticamente.

## reid_model_path
Percorso del modello ONNX per il riconoscimento facciale (embedding) usato dal Re-ID.
È consigliato usare il modello Intel SFace incluso in OpenCV contrib:
models/face_recognition_sface_2021dec.onnx.

## reid_similarity_th
Soglia di similarità (cosine similarity) tra due volti per considerarli la stessa persona.
- Valori più alti → meno falsi positivi, ma rischio di non riconoscere la stessa persona con illuminazione diversa.
- Valori più bassi → più tolleranza, ma possibili accoppiamenti errati.
Esempio: 0.35 – 0.40 è un buon punto di partenza per SFace.

## reid_cache_size
Numero massimo di volti memorizzabili nella cache del Re-ID.
Ogni voce contiene embedding e timestamp dell’ultima volta che la persona è stata vista.
Quando il limite è superato, vengono eliminati i più vecchi.

## reid_memory_ttl_sec
Durata della “memoria” del Re-ID, in secondi.
Se una persona non viene più vista per più di questo intervallo, la sua impronta viene rimossa e, al successivo ingresso, riceverà un nuovo ID.
Esempio: 600 = 10 minuti.

## count_dedup_ttl_sec
Intervallo minimo (in secondi) prima che una stessa persona possa essere nuovamente conteggiata dopo un attraversamento della tripwire.
Serve a evitare doppi conteggi per chi rientra subito nell’inquadratura o passa più volte davanti al display in poco tempo.
Esempio: 600 = 10 minuti di blocco conteggio per la stessa persona.

---

## 📊 Metriche / Aggregazione

Ogni attraversamento (evento CROSS) viene inviato al modulo `MinuteAggregator`.  
L’aggregatore accumula eventi per finestre temporali regolari, permettendo di analizzare flussi di persone nel tempo.

### metrics_window_sec
Durata, in secondi, della finestra temporale per raggruppare eventi.  
Es.: 60 = conteggio per minuto.  
Ogni finestra produce un record con i totali per sesso e fascia d’età.  
Finestra più lunga (es. 300s) → dati più stabili ma meno reattivi.

### metrics_retention_min
Tempo di retention dei record di aggregazione (in minuti reali).  
Definisce quanto a lungo le statistiche rimangono in memoria prima di essere scartate.  
Esempio: `metrics_window_sec=60`, `metrics_retention_min=120` → memorizza gli ultimi 120 minuti di dati.  
Aumenta per report storici più lunghi o interrogazioni più ampie su `/metrics/minute?last=N`.

---

## 🔍 Relazione tra tracker e tripwire

1. **YuNet** rileva un volto nel frame.  
2. **Tracker** assegna o aggiorna un `track_id`.  
3. Ogni `track_id` conserva la posizione del volto (centro del box).  
4. Quando il centro attraversa la **tripwire (A→B o B→A)**, viene generato un evento `CROSS`.  
5. L’evento alimenta l’**aggregatore**, che aggiorna i contatori per il minuto in corso.  
6. Le statistiche sono consultabili via `/metrics/minute`.

In sintesi, il tracker mantiene la coerenza nel tempo, la tripwire trasforma il movimento in eventi, e l’aggregatore li converte in dati analitici aggregati.

---

## 🔗 Suggerimenti rapidi

- **Performance:** abbassa `detector_resize_width`, `target_fps`, alza `cls_interval_ms`.  
- **Precisione:** alza `yunet.score_th`, riduci `tracker_iou_th`, tieni `cls_min_face_px` ≥ 64.  
- **Combinato Intel:** `combined_input_size=[62,62]`, `combined_bgr_input=true`, `combined_scale01=false`, `combined_age_scale=100.0`, `combined_gender_order=["female","male"]`.  
- **Combinato InsightFace:** `combined_input_size=[96,96]`, `combined_bgr_input=true` (o `false` se il repo specifica RGB), `combined_scale01=false`, `combined_age_scale=100.0`.
