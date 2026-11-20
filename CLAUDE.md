# CLAUDE.md — temi cluster fedeli all'OEM per BMW iDrive 7 (cluster-render / MGU App Kit)

Regole per sviluppare **nuovi temi del quadro strumenti** che riproducono il più fedelmente possibile
i layout originali BMW iDrive 7 (Live Cockpit Professional, 12,3", 1920 × 720), disegnati con il
renderer GPU del kit (`vendor/idrive7-apps/kit/renderd`, record.js / gauges.js) o con cluster-render.

> **Stato del repository.** Questo repo (`mgu-apps`) contiene le app per il display centrale e il kit
> vendorizzato. Il codice di **cluster-render** (`apps/`, `bench/`, `docs/DEVELOPING.md`, il tema "M"
> esistente) **non è qui**. Prima di scrivere un tema, leggi quei file nel loro repository e aggiorna
> le sezioni marcate *[da confermare]* con i nomi reali di API, file e comandi.

## 1. Vincoli non negoziabili

1. **Sfondo `#000000` puro**, sempre: il `clear` di ogni frame e ogni area "vuota" del cluster sono
   nero assoluto (sul pannello del cluster il nero = pixel spento, come l'OEM). Non usare i `bg` dei
   temi di `kit/runtime/themes.js` (`#050506`, `#04060a`, …): sono per il display centrale.
   Gradienti e pannelli grigi (es. le "conchiglie" del layout Sport) stanno *sopra* il nero, mai al
   posto del nero.
2. **30 FPS fissi**: limita il renderer con `gpu.fps(30)` (mgu-render, vedi
   `vendor/idrive7-apps/kit/renderd/README.md`, "Using it in an app") o con il cap equivalente di
   cluster-render *[da confermare]*. Il budget è 33,3 ms a frame: il disegno JS deve restare sotto
   ~8 ms, niente allocazioni per frame (riusa Recorder, array e stringhe), niente `textWidth` ricalcolato
   per testi statici.
3. **Movimento come l'OEM**: lancette e barre scorrono con `smooth()` di `gauges.js` (molla
   criticamente smorzata); i numeri digitali cambiano al massimo ogni 200 ms con `hold()` (velocità,
   come il cluster BMW). Mai saltare di valore a 30 FPS.
4. **Nessun marchio BMW disegnato**: niente logo M a tre strisce, logo BMW o nomi di modello come
   grafica (vedi il commit "theme names without BMW's M" e il Disclaimer del README). Al posto del
   logo M usa un elemento neutro (tre barrette di colore generico o spazio vuoto). Nomi dei temi
   senza "M" / "BMW" (es. "Sport", "Track", "Classic").
5. **Sola lettura**: i valori arrivano dai segnali che l'auto già pubblica (`kit/runtime/signals.js`,
   `car.js`); un tema non scrive mai su una centralina.
6. **Nessun software o asset BMW nel repo**: niente screenshot OEM, font OEM estratti, bitmap del
   cluster. Le immagini di riferimento restano fuori dal repo (o in una cartella ignorata da git);
   la grafica si ricostruisce in primitive vettoriali.

## 2. Fedeltà all'OEM: come misurare

- **Si parte dalla foto di riferimento**, raddrizzata e scalata a 1920 × 720. Ogni elemento ha una
  tabella di misure (centro, raggio, angoli d'inizio/fine, spessori, posizione dei numeri) scritta
  come costanti in cima al file del tema, con un commento che dice da quale riferimento viene.
- **Geometria prima del colore**: archi, scale e tacche devono coincidere con la foto entro ±2 px
  sovrapponendo il render del bench alla foto al 50% di opacità.
- **Scale OEM**: tachimetro 0–330 km/h (numeri ogni 30, tacca lunga ogni 30, corta ogni 10);
  contagiri 0–8 ×1000 1/min con zona rossa da regime della vettura; carburante 0 – 1/2 – 1 con
  riserva rossa; olio 50 – 100 – 150 °C (blu sotto 50, rosso sopra ~140); refrigerante 70 – 100 – 130 °C;
  boost 0 – 10 – 20 psi. Le unità seguono le impostazioni dell'auto (km/h ↔ mph, °C ↔ °F, psi ↔ bar).
- **Tipografia**: numeri in stile DIN-condensato bianco, unità più piccole e più chiare
  (`km/h`, `1/min x1000`, `°C`). Usa i font cotti del kit (`kit/runtime/fonts/num.json`, `bold.json`,
  `light.json`, `gear.json`); se serve un glifo mancante, aggiungilo con `kit/tools/bake_fonts.py` da
  un font con licenza libera, mai da un font BMW.
- **Palette OEM** (punto di partenza, da tarare sulla foto): bianco scala `#ffffff`, tacche secondarie
  `#bfc3c8`, pannelli `#2a2d33`→`#15171b`, rosso M/redline `#e1001a`, bagliore rosso `#ff1f2d` al 35–60%
  di alfa, blu freddo `#2f7bff`, verde spie `#3ddc4a`, ambra `#ffb31a`.
- **Bagliori e sfumature** (il rosso alla base delle scale, l'alone sotto la lancetta) si fanno con
  `rampRect` / `arc` a più passaggi con alfa decrescente, mai con bitmap.

## 3. Struttura di un tema *[da confermare con docs/DEVELOPING.md]*

- Un tema = un modulo che esporta `{ id, name, layout, palette, draw(rec, state, t) }`.
  `draw` riceve un Recorder (`kit/runtime/record.js`) già pulito con `#000000` e lo stato del veicolo
  già normalizzato (velocità, giri, marcia, modalità di guida, carburante, temperature, TPMS,
  autonomia, ora, temperatura esterna, limite di velocità, nome della via).
- Costanti di layout separate dal codice di disegno; nessun numero "magico" dentro `draw`.
- Elementi riutilizzabili (scala ad arco, barra contagiri curva, mini-gauge carburante/olio,
  widget TPMS, cartello limite, barra di stato inferiore) vanno in moduli condivisi, non copiati
  tra temi.
- Ogni dato può mancare: `null` → `--` o elemento nascosto, mai un valore inventato.

## 4. Verifica prima di consegnare

1. Bench / emulatore *[comando da confermare in `bench/README.md`]*: render a 1920 × 720 con valori di
   prova (0 km/h, minimo, metà scala, fondo scala, redline, carburante in riserva, olio freddo).
2. Confronto sovrapposto con la foto di riferimento (diff a ±2 px sulla geometria).
3. Log FPS: 30 stabili, tempo per frame nel log del renderer (`display: … fps sent, renderer …`).
4. Test del kit e dell'app interessata (`npm test` nella cartella dell'app; i test del kit sono in
   `vendor/idrive7-apps/kit/tools/test_*.js`).
5. Nessun asset BMW aggiunto (`git status` prima del commit).

## 5. Convenzioni del repository

- Una cartella per app con `app/` (`app.json` + codice), registrata in `mgu-repo.json`.
- Il kit in `vendor/idrive7-apps` è vendorizzato: non modificarlo direttamente; le modifiche vanno in
  `patches/` come per quella esistente.
- Commit in stile `mgu-apps X.Y.Z: <cosa cambia>`; documentazione utente in inglese nei README.
