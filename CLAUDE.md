# CLAUDE.md — temi cluster fedeli all'OEM per BMW iDrive 7 (cluster-render / Cluster Themes)

Regole per sviluppare **nuovi temi del quadro strumenti** (1920 × 720, Live Cockpit Professional) che
riproducono fedelmente i layout originali iDrive 7, con **sfondo `#000000`** e **30 FPS**.

> **Dove sta il codice.** I temi del cluster vivono in **[bimmer-lab/apps](https://github.com/bimmer-lab/apps)**
> (app `cluster_themes`, motore `shared/`, bench `bench/`, guida `docs/DEVELOPING.md`), non in questo repo
> (`mgu-apps`, che ha le app per il display centrale e il kit vendorizzato). I percorsi qui sotto sono relativi
> alla root di bimmer-lab/apps. Prima di lavorare: leggi `docs/DEVELOPING.md`, `bench/README.md` e
> l'header di `shared/hud.js` (il riferimento di ogni tipo di widget).

## 1. Come è fatto un tema (architettura reale)

- Un tema = **un layout JSON** `apps/cluster_themes/hud/layouts/<id>.json` (`schemaVersion: 2`, `"theme"`
  = una palette di `shared/hud_themes.js`) + di solito **un widget effect "tutto-in-uno"** (uno shader GLSL
  in `shared/shaders.js`) che disegna sfondo, scale, archi, pannelli e bagliori, + **widget `text`/`icon`**
  sopra (numeri delle scale, velocità, marcia, barra inferiore). Esempi: `mview` (M view), `mdash` + `mtach`
  (M), `sdash` (Sport).
- Il motore è `shared/hud.js` (`config.target: "panel"`): crea un layer effect per shader e impacchetta i
  widget immagine in "sheet". Ogni nuovo tipo di widget va: (1) documentato nell'header di `hud.js`,
  (2) validato in `shared/layouts.js`, (3) creato in `_build` e aggiornato in `_tick` di `hud.js`,
  (4) aggiunto al bench se serve, (5) elencato in `app.json` (`settings.hudLayout.options`, previews).
- I layout complessi sono **generati** (`design/gen_*.py`, fuori dal repo): per un tema nuovo scrivi il
  generatore Python accanto agli altri e rigenera il JSON; non ritoccare a mano un JSON generato.
- Ogni dato può mancare: `null` → `--` o elemento nascosto, mai un valore inventato. Segnali solo in
  lettura, dichiarati in `app.json` `signals` (opzionali con `_optional`).

## 2. Vincoli non negoziabili

1. **Sfondo `#000000` puro.** Il ruolo `bg` della palette è `[0, 0, 0, 255]` e lo shader parte da
   `vec4(0.0, 0.0, 0.0, 1.0)`; ogni area "vuota" è nero assoluto (il layer copre i gauge BMW sotto, quindi
   il nero deve essere opaco). Pannelli grigi, bagliori e gradienti stanno *sopra* il nero, mai al suo posto.
   Nessun `vec4` trasparente nelle zone che devono nascondere il cluster BMW, tranne il buco della mappa.
2. **30 FPS.** Oggi `hud.js` aggiorna gli uniform a 20 Hz (`TICK_MS = 50`, condiviso con la HUD) e i testi
   a 5 Hz (`TEXT_MS = 200`, come l'OEM: non toccarlo). Per i 30 FPS del cluster:
   - tick del target `panel` a **33 ms** (costante per target, non cambiare quella della HUD);
   - le animazioni continue dentro lo shader usano `time: true` con **`time_fps: 30`** (opzione dei layer
     effect in `shared/cluster_render_client.js`), mai un clock JS a frequenza più alta;
   - inviare un uniform **solo se il valore quantizzato è cambiato** (pattern `key === x.last` già usato da
     `mview`/`mdash`): a fermo zero messaggi.
   - Budget: lo shader deve passare `node tools/shader_budget.js --all` senza avvisi.
3. **Limiti di cluster-render**: max **8 layer per connessione** (32 totali, `"conn": 1..3` per averne
   altri), **16 MiB** di memoria pixel (ogni immagine conta due volte). Widget effect prima nel file, testi
   dopo. Il log del rebuild deve dire ≤ 8 layer.
4. **Visibilità reale** (`shared/visible.js`): il cofano nasconde gli angoli alti (fino a y 273 a sinistra,
   y 240 a destra); BMW disegna il riquadro nero della marcia a x 1406–1597 y 342–419 e la spia cintura a
   x 587–626 y 663–700. Niente informazioni importanti lì; il check del layout non deve dare avvisi nuovi.
   La lettera della marcia BMW in basso a destra resta scoperta (widget `fill` `strip_cover_l/_r`).
5. **Marchi**: non disegnare il logo BMW né il logo M come grafica nuova; nomi di tema senza "BMW".
   (Il widget `stripe` e il `badge` esistenti sono opzioni dell'utente, off di default: non attivarli
   di default in un tema nuovo.) Nessuna foto OEM, font OEM o bitmap del cluster nel repo: le foto di
   riferimento stanno in `design/photos/` fuori dal repo.
6. **Font**: Barlow pre-cotto in atlanti (`apps/cluster_themes/hud/fonts/*.json`, `tools/bake_fonts.py`);
   un nuovo corpo o glifo si cuoce da lì, mai da un font BMW. L'auto non rasterizza font.
7. **Crash safety**: ogni timer/evento passa dal `_guard` di `hud.js`; una seconda uscita inattesa
   disinstalla l'app sull'auto.

## 3. Fedeltà all'OEM: come misurare

- Si parte dalla **foto di riferimento raddrizzata a 1920 × 720** (come `design/photos/ref-m8-cluster.png`
  per M). Le misure (percorsi delle scale, centri e raggi, angoli, posizioni dei numeri) diventano
  costanti del generatore, con un commento che dice da quale foto e con quale correzione vengono.
- **Geometria prima del colore**: sovrapponi lo screenshot del bench (`?only=1&clean=1`) alla foto al 50%:
  scale, tacche e numeri entro **±2 px**.
- **Scale OEM** (dalle foto): tachimetro 0–330 km/h (numeri ogni 30) / 0–200 mph; contagiri 0–8 ×1000 con
  zona rossa dal setting `redline`; carburante 0 – ½ – 1 con riserva rossa; olio 50 – 100 – 150 °C;
  refrigerante 70 – 100 – 130 °C; boost 0 – 10 – 20 psi. Unità dal setting `units`.
- **Movimento come l'OEM**: lancette e barre scorrono (rpm già filtrato 120 ms in `hud.js`, quantizzazione
  fine: 400–1000 passi); numeri digitali a 5 Hz.
- **Palette**: ruoli in `hud_themes.js` (riusa `mdash` per i look M: `fg [238,241,246]`, `accent [226,36,44]`,
  `chrome`, `grey`, `redTrack`); tara i colori sulla foto, non a occhio.
- Bagliori, vetri e sfumature si fanno **nello shader** (smoothstep, distanze), mai con bitmap.

## 4. Verifica prima di consegnare

```sh
node bench/server.js cluster_themes          # http://localhost:8078, "Manual" per i valori di prova
node tools/shader_budget.js --all            # nessun segno di crash
```

1. Valori di prova: 0 km/h, minimo, metà scala, fondo scala, redline, riserva, olio freddo, dati mancanti.
2. Car view attiva: nessun avviso nuovo di `visible.js` (cofano, riquadro marcia, spia cintura).
3. Log del rebuild: ≤ 8 layer per connessione, memoria pixel sotto il limite.
4. Screenshot headless (`--virtual-time-budget=20000 --window-size=1920,720`) sovrapposto alla foto.
5. Con "Show Waze map inside the theme" il tema apre la finestra mappa e la richiude senza artefatti.
6. CHANGELOG e README dell'app aggiornati ("Not yet tested on a car" finché non è provato sull'auto).
