# Istruzioni di Progetto: Replica 1:1 Cluster BMW iDrive 7 (MGU)

## 1. Contesto e Architettura
- **Repository di riferimento:** `bimmer-lab/apps` (ecosistema MGU App Kit / `cluster-render` per head unit BMW iDrive 7).
- **Target prestazionale:** Rendering fluido a **30 FPS** reali (`< 33.3 ms` per frame).
- **Workflow di sviluppo:** Prima di scrivere nuovo codice, analizza la struttura esistente del repo (`apps/`, `bench/`, `shared/`, `tools/`, `mgu-repo.json`) e sfrutta il banco di prova (`bench/`) per il rendering e la verifica visiva.

## 2. Regola Fondamentale: Fedeltà 1:1 alle 7 Immagini di Riferimento
- L'obiettivo **non** è reinterpretare o stilizzare l'interfaccia in chiave web/UI generica: le **7 immagini di riferimento** fornite dall'utente (vedi sezione 2b) devono essere riprodotte **fedelmente 1:1** per geometria, proporzioni, colori, bagliori, raccordi, spessori e tipografia.
- **Divieto di stime "a occhio":** Non approssimare sagome poligonali smussate (es. quadranti BMW a "parentesi" spezzata, vaschette inferiori, raccordi dei satelliti) con semplici cerchi o archi generici. Usa coordinate normalizzate/esatte, polilinee con raccordi calcolati o asset/maschere pre-renderizzati allineati al pixel sulle immagini originali.
- **Gestione tipografica:** Non sostituire i numeri/etichette con font generici evidenti. Usa asset/sprite pre-renderizzati dai riferimenti o i font tecnici più fedeli previsti dal progetto, calibrando kerning, altezza e spessore per combaciare con le 7 immagini.

## 2b. Categorie, Layout e Slot
L'app raccoglie i layout in tre categorie, scelte dalle impostazioni dell'app (categoria -> layout -> contenuto dello slot):

| Categoria | Layout | Riferimenti |
|---|---|---|
| **ID7** | **Strada**: tachimetro a sinistra, contagiri a destra, mappa al centro. **Corsa**: marcia tra due lame del contagiri, refrigerante/boost a sinistra, pneumatici a destra | foto 1, 2 |
| **ID8** | **A**: tachimetro blu e contagiri rosso "a parentesi" ai lati, velocità e marcia, slot centrale. **B**: esagono blu centrale con marcia e velocità, tachimetro blu a sinistra, contagiri a rampa rossa a destra. **C**: marcia al centro, contagiri a rampa rossa, olio/carburante a destra, slot sinistro | foto 3 (A), 5 (B), 4/6/7 (C) |
| **Custom** | da definire in seguito | — |

- **Slot** (contenuti intercambiabili, moduli separati che non ridisegnano la cornice):
  - ID8-A centro: **mappa Waze** oppure **pressione turbo - potenza - coppia** (al posto dell'auto con la strada della foto 3), quest'ultimo nello **stile delle letture della foto 7** (immagine di dettaglio fornita dall'utente): icona a sinistra, tre valori di scala (minimo, medio, massimo) in cifre squadrate bianche, unità a destra, sotto una barra a doppia linea divisa in 4 segmenti con tagli obliqui, riempita fino al valore, con zona rossa in fondo scala; pannello blu con bande diagonali.
    - **Fondoscala impostabile dall'app** per ciascuna delle tre righe; le tacche numerate si ricavano da esso: **0 - metà - fondoscala** (es. turbo 3 bar -> 0 - 1.5 - 3). Intervalli ammessi del fondoscala: **potenza 150-640 kW**, **coppia 400-1000 Nm**; turbo solo **1.6 / 2 / 2.6 / 3 bar** (tacche 0 - 0.8 - 1.6, 0 - 1 - 2, 0 - 1.3 - 2.6, 0 - 1.5 - 3: sempre un solo decimale), nessun altro valore.
    - **Unità** scelte dall'app: potenza **kW / PS / HP** (PS = cavallo metrico, 1 kW = 1,3596 PS; HP = horsepower, 1 kW = 1,341 HP); coppia **Nm / kgm / lb-ft** (1 Nm = 0,10197 kgm = 0,73756 lb-ft).
    - **Numeri netti:** il fondoscala, anche dopo la conversione, è un intero **divisibile per 4** (arrotondato per eccesso), così i 4 segmenti della barra e la tacca centrale valgono numeri interi (es. 640 kW -> 870,1 PS -> **872**: 0 - 436 - 872; 1000 Nm -> 737,6 lb-ft -> **740**: 0 - 370 - 740).
  - **Contagiri ID8:** scala **8** (benzina, 0-8) o **6** (diesel, 0-6, 7 tacche equidistanti sulla stessa altezza). Zone che seguono la temperatura dell'olio (layer dinamico): **benzina** olio **sotto 60 °C** tutto rosso da 4000; **60-75 °C** giallo da 4000 e rosso da 5000; **da 75 °C** giallo da 6500 e rosso impostabile dall'app tra 7000 e 7500; **diesel** sotto 60 °C rosso da 3000; 60-75 °C giallo da 3000 e rosso da 4000; da 75 °C giallo da 4500 e rosso da 5000 (a gradini, `config.json` `rev_zones`).
  - **Carburante ID8:** la barra diventa **gialla in riserva** e **rossa quasi a secco**, con il criterio scelto dall'app: **litri** (giallo a 8 L, rosso a 4 L) oppure **autonomia** (giallo a 90 km, rosso a 50 km); entrambi da `KombiIF statusRangeTankStatus`.
  - ID8-C sinistra: **G-meter** (foto 4), **mappa/navigazione** (foto 6) oppure **4 letture** con scala: boost, refrigerante, olio, temperatura esterna (foto 7).
- Struttura: un `config.json` (quote normalizzate) e uno statico pre-renderizzato per ogni layout, in cartelle `id7/`, `id8/`, `custom/`; gli slot in una cartella propria, riusabili tra layout. Le foto di riferimento restano fuori da git.

## 3. Strategia di Rendering (Statico vs Dinamico)
- **Layer Statici (1x all'avvio):** Pre-renderizza una sola volta sfondi, cornici doppie, gradienti interni, bagliori (blur/glow), tacche e numeri fissi per ciascun layout (5 cornici: 2 ID7 + 3 ID8), salvandoli in cache/texture in memoria.
- **Layer Dinamici (30 FPS):** Nel loop live aggiorna e disegna esclusivamente gli elementi mobili o variabili (lancette/indicatori con relativa scia direzionale, valori numerici live, marcia, spie/stati) facendo compositing rapido sopra il layer statico in cache.

## 4. Regole Rigide di Risparmio Token e Iterazione
- **Modifiche chirurgiche:** Non riscrivere mai interi file o moduli già funzionanti. Restituisci esclusivamente la singola funzione, il dizionario di configurazione o il diff mirato da aggiornare.
- **Un componente alla volta:** Quando rifinisci la somiglianza visiva rispetto a una delle 7 immagini, intervieni su un solo elemento per volta (es. profilo esterno -> raccordi -> tacche -> bagliori -> lancetta) senza alterare ciò che è già allineato.
- **Calibrazione rapida:** Mantieni sempre centralizzate le coordinate/quote di ciascun layout e slot in strutture dati pulite e prevedi una modalità di overlay/confronto al 50% sopra l'immagine originale per verificare l'allineamento al pixel.
