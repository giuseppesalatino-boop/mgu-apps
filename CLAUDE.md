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
  - ID8-C sinistra: **G-meter** (foto 4), **mappa/navigazione** (foto 6) oppure **4 letture** con scala: boost, refrigerante, olio, temperatura esterna (foto 7).
- Struttura: un `config.json` (quote normalizzate) e uno statico pre-renderizzato per ogni layout, in cartelle `id7/`, `id8/`, `custom/`; gli slot in una cartella propria, riusabili tra layout. Le foto di riferimento restano fuori da git.

## 3. Strategia di Rendering (Statico vs Dinamico)
- **Layer Statici (1x all'avvio):** Pre-renderizza una sola volta sfondi, cornici doppie, gradienti interni, bagliori (blur/glow), tacche e numeri fissi per ciascun layout (5 cornici: 2 ID7 + 3 ID8), salvandoli in cache/texture in memoria.
- **Layer Dinamici (30 FPS):** Nel loop live aggiorna e disegna esclusivamente gli elementi mobili o variabili (lancette/indicatori con relativa scia direzionale, valori numerici live, marcia, spie/stati) facendo compositing rapido sopra il layer statico in cache.

## 4. Regole Rigide di Risparmio Token e Iterazione
- **Modifiche chirurgiche:** Non riscrivere mai interi file o moduli già funzionanti. Restituisci esclusivamente la singola funzione, il dizionario di configurazione o il diff mirato da aggiornare.
- **Un componente alla volta:** Quando rifinisci la somiglianza visiva rispetto a una delle 7 immagini, intervieni su un solo elemento per volta (es. profilo esterno -> raccordi -> tacche -> bagliori -> lancetta) senza alterare ciò che è già allineato.
- **Calibrazione rapida:** Mantieni sempre centralizzate le coordinate/quote di ciascun layout e slot in strutture dati pulite e prevedi una modalità di overlay/confronto al 50% sopra l'immagine originale per verificare l'allineamento al pixel.
