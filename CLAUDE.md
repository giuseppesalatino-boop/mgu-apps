# Istruzioni di Progetto: Replica 1:1 Cluster BMW iDrive 7 (MGU)

## 1. Contesto e Architettura
- **Repository di riferimento:** `bimmer-lab/apps` (ecosistema MGU App Kit / `cluster-render` per head unit BMW iDrive 7).
- **Target prestazionale:** Rendering fluido a **30 FPS** reali (`< 33.3 ms` per frame).
- **Workflow di sviluppo:** Prima di scrivere nuovo codice, analizza la struttura esistente del repo (`apps/`, `bench/`, `shared/`, `tools/`, `mgu-repo.json`) e sfrutta il banco di prova (`bench/`) per il rendering e la verifica visiva.

## 2. Regola Fondamentale: Fedeltà 1:1 alle 5 Immagini di Riferimento
- L'obiettivo **non** è reinterpretare o stilizzare l'interfaccia in chiave web/UI generica: le **5 immagini di riferimento** fornite dall'utente (i 5 layout/stati del cluster) devono essere riprodotte **fedelmente 1:1** per geometria, proporzioni, colori, bagliori, raccordi, spessori e tipografia.
- **Divieto di stime "a occhio":** Non approssimare sagome poligonali smussate (es. quadranti BMW a "parentesi" spezzata, vaschette inferiori, raccordi dei satelliti) con semplici cerchi o archi generici. Usa coordinate normalizzate/esatte, polilinee con raccordi calcolati o asset/maschere pre-renderizzati allineati al pixel sulle immagini originali.
- **Gestione tipografica:** Non sostituire i numeri/etichette con font generici evidenti. Usa asset/sprite pre-renderizzati dai riferimenti o i font tecnici più fedeli previsti dal progetto, calibrando kerning, altezza e spessore per combaciare con le 5 immagini.

## 3. Strategia di Rendering (Statico vs Dinamico)
- **Layer Statici (1x all'avvio):** Pre-renderizza una sola volta sfondi, cornici doppie, gradienti interni, bagliori (blur/glow), tacche e numeri fissi per ciascuno dei 5 layout, salvandoli in cache/texture in memoria.
- **Layer Dinamici (30 FPS):** Nel loop live aggiorna e disegna esclusivamente gli elementi mobili o variabili (lancette/indicatori con relativa scia direzionale, valori numerici live, marcia, spie/stati) facendo compositing rapido sopra il layer statico in cache.

## 4. Regole Rigide di Risparmio Token e Iterazione
- **Modifiche chirurgiche:** Non riscrivere mai interi file o moduli già funzionanti. Restituisci esclusivamente la singola funzione, il dizionario di configurazione o il diff mirato da aggiornare.
- **Un componente alla volta:** Quando rifinisci la somiglianza visiva rispetto a una delle 5 immagini, intervieni su un solo elemento per volta (es. profilo esterno -> raccordi -> tacche -> bagliori -> lancetta) senza alterare ciò che è già allineato.
- **Calibrazione rapida:** Mantieni sempre centralizzate le coordinate/quote di ciascuno dei 5 layout in strutture dati pulite e prevedi una modalità di overlay/confronto al 50% sopra l'immagine originale per verificare l'allineamento al pixel.
