# stream-ctr — Architektura, metodologia i analiza krytyczna

Dokument opisuje organizację kodu projektu `stream-ctr`, metodologię badawczą jaką realizuje
oraz omawia kluczowe rozwiązania (selekcja cech, detektor dryfu, adaptacyjny ensemble).
W ostatniej części znajduje się analiza tego, **co dzieje się z modelami w momencie
reselekcji cech**, **jak reselekcja jest sprzęgnięta z detektorem dryfu** oraz **lista
zidentyfikowanych błędów logicznych w metodologii**.

---

## 1. Cel projektu i metodologia eksperymentalna

Projekt jest stanowiskiem badawczym dla problemu **klasyfikacji binarnej w strumieniach
danych** (typowy benchmark CTR — Avazu/Criteo — uzupełniony o syntetyczny strumień
Agrawal). Pytanie badawcze:

> Czy adaptacyjny mechanizm **reselekcji cech sprzężony z detektorem dryfu**
> poprawia jakość modeli strumieniowych w porównaniu z (a) brakiem selekcji,
> (b) statyczną selekcją top-K po rozgrzewce i (c) ciągłym online-rerankingiem?

Metodologia jest **prequential** (tzw. *test-then-train*): dla każdej instancji strumienia
model najpierw produkuje predykcję (na potrzeby metryk), a następnie jest na niej trenowany.
Implementacja w `PrequentialEvaluator.run()` realizuje ten schemat dosłownie
(`PrequentialEvaluator.java:121-172`):

```text
for each instance raw in stream:
    filtered  = selector.filter(raw)           # opcjonalnie
    p         = model.predictProbability(filtered)
    metrics.update(p, y)
    drift?    = detector.detect( signal(p, y) ) # 0/1-loss albo log-loss
    if drift: driftHandler.onDrift(t, raw, model)
    model.train(filtered)
    selector.observe(raw, y)
```

Przebieg eksperymentu (`Main.java`) jest dwufazowy:

* **Faza 1 — baseline** (`RunExperiments` + `ExperimentMatrix`): macierz
  `dataset × model × selector` z neutralnym detektorem ADWIN, który
  **mierzy liczbę dryfów, ale na nie nie reaguje** (brak `driftHandler`a, więc dryfty są
  jedynie logowane do CSV).
* **Faza 2 — drift-aware** (`RunDriftAwareExperiments` + `DriftAwareExperimentBuilder`):
  pełny pipeline z reaktywną reselekcją cech (`DriftAwareSelector`) oraz dedykowanym
  ensemblem (`DriftAwareSrpModel`).

Oba przebiegi produkują ten sam zestaw artefaktów (long-format CSV z metrykami w czasie,
podsumowanie końcowe i tabela dryfów — `ResultExporter.java`).

### 1.1 Mierzone metryki

`ExperimentMatrix.defaultMetrics(window)` definiuje pięć równolegle liczonych metryk
(`ExperimentMatrix.java:33-41`):

| Metryka | Implementacja | Sens |
|---|---|---|
| Accuracy (kumulacyjna) | `AccuracyMetric` | klasyfikacja przy progu 0,5 |
| LogLoss (kumulacyjna) | `LogLossMetric` | jakość rozkładu prawdopodobieństw |
| AUC | `AucMetric(10_000)` | Mann–Whitney na ostatnich 10 tys. predykcji |
| Windowed Accuracy | `WindowedMetric(1_000, …)` | reaktywna acc. na 1k najnowszych |
| Windowed LogLoss | `WindowedMetric(1_000, …)` | reaktywny logloss na 1k najnowszych |

`AUC` jest liczony „okresowo” — na buforze cyklicznym o pojemności 10 000 — co daje
wrażliwość lokalną, a nie globalną.

---

## 2. Struktura pakietów i odpowiedzialności

```
stream
├── config/        # ścieżki, opisy datasetów
├── provider/      # adaptery na strumień (ARFF, Agrawal)
├── features/      # selekcja cech (4 strategie) + ranking IG + filtr instancji
├── drift/         # adapter ADWIN + DriftEvent
├── model/         # opakowania na klasyfikatory MOA (HT, HAT, SRP)
├── ensemble/      # DriftAwareSRP + SubspaceManager + WeightManager
├── evaluation/    # prequential, metryki, snapshoty
├── experiment/    # macierz eksperymentów, runner, eksporter wyników
└── Main           # phase1 + phase2 driver
```

Granice modułów są starannie utrzymane: każdy `provider`/`model`/`selector` jest
interfejsem (`StreamProvider`, `StreamModel`, `FeatureSelector`, `DriftDetector`),
a `Experiment` jest rekordem zawierającym jedynie *fabryki* (`Supplier<…>`) — co pozwala
uruchamiać każdy eksperyment od stanu zerowego (świeży model, świeży detektor) bez ryzyka
zanieczyszczenia stanu między przebiegami.

### 2.1 Kluczowe abstrakcje

* `StreamProvider` — strumień instancji (`hasNext/next/getHeader/restart`). Implementacje:
  `ArffStreamProvider` (Avazu/Criteo z plików ARFF) i `AgrawalStreamProvider`
  (syntetyczny dryf — sudden albo gradual przez okno przejściowe).
* `StreamModel` — interfejs modelu strumieniowego: `initialize(header)`,
  `predictProbability(inst)`, `train(inst)`, `reset()`. Implementacje to opakowania na
  klasyfikatory MOA: `HoeffdingTreeModel`, `HatModel`, `SrpModel` oraz własna implementacja
  `DriftAwareSrpModel`.
* `FeatureSelector` — `initialize(header) → filter(inst) → observe(inst, y)`.
  Cztery implementacje są opisane w §3.
* `DriftDetector` — `detect(double signal) → boolean`. Jedyna implementacja:
  `AdwinDriftDetector` (delegacja do `moa.classifiers.core.driftdetection.ADWIN`).
* `Metric` — strumieniowa metryka.

---

## 3. Strategia selekcji cech — cztery rozwiązania

Wszystkie selektory używają tego samego `InfoGainRanker`a (estymacja **Information Gain**
z dyskretyzacją numeryczną na 10 kwantylowych przedziałów —
`InfoGainRanker.condEntropyNumeric`, `InfoGainRanker.java:72-107`).

### 3.1 `NoSelector`
Identyczność — model widzi pełny nagłówek źródła. Punkt referencyjny.

### 3.2 `StaticTopKSelector` (offline-warmup, frozen)
Selektor zbiera pierwsze `warmupSize` instancji do listy, po zakończeniu rozgrzewki liczy IG,
wybiera `topK` najlepszych cech i **już nigdy ich nie zmienia**
(`StaticTopKSelector.observe`).

W `ExperimentMatrix.staticPreWarmed(...)` dokonano interesującego zabiegu: zanim
eksperyment ruszy, otwierany jest osobny strumień, na którym selektor jest *prerozgrzewany*
z `warmupSize` instancji. Dzięki temu model od pierwszej instancji widzi już prawidłowo
przefiltrowany nagłówek (`ExperimentMatrix.java:97-114`). To oznacza, że dla fazy 1
statyczny selektor „wie więcej” niż online — co jest **świadomą decyzją projektową**,
ale wpływa na uczciwość porównania (patrz §7).

### 3.3 `OnlineRankingSelector` (sliding buffer rerank co N)
Buduje bufor o pojemności `bufferSize`. Co `rerankEvery` obserwacji wykonuje pełny ranking
IG na całej zawartości bufora i, jeżeli zbiór top-K się zmienił, ogłasza zmianę przez
`ReinitListener` (`OnlineRankingSelector.java:86-107`).

Domyślnie używany w `ExperimentMatrix` z `bufferSize == warmup == rerankEvery == 5_000`,
więc reranking odbywa się co 5 tys. instancji. Selektor nie wymaga rozgrzewki: zwraca
początkowy zbiór „pierwszych top-K cech wg indeksu” aż do pierwszego rerankingu
(`OnlineRankingSelector.initialize`).

### 3.4 `DriftAwareSelector` (reaktywna reselekcja sterowana dryfem) — kluczowe rozwiązanie

Najważniejszy komponent fazy 2. Logika (`DriftAwareSelector.java`):

* W każdej obserwacji selektor utrzymuje **bufor cykliczny** ostatnich `windowSize`
  instancji (`recentBuffer`).
* Gdy detektor dryfu zgłosi sygnał, `PrequentialEvaluator` wywołuje
  `selector.onDriftDetected()`, który:
  1. wykonuje **snapshot pre-drift** = aktualny `recentBuffer`,
  2. liczy `igBefore` na tym snapshocie,
  3. czyści bufor i przechodzi w stan `awaitingPostDrift`.
* Selektor **kontynuuje obserwację** kolejnych instancji, aż zbierze nowe
  `windowSize` próbek.
* Po zebraniu pełnego okna post-drift wywołuje `adaptFeatures()`:
  ranking `igAfter` → `delta = igAfter − igBefore` → wyznaczenie nowego top-K
  z domieszką cech, których IG wzrosło o więcej niż `changeThreshold`,
  i odrzuceniem cech, których IG spadło o więcej niż `changeThreshold`
  (`DriftAwareSelector.java:139-196`).
* Wynik publikowany przez `AdaptationListener` zawiera: nowy nagłówek, zbiór wybranych,
  zbiór `removed`, zbiór `added`, mapę `delta` IG. To listener decyduje, co zrobić z
  modelem (patrz §5).

Innymi słowy — jest to **mechanizm porównawczy „przed/po” dryfcie**, sterowany zewnętrznie
przez ADWIN. Rozwiązanie jest znacznie ograniczone w stosunku do online rerankingu
(selektor nie zmienia cech bez sygnału dryfu) i jednocześnie reaktywne (pełna podmiana
zestawu cech zamiast inkrementalnego dostrajania).

---

## 4. Detekcja dryfu

Detektorem jest jedna instancja **ADWIN** (`AdwinDriftDetector`, delegacja do MOA;
`AdwinDriftDetector.java`). Sygnał wejściowy konfigurowany jest w
`PrequentialEvaluator.computeDriftSignal`:

* domyślnie 0/1-loss `(yHat == y) ? 0.0 : 1.0`,
* opcjonalnie log-loss przy `driftSignalLogLoss(true)` (w obecnej macierzy nie używane).

ADWIN sam decyduje o cięciu okna; po `detect() == true` wewnętrzne okno detektora jest
przycinane. Detektor nigdy nie jest jawnie resetowany pomiędzy dryftami — jedyny `reset()`
następuje w `PrequentialEvaluator.run` przed startem.

**Faza 1**: detektor zlicza dryfy, ale `driftHandler` nie jest ustawiony, więc nic się
nie dzieje poza zapisem do `driftEvents`. Pozwala to zmierzyć „ile by było alarmów”.

**Faza 2**: w `DriftAwareExperimentBuilder` `driftHandler` przekazuje sygnał do
`DriftAwareSelector.onDriftDetected()`, ale dopiero **po okresie warmupu**
(2 000 instancji dla Avazu/Criteo, 5 000 dla Agrawal — `Spec.warmupInstances`).
Warmup zapobiega fałszywym alarmom z niestabilnego modelu na samym początku strumienia.

---

## 5. Co dzieje się z modelami w chwili reselekcji cech

Tu logika rozdziela się **na dwie zupełnie różne ścieżki** w zależności od typu modelu —
i jest to najistotniejszy element całej metodologii.

### 5.1 Modele monolityczne (`HoeffdingTreeModel`, `HatModel`) + `DriftAwareSelector`

Listener w `DriftAwareExperimentBuilder.runDriftAwareSelectorWithModel`
(`DriftAwareExperimentBuilder.java:51-56`):

```java
selector.withListener((idx, newHeader, sel, removed, added, delta) -> {
    System.out.println("  [adapt] @" + idx + " removed=" + removed + " added=" + added);
    model.reset();
    model.initialize(newHeader);
});
```

**Skutek:** w momencie zmiany top-K cały model jest *zerowany*. `HoeffdingTree.reset()`
buduje nowe drzewo od zera (`HoeffdingTreeModel.buildClassifier`); cała wiedza zebrana od
początku strumienia (lub od poprzedniej adaptacji) zostaje **utracona**. Po
`model.initialize(newHeader)` model startuje od pustego drzewa z nowym schematem cech.

Jest to bardzo agresywne. W praktyce dla każdego dryfu HT/HAT zaczyna trening niemal od
zera i przez kolejne tysiące instancji jest niedouczony. Częste fałszywe alarmy ADWIN-a
→ częste restarty → permanentnie niedouczony model.

### 5.2 Ensemble adaptacyjny (`DriftAwareSrpModel`) + `DriftAwareSelector`

Listener kieruje sygnał do dedykowanej, *granularnej* logiki adaptacji
(`DriftAwareExperimentBuilder.java:102-106` → `DriftAwareSrpModel.onDriftDetected`):

```java
selector.withListener((idx, newHeader, sel, removed, added, delta) -> {
    dasrp.onDriftDetected(removed, added);   // weak=removed, strong=added
});
```

`DriftAwareSrpModel.onDriftDetected(weakFeatures, strongFeatures)` realizuje:

1. **Cooldown** — jeżeli od poprzedniej adaptacji minęło mniej niż `cooldownInstances`
   instancji, adaptacja jest pomijana (`DriftAwareSrpModel.java:134-141`).
2. **Adaptacja podprzestrzeni** — `SubspaceManager.adaptSubspaces(weakFeatures,
   strongFeatures, minOverlap, resetRatio)` modyfikuje wektory cech indywidualnie dla
   każdego z `ensembleSize` modeli (`SubspaceManager.java:67-107`):
   * Dla każdego modelu liczone jest `overlap = subspace ∩ weakFeatures`.
   * Jeśli `|overlap| < minOverlap` — model jest pomijany (jego subspace się nie zmienia).
   * Jeśli `|overlap| ≥ minOverlap` — z subspace’u są usuwane cechy słabe i dolosowywane
     z puli `strongFeatures` (z fallbackiem na losowe cechy, jeśli puli mocnych jest za mało).
   * Jeśli `overlap.size() / oldSub.size() ≥ resetRatio` (domyślnie 0,5), ten model trafia
     na listę `modelsToReset`.
3. **Reset wybranych modeli** — dla indeksów z `modelsToReset` budowany jest świeży model
   (`baseFactory.get()`) zainicjalizowany nowym, podstawionym podschematem cech
   (`DriftAwareSrpModel.java:150-158`). Pozostałe modele zostają **w stanie wytrenowanym**.
4. **Zarządzanie wagami** — dla zresetowanych modeli `WeightManager.onModelsReset(...)`
   ustawia ich wagę na `resetWeight` (domyślnie 0,3); `WeightManager.decay()` na każdej
   kolejnej instancji liniowo zwiększa ją o `decayRate` (0,001 dla CTR, 0,005 dla Agrawal),
   aż dojdzie do `normalWeight` (1,0). Predykcja zespołu jest średnią ważoną
   prawdopodobieństw poszczególnych modeli (`WeightManager.weightedPrediction`).

W efekcie ensemble **częściowo zachowuje wiedzę** mimo dryfu — modele słabo nakładające
się ze zbiorem cech słabych zostają nietknięte i mogą stabilizować predykcję, podczas gdy
zresetowane elementy ensemble’u uczą się od zera, ale ich głos jest tłumiony przez
mechanizm wag.

### 5.3 Krótkie zestawienie obu ścieżek

| Aspekt | HT/HAT + DA-Selector | DASRP + DA-Selector |
|---|---|---|
| Granularność reakcji | całość modelu | per-tree w ensemble’u |
| Utrata wiedzy | totalna przy każdej adaptacji | częściowa, sterowana `resetRatio` |
| Tłumienie świeżych modeli | brak (od razu pełna waga) | `resetWeight=0.3` + liniowe odbudowywanie |
| Sygnał wejściowy do adaptacji | nowy nagłówek (zbiór po reranku) | `(removed, added)` — różnica top-K |
| Cooldown adaptacji | brak | `cooldownInstances` (domyślnie 0, ale konfigurowalne) |

---

## 6. Sprzężenie reselekcji cech z detektorem dryfu (faza 2)

Sekwencja zdarzeń w pełnym pipeline’ie (DASRP):

```
[t]  raw = ObservingProvider.next()         # ObservingProvider także robi selector.observe(raw, y)
     filtered = raw                         # PrequentialEvaluator nie ma własnego selektora
     p = dasrp.predictProbability(raw)      # dasrp samodzielnie filtruje przez sortedSubspaces
     metrics.update(p, y)
     signal = (yHat == y) ? 0 : 1
     drift = adwin.detect(signal)
     if drift and t >= warmup:
         selector.onDriftDetected()         # snapshot pre-drift
         (selektor wchodzi w awaitingPostDrift)
     dasrp.train(raw)
     ...
[t+1..t+windowSize]  selektor zbiera okno post-drift
[t+windowSize]       selector.adaptFeatures()  → wywołuje listener:
                     dasrp.onDriftDetected(removed, added)
                     - SubspaceManager.adaptSubspaces (per tree swap)
                     - selektywny model reset
                     - WeightManager.onModelsReset → wagi 0.3 dla resetowanych
[t+windowSize+1...]  WeightManager.decay() liniowo odbudowuje wagi
```

**Kluczowe obserwacje sprzężenia:**

1. **Drift detection i feature reselection są rozdzielone w czasie** — adaptacja zachodzi
   nie w momencie wykrycia dryfu, lecz `windowSize` instancji później.
2. **ADWIN sygnalizuje pojedyncze zdarzenie**, ale `DriftAwareSelector` wymaga zebrania
   pełnego okna post-drift, więc kolejne sygnały ADWIN-u w tym czasie są **nadpisywane**
   (każde wywołanie `onDriftDetected()` zaczyna procedurę od zera, vide §7).
3. Top-K wybierany w `DriftAwareSelector` służy jednocześnie jako **proxy dla „silnych
   cech”** ensemble’a: `removed` jest interpretowane jako zbiór cech słabych, `added`
   jako mocnych (w `DriftAwareSrpModel.onDriftDetected`). To uproszczenie semantyczne
   omawiam w §7.

---

## 7. Analiza krytyczna — błędy logiczne i potencjalne wątpliwości

Poniżej lista zidentyfikowanych zagrożeń logicznych w metodologii i implementacji,
posortowana wg wagi.

### 7.1 [POWAŻNE] Niespójność stanu między `SubspaceManager` a `DriftAwareSrpModel`

W `SubspaceManager.adaptSubspaces` (`SubspaceManager.java:78-105`) podprzestrzeń modelu
**jest mutowana w miejscu** dla każdego modelu z `|overlap| ≥ minOverlap`, niezależnie
od tego, czy ten model trafi później na listę `modelsToReset`:

```java
sub.removeAll(overlap);                        // <-- mutuje subspace[m]
sub.addAll(replacements);
...
if (overlapRatio >= resetRatio) {
    modelsToReset.add(m);                      // <-- tylko ci wracają do reset listy
}
```

Tymczasem `DriftAwareSrpModel.onDriftDetected` aktualizuje swoje lokalne kopie
`sortedSubspaces[m]` / `subspaceHeaders[m]` **wyłącznie dla modeli zresetowanych**
(`DriftAwareSrpModel.java:150-158`).

**Skutek:** dla modelu z `|overlap| ≥ minOverlap`, ale `overlapRatio < resetRatio`:
* `subspaceManager.subspace(m)` przestawia się na nowe cechy,
* `dasrp.sortedSubspaces[m]` / `subspaceHeaders[m]` pozostają na starych,
* drzewo nadal predykcuje i trenuje na **starych** cechach,
* przy kolejnym zdarzeniu dryfu `SubspaceManager` policzy overlap z nową, nieznaną drzewu
  podprzestrzenią — semantyka „czy dany model używa cech słabych?” staje się nieprawdziwa.

To jest klasyczna pułapka „cichej mutacji wspólnego stanu”. Stan nigdy się nie pokrywa,
a kolejne adaptacje pogłębiają rozjazd.

**Sugestia naprawy:** albo warunkowo zmieniać podprzestrzeń tylko dla modeli, które będą
resetowane, albo zawsze, gdy zmienia się podprzestrzeń, wymuszać reset (ale wtedy
`resetRatio` traci sens). Najczystszy patch:

```java
// w SubspaceManager.adaptSubspaces, ZAMIAST mutować in-place:
if (overlapRatio >= resetRatio) {
    sub.removeAll(overlap);
    sub.addAll(replacements);
    // …reszta…
    modelsToReset.add(m);
}
// modele bez resetu zostają z oryginalnym subspace.
```

W obecnym `defaultSpecs` `resetRatio=0.5` i `minOverlap=1` — przy podprzestrzeniach
o rozmiarze 4 (Agrawal) lub 20 (CTR) reżim ten jest bardzo łatwy do trafienia, więc
problem realnie wystąpi w eksperymentach.

### 7.2 [POWAŻNE] `DriftAwareSelector.onDriftDetected` nadpisuje stan przy szybkich, kolejnych dryftach

Metoda nie sprawdza, czy jesteśmy już w stanie `awaitingPostDrift`
(`DriftAwareSelector.java:128-137`):

```java
public void onDriftDetected() {
    if (recentBuffer.size() < Math.max(2, windowSize / 4)) return;
    preDriftSnapshot = new ArrayList<>(recentBuffer);
    igBefore = ranker.rank(originalHeader, preDriftSnapshot);
    recentBuffer.clear();              // <-- traci dane post-drift z poprzedniego okna
    awaitingPostDrift = true;
    postDriftCollected = 0;
}
```

Przy zwarcie następujących sygnałach ADWIN (np. niestabilne strefy strumienia) procedura
adaptacji nigdy się nie domyka — bufor jest perpetualnie czyszczony, licznik resetuje się,
a `adaptFeatures()` może nie zostać wywołane mimo wielokrotnych alarmów. Co gorsza,
nowy `igBefore` jest liczony na buforze, który dopiero co był post-drifted dla poprzedniego
sygnału — więc nawet jeśli adaptacja w końcu się odpali, baza porównawcza jest skażona.

**Sugestia naprawy:** `if (awaitingPostDrift) return;` na początku metody (zignorować
sygnały podczas zbierania okna post-drift), ewentualnie kolejka dryfów do późniejszej
obróbki.

### 7.3 [ŚREDNIE] „Pre-drift snapshot” nie jest pre-drift

`recentBuffer` zawiera ostatnie `windowSize` instancji **przed wywołaniem**
`onDriftDetected()`, ale ADWIN ma znaczne opóźnienie detekcji (typowo setki–tysiące
instancji). Część tego okna to już dane z nowego rozkładu. `igBefore` mieszany jest więc
z fragmentem post-drift, co osłabia różnicę `delta = igAfter − igBefore` i rozmywa decyzję
o swapie cech.

**Sugestia:** użyć węższego, „starego” okna (np. `[t − 2·windowSize, t − windowSize]`)
albo skorzystać z jawnego wskaźnika punktu cięcia z ADWIN-a (jeśli udostępniony).

### 7.4 [ŚREDNIE] `removed`/`added` ≠ globalna zmiana siły cech

W ścieżce DASRP listener przekazuje:

```java
dasrp.onDriftDetected(removed, added);   // (weakFeatures, strongFeatures)
```

`removed` to cechy, które wypadły z **top-K** (czyli przestały być w pierwszej dwudziestce),
`added` — które do top-K weszły. To **nie jest** ten sam zbiór, co cechy realnie
słabe/mocne w ensemble’u. W szczególności:

* cecha, która była i pozostała na pozycji 21–22, nigdy się nie pojawi w `removed`,
  nawet jeśli ensemble jej używa i jej IG dramatycznie spadło,
* cecha, której IG urósł, ale nie weszła do top-K, nigdy nie znajdzie się w `added`,
* `delta` IG (która faktycznie niesie informację globalną) jest dostępna w listenerze,
  ale **nie jest używana** przez `DriftAwareSrpModel.onDriftDetected`.

**Sugestia:** rozważyć użycie `delta` z progiem do generowania `weakFeatures`/`strongFeatures`
zamiast różnicy zbiorów top-K.

### 7.5 [ŚREDNIE] Pełny reset HT/HAT przy każdym alarmie selektora

§5.1: każdy sygnał `DriftAwareSelector` → `model.reset()` na monolitycznym modelu.
W obecnej konfiguracji ADWIN z `delta=0.002` na zaszumionym strumieniu Avazu/Criteo
łatwo da kilkanaście–kilkadziesiąt alarmów na 200 tys. instancji, a każdy z nich
oznacza utratę całego drzewa. To w praktyce sprawia, że model HT/HAT z DA-Selector ma
wbudowaną dyskwalifikację względem pełnego ensemble’u DASRP — czyli porównanie nie jest
„uczciwe” i może wyglądać jak triumf DASRP, podczas gdy faktycznie to artefakt
strategii reset-on-adapt.

**Sugestia:** rozważyć wariant „soft reset” — np. dotrenowanie istniejącego drzewa na
nagłówku z nakładającą się częścią cech, albo *pamięć krótkotrwała* (zachowanie modelu
wstecznego jako fallbacka).

### 7.6 [DROBNE] Sygnał dryfu liczy się w trakcie warmupu, ale handler jest tłumiony

W `PrequentialEvaluator.run` ADWIN dostaje sygnał na każdą instancję od t=0. Handler dryfu
ignoruje zdarzenia z `idx < warmup`, ale ADWIN już przemielił dane i potencjalnie **przyciął
swoje wewnętrzne okno**, więc po zakończeniu warmupu detektor nie startuje od stanu
czystego. Liczba dryfów raportowana w `driftCount`/`driftEvents` zawiera również te
ignorowane.

**Sugestia:** albo opóźnić start ADWIN-a do końca warmupu, albo wykluczyć alarmy z warmupu
z eksportowanego CSV.

### 7.7 [DROBNE] „Pre-rozgrzewany” `StaticTopKSelector` stwarza asymetrię w fazie 1

`ExperimentMatrix.staticPreWarmed` otwiera **dodatkowy strumień**, zużywa pierwsze 5 tys.
instancji, oblicza ranking i dopiero wtedy puszcza eksperyment. Tymczasem
`OnlineRankingSelector` startuje „na zimno” z heurystyką „pierwsze topK indeksów” aż do
pierwszego rerankingu. To znaczy, że na samym początku porównania (kluczowa początkowa
część strumienia, gdzie pojawia się większość gradientu uczenia) `static_topk` ma już
realny ranking, a `online_ranking` — de facto losowy. Ostateczne metryki kumulacyjne są
przez to nieco zniekształcone na korzyść `static_topk`. Dla strumienia 200 tys. instancji
efekt jest umiarkowany, ale dla ablacji na początku strumienia istotny.

**Sugestia:** dodać symetryczny warmup do `OnlineRankingSelector` (zwracać oryginalny
zestaw cech do końca pierwszego okna i nie raportować wyników z tego prefiksu).

### 7.8 [DROBNE] `model.reset()` + natychmiast `model.initialize(newHeader)` w listenerze HT/HAT

Sekwencja `reset()` → `initialize(newHeader)` w `HoeffdingTreeModel` powoduje *podwójną*
budowę klasyfikatora: pierwsza z poprzednim nagłówkiem, druga z nowym. Marnuje to trochę
CPU, semantycznie jest poprawne, ale zbędne.

### 7.9 [DROBNE] `AucMetric.getValue()` jest kosztowny przy snapshotach

Pomiar AUC na buforze 10 000 instancji jest dość kosztowny przy każdym snapshot’cie
(`logInterval=1_000`), bo w `getValue()` budowana jest pełna lista wskaźników, sortowana
i przebiegana dla rang. Bez wpływu na poprawność, ale dla 1 000 snapshotów w długim
przebiegu dochodzi do milionów alokacji `int[1]`.

### 7.10 [DROBNE] `AgrawalStreamProvider` używa `Math.random()` w mostku gradualnym

Linia `boolean useAfter = Math.random() < progress;` (`AgrawalStreamProvider.java:84`)
korzysta z globalnego, niereprodukowalnego RNG zamiast lokalnego, opartego na `seed`-zie.
Wyniki Agrawal-gradual nie są więc w pełni reprodukowalne między uruchomieniami,
mimo wstrzyknięcia `seed` do generatorów.

**Sugestia:** użyć `new Random(seed)` jako pola klasy.

---

## 8. Podsumowanie metodologiczne

Projekt jest dobrze zorganizowanym, modularnym stanowiskiem do **prequential evaluation**
modeli strumieniowych pod kątem wpływu strategii selekcji cech, ze szczególnym
uwzględnieniem **adaptacji wynikającej z detekcji dryfu**. Kluczową kontrybucją
metodologiczną są dwie konstrukcje:

* `DriftAwareSelector` — porównanie IG przed/po dryfcie ze swapem top-K,
* `DriftAwareSrpModel` z `SubspaceManager` + `WeightManager` — selektywna adaptacja
  ensemble’u z mechanizmem stopniowego odzyskiwania wag.

Najbardziej obciążający metodologię błąd (§7.1) — niespójność `SubspaceManager` ↔
`DriftAwareSrpModel` przy `overlapRatio < resetRatio` — sprawia, że dla części
eksperymentów ensemble realnie pracuje na nieaktualnym opisie cech, co podważa
interpretację porównań między wariantami z różnymi `resetRatio`. Drugi krytyczny punkt
(§7.2) — utrata zdarzeń adaptacji przy zwartych dryftach — może obniżać reaktywność
DA-Selectora w trudnych fragmentach strumienia. Pozostałe punkty są bardziej ergonomiczne
(§7.3–7.10), lecz część z nich (§7.5, §7.7) wpływa na uczciwość porównania.

Po naprawieniu §7.1 i §7.2 architektura jest spójna i zdolna do wiarygodnej oceny
hipotezy badawczej.
