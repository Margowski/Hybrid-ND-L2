# ND-LLM – Pilot 4: Batch-Vektorisierung und JIT

## Entscheidung

Für das aktuelle Modell ist die beste Ausführung ein **vektorisierter
Shadow-State-Kern mit Numba-JIT auf der CPU**. Er bewahrt alle Qualitätswerte
und liegt nur noch knapp über der schlanken Attention-Baseline.

| Modell | Gesamt sauber | Gesamt gestört | Zeit je 36-Token-Sequenz |
|---|---:|---:|---:|
| Attention-Baseline | 91,60 % | 76,38 % | 0,074 ms |
| Gegatetes eager ND-L2 | 93,12 % | 91,96 % | 5,704 ms |
| Shadow State, vektorisiert | 93,12 % | 91,96 % | 0,157 ms |
| **Shadow State, vektorisiert + JIT** | **93,12 %** | **91,96 %** | **0,097 ms** |

Damit gilt:

- Der JIT-Kern ist etwa **58,6-mal schneller** als das ursprüngliche gegatete
  eager Modell.
- Gegenüber dem vektorisierten Pfad spart JIT im selben Lauf rund **38 %**.
- Gegenüber Attention verbleiben nur etwa **0,024 ms** bzw. Faktor **1,32**.
- Alle zwölf Tests bestehen; der JIT-Ausgaberaum stimmt mit dem vektorisierten
  Referenzpfad bis auf numerische Rundungsgrenzen überein.

## Richtige Einsatzstrategie

Für einen dauerhaft laufenden Dienst wird der JIT-Kern beim Start einmal
angewärmt. Die erste Kodierung benötigt in der Messung etwa 1,8 Sekunden für
Kompilierung und Start; danach lag der Warmzustand bei etwa 0,08 bis 0,10 ms pro
Sequenz.

Für einen einzelnen, kurzlebigen Einmalprozess ist der reine vektorisierte
NumPy-Pfad sinnvoller, weil er keinen Kompilierungsstart benötigt. Eine GPU ist
bei 36 Tokens und 192 Zustandskoeffizienten derzeit nicht empfehlenswert:
Transfer- und Kernelstartkosten wären voraussichtlich größer als der Gewinn.

## Was bleibt offen?

Die aktuelle Beschleunigung verändert keine Modellgleichung. Erst wenn wir
längere Sequenzen, deutlich größere Felder oder viele parallele Nutzer
erreichen, sollten wir GPU, niedrigere Präzision oder ereignisgesteuerte
nichtlineare Updates erneut messen. Für den jetzigen Stand wäre das voreilige
Komplexität.
