# ND-LLM – Pilot 2 mit gelerntem Kontexttor

## Ergebnis

Das gelernte Kontexttor beseitigt die zuvor identifizierte Schwachstelle in
diesem Test vollständig. Das gegatete ND-L2-Modell gewinnt sowohl bei sauberen
als auch bei gestörten Eingaben.

| Modell | Kontext | Gesamt sauber | Gesamt gestört | Leistungserhalt |
|---|---:|---:|---:|---:|
| Attention | 93,9 % | 91,6 % | 76,4 % | 83,4 % |
| ND-Zustand | 83,9 % | 89,2 % | 81,2 % | 91,1 % |
| ND-L2 ohne Tor | 85,8 % | 89,8 % | 85,5 % | 95,3 % |
| **ND-L2 mit gelerntem Tor** | **100,0 %** | **93,1 %** | **92,0 %** | **98,8 %** |

Alle Werte sind Mittelwerte aus fünf Seeds mit jeweils 700 Trainings- und 250
Testsequenzen. Das Eingaberauschen betrug `sigma = 0,08`.

## Was wurde gelernt?

Das Tor besitzt sechs trainierbare Koeffizienten:

\[
g_t=\sigma\!\left(w^\top\varphi(x_t)\right).
\]

Es lernt aus den Trainingssequenzen, welche lokalen Merkmale ein
Kontextschreibereignis kennzeichnen. Anschließend gilt

\[
C_{t+1}=(1-g_t)\mathcal D C_t+g_t B_Cx_t.
\]

Bei einem Kontextwechsel nähert sich `g` dem Wert 1 und ersetzt den schnellen
Kontext. Bei anderen Ereignissen bleibt `g` nahe 0 und schützt den bestehenden
Zustand. Das langsame Feld und die multidimensionalen Kopplungen bleiben
erhalten.

## Vergleich zum ersten Pilot

- Kontext: plus 14,2 Prozentpunkte gegenüber ND-L2 ohne Tor.
- Gesamt sauber: plus 3,4 Prozentpunkte gegenüber ND-L2 ohne Tor und plus 1,5
  Prozentpunkte gegenüber Attention.
- Gesamt gestört: plus 6,5 Prozentpunkte gegenüber ND-L2 ohne Tor und plus 15,6
  Prozentpunkte gegenüber Attention.
- Selbstzustand: keine nachweisbare Verbesserung; gegenüber dem ungateten
  Modell im Mittel minus 0,7 Prozentpunkte, mit einem Konfidenzintervall, das
  null einschließt.

Die Verbesserung gegenüber Attention trat beim sauberen Gesamtwert in allen
fünf Seeds auf. Das approximative gepaarte 95-%-Intervall der Differenz beträgt
plus 0,8 bis plus 2,3 Prozentpunkte. Unter Rauschen beträgt es plus 13,1 bis
plus 18,1 Prozentpunkte.

## Einschränkung

Das Resultat beweist noch keine Überlegenheit gegenüber einem trainierten LLM.
Der Datensatz enthält einen expliziten lokalen Kontexttyp, den das Tor
überwacht lernen kann. Bewiesen ist daher nur der engere Mechanismus:

> Wenn Kontextwechsel zuverlässig erkennbar sind, verbindet ein selektives
> Überschreibungstor in diesem Pilot schnelle Anpassung mit der Robustheit des
> multidimensionalen L2-Zustands.

Der nächste anspruchsvolle Test muss Kontextwechsel aus Bedeutung und
Widerspruch erkennen, ohne einen expliziten Kontextmarker zu erhalten.
