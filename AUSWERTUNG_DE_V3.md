# ND-LLM – Pilot 3 mit virtuellem Shadow State

## Ergebnis

Die virtuelle Fortschreibung funktioniert in diesem Modell exakt und spart
messbar Rechenzeit. Gegenüber dem gegateten eager Modell bleiben alle
Qualitätswerte unverändert; die Zeit pro Beispiel sinkt um knapp 30 Prozent.

| Modell | Gesamt sauber | Gesamt gestört | Zeit je Beispiel |
|---|---:|---:|---:|
| ND-L2 mit gelerntem Tor | 93,12 % | 91,96 % | 5,770 ms |
| **ND-L2 mit virtuellem Zustand** | **93,12 %** | **91,96 %** | **4,015 ms** |

Über die fünf Seeds beträgt die Beschleunigung je Lauf zwischen 28,62 und
33,46 Prozent; im Mittel spart sie 1,756 ms pro Beispiel (30,4 Prozent).

## Was geschieht technisch?

Der laufende Vorgang besitzt einen Shadow State. Die vollständigen
Gedächtnis- und Relationsfelder werden nicht nach jedem Token materialisiert.
Stattdessen speichert das System nur ihre lokalen Impulse und die skalaren
Rückkopplungsfaktoren:

\[
z_t=q_t(Az_{t-1}+u_t).
\]

Am erfolgreichen Ende werden alle Deltas in einem Schritt aufgelöst:

\[
z_T=\sum_{\tau=1}^{T}
\left(\prod_{k=\tau}^{T}q_k\right)A^{T-\tau}u_\tau.
\]

Der schnelle Kontext sowie die nichtlinearen Affekt-, Selbst- und
Semantikfelder werden weiterhin unmittelbar aktualisiert. Das ist notwendig,
weil sie die nächste interne Bewertung beeinflussen. Erst wenn Materialisierung
und Sicherheitsprüfung erfolgreich sind, wird der neue Zustand committed. Bei
einem Fehler bleibt der vorherige Commit unverändert.

## Warum bleibt die Qualität gleich?

Die Diffusion der Gedächtnis- und Relationsfelder ist in diesem Pilot linear;
ihre Rückkopplung ist ein gemeinsamer Skalar. Deshalb kann die Fortschreibung
algebraisch exakt umgeformt werden. Der Unit-Test vergleicht alle 192
Ausgabemerkmale direkt mit dem eager Modell und akzeptiert nur Unterschiede im
numerischen Rundungsbereich.

## Grenze und nächster Optimierungsschritt

Die Virtualisierung macht aus dem Modell noch kein schnelles LLM: Es bleibt
etwa 54-mal langsamer als die sehr schlanke Attention-Baseline dieses Tests.
Sie entfernt jedoch einen realen Anteil unnötiger Arbeit, ohne Qualität zu
opfern. Der nächste sichere Hebel ist Batch-Vektorisierung und kompilierte
Ausführung; erst danach sollten wir bei den nichtlinearen Feldern mit
ereignisgesteuerten oder mehrstufigen Updates experimentieren.
