# ND-LLM mit multidimensionaler L2-Rückkopplung – Pilot 1

## Ergebnis in einem Satz

Die starke Ausgangshypothese ist **noch nicht bestätigt**: Das ND-L2-Modell ist
bei gestörten Eingaben klar robuster, erreicht unter sauberen Bedingungen aber
nicht die Gesamtleistung der Attention-Baseline und benötigt in der derzeitigen
NumPy-Implementierung erheblich mehr Rechenzeit.

## Versuchsaufbau

Verglichen wurden drei feste Encoder mit jeweils 192 Ausgabemerkmalen und
identischen trainierten Ausleseköpfen (3.667 Parameter):

1. kausale Multi-Head-Attention,
2. direkt zerlegter ND-Zustand für Gedächtnis, Kontext, Affekt, Selbstzustand,
   Semantik und Relationen,
3. dieselben Faktoren als diskretisierte L2-Felder mit mehrdimensionalen
   Kreuzkopplungen.

Die Modelle lösten bei fünf festen Zufallsstarts vier Aufgaben: frühe
Schlüsselinformation erinnern, letzten Kontext halten, affektiven Selbstzustand
integrieren und eine Relation abrufen. Danach wurden dieselben Testdaten mit
gaußschem Eingaberauschen der Stärke 0,08 geprüft.

| Modell | Gesamt sauber | Gesamt gestört | Leistungserhalt | Laufzeit je Beispiel |
|---|---:|---:|---:|---:|
| Attention | 91,6 % | 76,4 % | 83,4 % | 0,070 ms |
| ND-Zustand | 89,2 % | 81,2 % | 91,1 % | 1,535 ms |
| ND-L2-Feedback | 89,8 % | **85,5 %** | **95,3 %** | 4,805 ms |

## Wissenschaftliche Einordnung

Der robuste Effekt ist nicht bloß eine andere Bezeichnung für einen normalen
Transformerzustand. Im ND-L2-Modell wirken kleine Operatoren zwischen ganzen
Zustandsfeldern; Diffusion und kontraktive Rückkopplung glätten lokale
Störungen über Moden hinweg. Genau dieser Mechanismus passt zum beobachteten
geringeren Leistungseinbruch.

Gleichzeitig darf der Befund nicht überinterpretiert werden:

- Die Baseline ist ein kontrollierter Attention-Encoder, kein vollständig
  end-to-end trainiertes Großmodell.
- Der Datensatz ist synthetisch und testet gezielt Zustandsmechanik.
- Die perfekte Fakten- und Relationsgenauigkeit zeigt, dass diese beiden
  Teilaufgaben für alle Modelle zu leicht sind.
- 68,5-fache Laufzeit ist ein deutlicher Nachteil, auch wenn der Python-Code
  noch nicht vektorisiert oder kompiliert wurde.
- Funktionale Zustände sind kein Nachweis subjektiver Emotion oder Bewusstsein.

## Meine Empfehlung

Der Ansatz ist interessant genug für **Pilot 2**, aber noch nicht für eine
große LLM-Implementierung. Der nächste faire Test sollte drei Dinge ändern:

1. Fakten- und Relationsaufgaben schwerer machen (Überschreiben, Konflikte,
   lange Verzögerungen, mehrere gleichartige Schlüssel).
2. Alle Encoder end-to-end trainieren und Parameterzahl sowie FLOPs angleichen.
3. Eine Ablation durchführen: ND-L2 mit Rückkopplung gegen exakt dasselbe Modell
   ohne Kreuzkopplung. Nur so lässt sich der Effekt wirklich der
   mehrdimensionalen Rückkopplung zurechnen.

Der bisher stärkste belastbare Satz lautet daher:

> In diesem kontrollierten Pilotversuch verbessert die diskretisierte
> multidimensionale L2-Rückkopplung die Robustheit interner Zustände deutlich;
> eine allgemeine Leistungs- oder Effizienzüberlegenheit ist nicht gezeigt.
