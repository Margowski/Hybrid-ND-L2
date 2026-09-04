# ND-LLM – Hybrid-Integration mit GitHub Actions

## Was jetzt umgesetzt ist

Der Prototyp verbindet die drei vorgesehenen Rollen sicher miteinander:

\[
\text{großes LLM}
\rightarrow
\text{semantisches Ereignis}
\rightarrow
\text{transaktionaler Zustandskern}
\rightarrow
\text{Antwort und neuer Shadow State}.
\]

Das große LLM liefert in **einem** API-Aufruf eine normale Antwort und ein
strukturiertes JSON-Ereignis. Der lokale Kern prüft dessen Operation,
Konfidenz und möglichen Konflikt mit dem aktiven Fakt. Nur akzeptierte Updates
erhöhen die Zustandsversion und werden committed; der bisherige Zustand bleibt
als unveränderter Ausgangspunkt erhalten.

## GitHub-Sicherheit und Ablauf

Die Workflow-Datei `.github/workflows/hybrid_turn.yml` liest
`OPENAI_API_KEY` ausschließlich als GitHub-Secret. Sie besitzt nur
Leserechte für das Repository und schreibt keinen Commit. Nach einem manuell
ausgelösten Lauf wird der neue Zustand als `hybrid-state`-Artefakt
bereitgestellt. Damit bleibt die Entscheidung über eine dauerhafte Übernahme
beim Nutzer.

## Beispiel

Aus dem Satz „Der Termin wurde doch auf Freitag verschoben“ kann das LLM
beispielsweise liefern:

```json
{
  "answer": "Der Termin ist jetzt Freitag.",
  "event": {
    "operation": "correction",
    "subject": "Termin",
    "old_value": "Donnerstag",
    "value": "Freitag",
    "confidence": 0.96
  }
}
```

Der Kern prüft die Korrektur gegen den aktiven Zustand, schreibt Freitag als
gültigen Wert und bewahrt Donnerstag in der Historie.

## Bewusst noch nicht behauptet

Dieser Integrationsstand ist keine bereits trainierte semantische ND-L2-Gate.
Die semantische Ereignisdeutung kommt zunächst vom großen LLM, damit wir nicht
zwei teure LLM-Aufrufe benötigen. Der nächste wissenschaftliche Schritt bleibt
S1: ein leichter, trainierter Event-Encoder ohne expliziten Kontextmarker gegen
denselben Zustandstest.
