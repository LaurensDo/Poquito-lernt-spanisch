# Poquito lernt Spanisch

Spanisch-Lern-App als einzelne Seite (`index.html`), gehostet über GitHub Pages. Lernstand liegt auf dem Gerät
(optional Abgleich über ein privates GitHub-Gist).

## Stimmen (vorab erzeugte Aufnahmen)

Alle festen Texte (Vokabeln, Sätze, Zahlen, Verbformen, Podcasts …) liegen als MP3 in `audio/`. Fehlt eine Aufnahme,
spricht die App mit der Stimme des Geräts. In den Einstellungen wählt man zwischen drei Stimmen:

| Stimme | Verzeichnis | Manifest | Modell |
| --- | --- | --- | --- |
| 🇪🇸 Kastilisch (Standard) | `audio/kf`, `audio/km` | `audio/manifest-k.json` | [Piper](https://github.com/OHF-Voice/piper1-gpl) `es_ES-sharvard-medium` (Sharvard-Korpus, CC BY 3.0) |
| ✨ Natürlich | `audio/f`, `audio/m` | `audio/manifest.json` | [Kokoro-82M](https://github.com/hexgrad/kokoro) (Apache-2.0); spricht eher lateinamerikanisch (z/c wie s) |
| 📱 Gerät | – | – | Sprachsynthese des Geräts |

Das Manifest ordnet Text → Datei zu. Neu erzeugen (Python 3.11):

```
node tools/audio/extract_texts.js > tools/audio/texts.json

# Kastilisch: pip install piper-tts torch transformers faster-whisper lameenc scipy soundfile numpy
python tools/audio/make_piper.py --model es_ES-sharvard-medium.onnx      # optional: --part 0/2
python tools/audio/build_manifest.py --engine k

# Natürlich: pip install torch kokoro faster-whisper lameenc scipy soundfile numpy
python tools/audio/make_audio.py                                          # optional: --part 0/3
python tools/audio/build_manifest.py
```

Jede Aufnahme wird automatisch geprüft. Kastilisch: Ein Phonemerkenner (`tools/audio/phoncheck.py`) vergleicht die
gesprochenen Laute mit denen, die espeak-ng für Spanien vorsieht (z. B. „th“ in *zapato*); auffällige Aufnahmen werden
neu gesprochen oder von Whisper gegengeprüft. Natürlich: Whisper. Nur geprüfte Aufnahmen kommen ins Manifest.

`python tools/audio/phoncheck.py aufnahme.mp3 "zapato"` zeigt, welche Laute ein Clip trifft.

## Tagesaufgabe und Lernstand

Die App merkt sich je Zeitform, ob sie *neu*, *in Übung* oder *gekonnt* ist (Selbsteinschätzung unter „Mein Niveau“,
dazu die Trefferquote aus den Runden; die Voreinstellung entspricht B1). Daraus entsteht jeden Tag ein fester Plan: das
Lernziel (die nächste nicht gekonnte Zeitform im Lernpfad oder ein selbst gewähltes), eine Zeitform zum Auffrischen
oder Entdecken, Wortschatz, dazu Bonusaufgaben. Fester Plan heißt: Er ändert sich im Lauf des Tages nicht, nur wenn man
Niveau oder Ziel ändert.

## Konjugationen prüfen

```
pip install verbecc
node tools/dump_verbs.js > /tmp/verbs.json
python tools/verify_verbs.py /tmp/verbs.json
```

Vergleicht alle Verben der Datenbank (ca. 500, 15 Zeiten) mit einer unabhängigen Bibliothek.
