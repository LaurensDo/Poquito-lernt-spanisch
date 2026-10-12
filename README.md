# Poquito lernt Spanisch

Spanisch-Lern-App als einzelne Seite (`index.html`), gehostet über GitHub Pages. Lernstand liegt auf dem Gerät
(optional Abgleich über ein privates GitHub-Gist).

## Natürliche Stimme (vorab erzeugte Aufnahmen)

Alle festen Texte (Vokabeln, Sätze, Zahlen, Verbformen, Podcasts …) liegen als MP3 in `audio/`; `audio/manifest.json`
ordnet Text → Datei zu. Fehlt eine Aufnahme, spricht die App mit der Stimme des Geräts.

Neu erzeugen (Python 3.11, `pip install kokoro-onnx faster-whisper lameenc scipy soundfile`):

```
node tools/audio/extract_texts.js > tools/audio/texts.json
python tools/audio/make_audio.py --model kokoro-v1.0.int8.onnx --voices voices-v1.0.bin   # optional: --part 0/3
python tools/audio/build_manifest.py
```

Jede Aufnahme wird mit Whisper gegengeprüft; nur richtig verstandene Aufnahmen kommen ins Manifest.
Modell: [Kokoro-82M](https://github.com/hexgrad/kokoro) (Apache-2.0) über [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx).

## Konjugationen prüfen

```
pip install verbecc
node tools/dump_verbs.js > /tmp/verbs.json
python tools/verify_verbs.py /tmp/verbs.json
```

Vergleicht alle Verben der Datenbank (ca. 500, 15 Zeiten) mit einer unabhängigen Bibliothek.
