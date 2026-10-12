#!/usr/bin/env python3
"""Erzeugt die Sprachausgabe von Poquito vorab (Kokoro-82M, Spanisch) als MP3.

  node tools/audio/extract_texts.js > tools/audio/texts.json
  python tools/audio/make_audio.py --model kokoro-v1.0.int8.onnx --voices voices-v1.0.bin [--part 0/3]
  python tools/audio/build_manifest.py

Jede Aufnahme wird mit Whisper gegengeprüft (Spracherkennung): Nur was richtig verstanden wird, kommt ins
Manifest. Alles andere liest die App weiter mit der Gerätestimme vor.
Kurze Einzelwörter spricht das Modell allein schlecht; sie werden deshalb in einem Trägerwort erzeugt
("Ahora jugar.") und an der Wortgrenze herausgeschnitten – die Grenze liefert das Modell selbst
(vorhergesagte Lautdauern), nicht die Spracherkennung.

Benötigt: pip install torch kokoro faster-whisper lameenc scipy soundfile numpy
Modell: Kokoro-82M (Apache-2.0), https://github.com/hexgrad/kokoro
"""
import argparse, json, os, re, sys, time, unicodedata
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import lameenc
from scipy.signal import resample_poly
import torch
from kokoro import KPipeline
from faster_whisper import WhisperModel

from common import VOICE_NAMES, SPEEDS, CARRIERS, TAG, file_id  # noqa: E402
SHORT_WORDS = 4


from audiolib import SR, number_words, key, lev, same, clean, to_mp3  # noqa: E402


class Maker:
    def __init__(self, threads):
        torch.set_num_threads(threads)
        self.pipe = KPipeline(lang_code="e", repo_id="hexgrad/Kokoro-82M")
        self.vocab = self.pipe.model.vocab
        self.w = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=threads)

    def synth_timed(self, text, voice, speed):
        """Audio plus die Frame-Position (je 600 Samples) der ersten Wortgrenze, falls der Text aus einem Stück besteht."""
        parts = []
        with torch.no_grad():
            for r in self.pipe(text, voice=VOICE_NAMES[voice], speed=speed):
                parts.append(r)
        audio = np.concatenate([p.output.audio.numpy() for p in parts]).astype(np.float32)
        boundary = None
        if len(parts) == 1 and parts[0].output.pred_dur is not None:
            kept = [c for c in parts[0].phonemes if c in self.vocab]
            dur = parts[0].output.pred_dur.tolist()
            if " " in kept and len(dur) == len(kept) + 2:
                i = kept.index(" ")
                before = sum(dur[: i + 1])          # Anfang des Leerzeichens (BOS + Laute davor)
                boundary = int((before + dur[i + 1] / 2) * 600)  # Mitte des Leerzeichens
        return audio, boundary

    def synth(self, text, voice, speed):
        return self.synth_timed(text, voice, speed)[0]

    # --- Spracherkennung im Stapel: viele kurze Clips in einem Durchlauf (Pausen dazwischen) ---
    def _asr_window(self, clips):
        gap = int(0.6 * SR)
        parts, spans, pos = [], [], 0
        for c in clips:
            parts += [np.zeros(gap, np.float32), c.astype(np.float32)]
            spans.append((pos + gap, pos + gap + len(c)))
            pos += gap + len(c)
        audio = np.concatenate(parts + [np.zeros(gap, np.float32)])
        a16 = resample_poly(audio, 2, 3).astype(np.float32)
        segs, _ = self.w.transcribe(a16, language="es", beam_size=2, word_timestamps=True, condition_on_previous_text=False)
        words = [x for s in segs for x in (s.words or [])]
        out = [[] for _ in clips]
        for w in words:
            mid = (w.start + w.end) / 2 * SR
            for idx, (a0, a1) in enumerate(spans):
                if a0 - 0.3 * SR <= mid < a1 + 0.3 * SR:
                    out[idx].append((w.word.strip(), w.start - a0 / SR, w.end - a0 / SR))
                    break
        return out

    def asr_words(self, clips, budget=24.0):
        """Wörter mit Zeitstempeln je Clip; Clips werden auf Fenster von höchstens `budget` Sekunden verteilt."""
        result, window, total = [None] * len(clips), [], 0.0

        def flush():
            for k, w in enumerate(self._asr_window([c for _, c in window]) if window else []):
                result[window[k][0]] = w

        for idx, c in enumerate(clips):
            d = len(c) / SR + 0.6
            if window and total + d > budget:
                flush(); window, total = [], 0.0
            window.append((idx, c)); total += d
        flush()
        return result

    def asr_text(self, clips):
        pad = np.zeros(int(0.25 * SR), np.float32)
        return [" ".join(w[0] for w in ws) for ws in self.asr_words([np.concatenate([pad, c, pad]) for c in clips])]

    def quietest(self, a, t0, t1):
        lo, hi = max(int(t0 * SR), 0), min(int(t1 * SR), len(a))
        if hi - lo < 240:
            return (lo + hi) // 2
        fr = 120
        e = [float(np.sqrt(np.mean(a[i:i + fr] ** 2))) for i in range(lo, hi - fr, 40)]
        return lo + int(np.argmin(e)) * 40 + fr // 2

    def make(self, items):
        """items: [{text, voice, kind}] -> [(audio | None, heard, grund)]"""
        res = [None] * len(items)
        short = [i for i, it in enumerate(items) if len(it["text"].split()) <= SHORT_WORDS]
        long_ = [i for i in range(len(items)) if i not in set(short)]
        speed = lambda i: SPEEDS.get(items[i]["kind"], 0.92)

        # lange Texte: direkt erzeugen und als Ganzes prüfen
        pending = list(long_)
        for attempt in range(3):
            if not pending:
                break
            audios = {i: clean(self.synth(items[i]["text"] if items[i]["text"][-1] in ".!?" else items[i]["text"] + ".", items[i]["voice"], speed(i) + (0, 0.03, -0.03)[attempt])) for i in pending}
            heard = self.asr_text([audios[i] for i in pending])
            nxt = []
            for i, h in zip(pending, heard):
                ref, hyp = key(items[i]["text"]), key(h)
                if lev(ref, hyp) <= max(2, int(len(ref) * 0.08)):
                    res[i] = (audios[i], h, None)
                else:
                    nxt.append(i); res[i] = (None, h, "lang")
            pending = nxt

        # kurze Texte: im Trägerwort erzeugen, an der Wortgrenze schneiden, prüfen
        pending = list(short)
        for carrier in CARRIERS:
            if not pending:
                break
            tails, idx = [], []
            for i in pending:
                audio, cut = self.synth_timed(f"{carrier} {items[i]['text']}.", items[i]["voice"], speed(i))
                if cut is None:
                    continue
                tails.append(clean(audio[cut:])); idx.append(i)
            heards = self.asr_text(tails) if tails else []
            ok = set()
            for i, t, h in zip(idx, tails, heards):
                text = items[i]["text"]
                res[i] = (None, h, "kurz")
                if same(h, text, 0.0) and 0.25 <= len(t) / SR <= 0.5 + 0.35 * len(text):
                    res[i] = (t, h, None); ok.add(i)
            pending = [i for i in pending if i not in ok]
        return [(r[0], r[1], r[2]) if r else (None, "", "leer") for r in res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default=os.path.join(os.path.dirname(__file__), "texts.json"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "audio"))
    ap.add_argument("--part", default="0/1", help="n/m: nur jeden m-ten Eintrag ab n (zum Parallelisieren)")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    n, m = map(int, args.part.split("/"))
    items = json.load(open(args.texts))[n::m]
    if args.limit:
        items = items[:args.limit]
    mk = Maker(args.threads)
    os.makedirs(args.out, exist_ok=True)
    done = failed = 0; t0 = time.time()
    todo = []
    for it in items:
        speed = SPEEDS.get(it["kind"], 0.92)
        fid = file_id(it["voice"], speed, it["text"])
        if not os.path.exists(os.path.join(args.out, it["voice"], fid + ".mp3")):
            todo.append((it, fid))
    BATCH = 14
    with open(os.path.join(args.out, f".meta-{n}.jsonl"), "a") as meta, open(os.path.join(args.out, f".failed-{n}.jsonl"), "a") as bad:
        for k in range(0, len(todo), BATCH):
            chunk = todo[k:k + BATCH]
            outs = mk.make([it for it, _ in chunk])
            for (it, fid), (a, heard, why) in zip(chunk, outs):
                voice, text = it["voice"], it["text"]
                if a is None:
                    bad.write(json.dumps({"text": text, "voice": voice, "kind": it["kind"], "heard": heard, "why": why}, ensure_ascii=False) + "\n"); bad.flush()
                    failed += 1
                    continue
                path = os.path.join(args.out, voice, fid + ".mp3")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(to_mp3(a))
                meta.write(json.dumps({"text": text, "voice": voice, "id": fid, "ms": int(len(a) / SR * 1000)}, ensure_ascii=False) + "\n"); meta.flush()
                done += 1
            print(f"[{n}/{m}] {done} ok, {failed} verworfen von {len(todo)}, {time.time() - t0:.0f}s", flush=True)
    print(f"[{n}/{m}] fertig: {done} ok, {failed} verworfen")


if __name__ == "__main__":
    main()
