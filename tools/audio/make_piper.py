#!/usr/bin/env python3
"""Erzeugt die kastilische Stimme von Poquito vorab als MP3 (Piper + Sharvard-Korpus, Spanien).

  node tools/audio/extract_texts.js > tools/audio/texts.json
  python tools/audio/make_piper.py --model es_ES-sharvard-medium.onnx [--part 0/2]
  python tools/audio/build_manifest.py --engine k

Jede Aufnahme wird zweifach geprüft: Ein Phonemerkenner (phoncheck.py) vergleicht die gesprochenen Laute mit denen,
die espeak-ng für Spanien vorsieht (z. B. „th“ in zapato). Piper ist zufallsgesteuert – fällt eine Aufnahme durch,
wird sie mit anderen Einstellungen neu gesprochen und die beste behalten. Bleibt auch die beste auffällig, hört
Whisper noch einmal zu; was dort nicht als derselbe Text verstanden wird, kommt nicht ins Manifest
(die App liest es dann mit der Gerätestimme vor).

Benötigt: pip install piper-tts torch transformers faster-whisper lameenc scipy soundfile numpy
Stimme: https://huggingface.co/rhasspy/piper-voices/tree/main/es/es_ES/sharvard/medium
Korpus: Sharvard (Aubanel, Davis & Kim 2014), CC BY 3.0, https://datashare.ed.ac.uk/handle/10283/574
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from math import gcd
from scipy.signal import resample_poly

from common import PIPER_SPEAKERS, PIPER_DIR, piper_length, piper_file_id  # noqa: E402
from audiolib import SR, same, clean, to_mp3  # noqa: E402
from phoncheck import Recognizer, phonemes_of  # noqa: E402

GOOD = 0.2           # darunter ist die Aufnahme sicher in Ordnung
ACCEPT = 0.3         # bis hierhin wird sie angenommen
ATTEMPTS = [(0.667, 0.8), (0.5, 0.6), (0.8, 1.0), (0.4, 0.5), (0.9, 0.8)]   # (noise_scale, noise_w_scale)


class PiperMaker:
    def __init__(self, model, threads):
        import torch
        from piper import PiperVoice
        from faster_whisper import WhisperModel
        torch.set_num_threads(threads)
        self.voice = PiperVoice.load(model)
        self.sr = self.voice.config.sample_rate
        self.rec = Recognizer()
        self.whisper = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=threads)

    def synth(self, text, voice, kind, attempt):
        from piper import SynthesisConfig
        noise, noise_w = ATTEMPTS[attempt % len(ATTEMPTS)]
        cfg = SynthesisConfig(speaker_id=PIPER_SPEAKERS[voice], length_scale=piper_length(kind), noise_scale=noise, noise_w_scale=noise_w)
        said = text if text[-1] in ".!?…" else text + "."
        a = np.concatenate([c.audio_float_array for c in self.voice.synthesize(said, syn_config=cfg)])
        g = gcd(SR, self.sr)
        return clean(resample_poly(a, SR // g, self.sr // g).astype(np.float32))

    def hear(self, clips):
        """Whisper-Text je Clip (mehrere Clips in einem Fenster, durch Pausen getrennt)."""
        out = []
        for k in range(0, len(clips), 12):
            gap = np.zeros(int(0.6 * SR), np.float32)
            parts, spans, pos = [], [], 0
            for c in clips[k:k + 12]:
                parts += [gap, c]
                spans.append((pos + len(gap), pos + len(gap) + len(c)))
                pos += len(gap) + len(c)
            a16 = resample_poly(np.concatenate(parts + [gap]), 2, 3).astype(np.float32)
            segs, _ = self.whisper.transcribe(a16, language="es", beam_size=2, word_timestamps=True, condition_on_previous_text=False)
            words = [w for s in segs for w in (s.words or [])]
            texts = [[] for _ in spans]
            for w in words:
                mid = (w.start + w.end) / 2 * SR
                for i, (a0, a1) in enumerate(spans):
                    if a0 - 0.3 * SR <= mid < a1 + 0.3 * SR:
                        texts[i].append(w.word.strip())
                        break
            out += [" ".join(t) for t in texts]
        return out

    def make(self, items):
        """items: [{text, voice, kind}] -> [(audio | None, per, grund)]"""
        res = []
        for it in items:
            ref = phonemes_of(self.voice, it["text"] if it["text"][-1] in ".!?…" else it["text"] + ".")
            best = (9.9, None, 9.9)   # (Fehlerquote, Aufnahme, Güte)
            for attempt in range(len(ATTEMPTS)):
                a = self.synth(it["text"], it["voice"], it["kind"], attempt)
                dur = len(a) / SR
                if not 0.2 <= dur <= 0.8 + 0.2 * len(it["text"]):
                    continue
                per, th_miss, th_n = self.rec.check(a, SR, ref)
                score = per + 0.25 * th_miss   # überhörtes „th“ (kastilisch) kostet extra: lieber neu sprechen
                if score < best[2]:
                    best = (per, a, score)
                if score <= GOOD:
                    break
            res.append(best[:2])
        # auffällige noch einmal mit Whisper prüfen
        shaky = [i for i, (per, a) in enumerate(res) if a is not None and per > ACCEPT]
        if shaky:
            heard = self.hear([res[i][1] for i in shaky])
            for i, h in zip(shaky, heard):
                tol = 0.1 if len(items[i]["text"].split()) > 4 else 0.0
                if not same(h, items[i]["text"], tol):
                    res[i] = (res[i][0], None)
        return [(a, per, None if a is not None else ("leer" if per > 9 else "laut")) for per, a in res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--texts", default=os.path.join(os.path.dirname(__file__), "texts.json"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "audio"))
    ap.add_argument("--part", default="0/1", help="n/m: nur jeden m-ten Eintrag ab n (zum Parallelisieren)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    n, m = map(int, args.part.split("/"))
    items = json.load(open(args.texts))[n::m]
    if args.limit:
        items = items[:args.limit]
    mk = PiperMaker(args.model, args.threads)
    done = failed = 0
    t0 = time.time()
    todo = []
    for it in items:
        fid = piper_file_id(it["voice"], it["kind"], it["text"])
        if not os.path.exists(os.path.join(args.out, PIPER_DIR[it["voice"]], fid + ".mp3")):
            todo.append((it, fid))
    BATCH = 12
    with open(os.path.join(args.out, f".meta-k{n}.jsonl"), "a") as meta, open(os.path.join(args.out, f".failed-k{n}.jsonl"), "a") as bad:
        for k in range(0, len(todo), BATCH):
            chunk = todo[k:k + BATCH]
            outs = mk.make([it for it, _ in chunk])
            for (it, fid), (a, per, why) in zip(chunk, outs):
                if a is None:
                    bad.write(json.dumps({"text": it["text"], "voice": it["voice"], "kind": it["kind"], "per": round(per, 3), "why": why}, ensure_ascii=False) + "\n"); bad.flush()
                    failed += 1
                    continue
                path = os.path.join(args.out, PIPER_DIR[it["voice"]], fid + ".mp3")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(to_mp3(a))
                meta.write(json.dumps({"text": it["text"], "voice": it["voice"], "id": fid, "ms": int(len(a) / SR * 1000), "per": round(per, 3)}, ensure_ascii=False) + "\n"); meta.flush()
                done += 1
            print(f"[{n}/{m}] {done} ok, {failed} verworfen von {len(todo)}, {time.time() - t0:.0f}s", flush=True)
    print(f"[{n}/{m}] fertig: {done} ok, {failed} verworfen")


if __name__ == "__main__":
    main()
