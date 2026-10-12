#!/usr/bin/env python3
"""Prüft, ob eine Aufnahme die Laute spricht, die kastilisches Spanisch verlangt (z. B. „th“ in „zapato“).

Ein Phonemerkenner (wav2vec2, https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft) hört die Aufnahme ab;
das Ergebnis wird mit den Lauten verglichen, die espeak-ng für den Text vorsieht (Spanien). PER = Anteil falsch
erkannter Laute (der Erkenner selbst irrt sich bei etwa jedem fünften Laut, das ist der Grundwert).

  python phoncheck.py aufnahme.mp3 "zapato"

Benötigt: pip install torch transformers soundfile scipy numpy piper-tts
"""
import re, sys
from math import gcd
import numpy as np
from scipy.signal import resample_poly

MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
STRIP = re.compile(r"[ˈˌː.,;:!?¿¡\s]")
# Der Erkenner trennt manche Laute nicht zuverlässig (Zungenschlag/Zungenrolle, weich/hart, ähnliche Vokale).
# Sie zählen als gleich. Bewusst NICHT zusammengelegt: θ und s – daran erkennt man Kastilisch.
CANON = {"ɾ": "r", "β": "b", "v": "b", "ð": "d", "ɣ": "ɡ", "g": "ɡ", "ʝ": "i", "ʎ": "i", "ʒ": "i", "ɟ": "i", "j": "i",
         "ʧ": "ʃ", "ɛ": "e", "ɪ": "e", "ə": "e", "ɜ": "e", "ø": "e", "œ": "e", "ɐ": "a", "ɑ": "a", "æ": "a", "ʌ": "a",
         "ɔ": "o", "ɒ": "o", "ʊ": "o", "u": "o", "w": "o", "ŋ": "n", "ɲ": "n"}
IGNORE = set("ːʲʰˈˌ")   # Längen-, Gaumen-, Behauchungs- und Betonungszeichen hört der Erkenner willkürlich


def canon(seq):
    return [CANON.get(c, c) for c in seq if c not in IGNORE]


def phonemes_of(voice, text):
    """Soll-Laute (einzelne Zeichen, ohne Betonung und Satzzeichen) nach espeak-ng für diesen Text."""
    return [c for sentence in voice.phonemize(text) for c in STRIP.sub("", "".join(sentence))]


def align(ref, hyp):
    """Levenshtein-Ausrichtung: Liste (soll, ist); None steht für eine Lücke."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, out = n, m, []
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            out.append((ref[i - 1], None)); i -= 1
        else:
            out.append((None, hyp[j - 1])); j -= 1
    return out[::-1]


class Recognizer:
    def __init__(self):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer, Wav2Vec2FeatureExtractor
        self.torch = torch
        self.tok = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(MODEL, do_phonemize=False)
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL).eval()

    def hear(self, audio, sr):
        g = gcd(16000, sr)
        a = resample_poly(audio.astype(np.float32), 16000 // g, sr // g).astype(np.float32)
        x = self.fe(a, sampling_rate=16000, return_tensors="pt").input_values
        with self.torch.no_grad():
            ids = self.model(x).logits.argmax(-1)
        return [c for p in self.tok.decode(ids[0]).split() for c in p]

    def check(self, audio, sr, expected):
        """(Fehlerquote, überhörte θ, Soll-θ) für eine Aufnahme gegen die Soll-Laute.
        θ wird getrennt gezählt: Daran erkennt man, ob wirklich kastilisch gesprochen wird (statt „s“)."""
        pad = np.zeros(int(0.2 * sr), np.float32)
        heard = self.hear(np.concatenate([pad, audio.astype(np.float32), pad]), sr)
        al = align(canon(expected), canon(heard))
        per = sum(1 for x, y in al if x != y) / max(len(expected), 1)
        theta = [y for x, y in align(expected, heard) if x == "θ"]
        return per, sum(1 for y in theta if y != "θ"), len(theta)

    def per(self, audio, sr, expected):
        """Fehlerquote einer Aufnahme gegen die Soll-Laute."""
        return self.check(audio, sr, expected)[0]


if __name__ == "__main__":
    import soundfile as sf
    from piper import PiperVoice
    path, text = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "piper/es_ES-sharvard-medium.onnx"
    audio, sr = sf.read(path, dtype="float32")
    ref = phonemes_of(PiperVoice.load(model), text)
    rec = Recognizer()
    p, miss, n = rec.check(audio, sr, ref)
    pad = np.zeros(int(0.2 * sr), np.float32)
    al = align(ref, rec.hear(np.concatenate([pad, audio, pad]), sr))
    print(f"PER {p:.2f}" + (f" · θ erkannt {n - miss}/{n}" if n else ""))
    print(" ".join(f"{x or '·'}→{y or '·'}" if x != y else x for x, y in al))
