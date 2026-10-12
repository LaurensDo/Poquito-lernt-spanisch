"""Gemeinsame Hilfen der Audio-Werkzeuge: Lautform für Vergleiche, Nachbearbeitung, MP3-Kodierung."""
import re, unicodedata
import numpy as np
import lameenc

SR = 24000


UNITS = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve", "veinte", "veintiuno", "veintidós", "veintitrés", "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho", "veintinueve"]
TENS = {3: "treinta", 4: "cuarenta", 5: "cincuenta", 6: "sesenta", 7: "setenta", 8: "ochenta", 9: "noventa"}


def number_words(n):
    if n < 30:
        return UNITS[n]
    if n == 100:
        return "cien"
    return TENS[n // 10] + (" y " + UNITS[n % 10] if n % 10 else "")


def key(s):
    """Lautliche Form: Schreibvarianten, die man nicht hört (b/v, h, z/c/s, ll/y …), werden angeglichen."""
    s = re.sub(r"\d+", lambda m: number_words(int(m.group())) if int(m.group()) <= 100 else m.group(), s)
    s = unicodedata.normalize("NFD", s.lower()).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    for a, b in (("qu", "k"), ("ll", "y"), ("v", "b"), ("z", "s"), ("ce", "se"), ("ci", "si"), ("c", "k"), ("h", ""), ("w", "u"), ("x", "ks")):
        s = s.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", s).replace(" ", "")


def lev(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return d[len(b)]


def same(hyp, ref, tol=0.0):
    a, b = key(hyp), key(ref)
    if a == b:
        return True
    return tol > 0 and lev(a, b) <= max(1, int(len(b) * tol))


def clean(audio):
    """Stille vorn/hinten kürzen, Lautstärke angleichen, kurze Pausen setzen."""
    a = audio.astype(np.float32)
    idx = np.where(np.abs(a) > 10 ** (-46 / 20) * max(float(np.max(np.abs(a))), 1e-6))[0]
    if len(idx):
        a = a[max(idx[0] - int(0.02 * SR), 0): idx[-1] + int(0.06 * SR)]
    rms = float(np.sqrt(np.mean(a ** 2))) or 1e-6
    a = a * (10 ** (-21 / 20) / rms)
    peak = float(np.max(np.abs(a))) or 1.0
    if peak > 0.95:
        a = a * (0.95 / peak)
    fade = int(0.012 * SR)
    a[:fade] *= np.linspace(0, 1, fade)
    a[-fade:] *= np.linspace(1, 0, fade)
    return np.concatenate([np.zeros(int(0.04 * SR), dtype=np.float32), a, np.zeros(int(0.10 * SR), dtype=np.float32)])


def to_mp3(a, kbps=40):
    enc = lameenc.Encoder()
    enc.set_bit_rate(kbps); enc.set_in_sample_rate(SR); enc.set_channels(1); enc.set_quality(2)
    pcm = (np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes()
    return bytes(enc.encode(pcm)) + bytes(enc.flush())
