"""Gemeinsame Einstellungen für make_audio.py und build_manifest.py (ohne schwere Abhängigkeiten)."""
import hashlib

VOICE_NAMES = {"f": "ef_dora", "m": "em_alex"}
SPEEDS = {"vocab": 0.9, "form": 0.9, "verb": 0.9, "number": 0.9, "sound": 0.88, "sentence": 0.9, "example": 0.9, "podcast": 0.82, "sample": 0.9, "misc": 0.9}
CARRIERS = ["Ahora", "Mira:", "Dime:", "Escucha:", "Dice:", "Oye:"]
TAG = "kokoro-v1.0-es-4"


def file_id(voice, speed, text):
    return hashlib.sha1(f"{TAG}|{voice}|{speed}|{text}".encode()).hexdigest()[:12]

# Zweite Stimme: Piper mit dem Sharvard-Korpus (Spanien, kastilisch). Sprecher: 0 = männlich, 1 = weiblich.
PIPER_TAG = "piper-sharvard-es-1"
PIPER_SPEAKERS = {"f": 1, "m": 0}
PIPER_LENGTH = {"number": 1.0}   # Dehnung der Laute; sonst PIPER_LENGTH_DEFAULT (passt das Tempo an die andere Stimme an)
PIPER_LENGTH_DEFAULT = 1.15
PIPER_DIR = {"f": "kf", "m": "km"}


def piper_length(kind):
    return PIPER_LENGTH.get(kind, PIPER_LENGTH_DEFAULT)


def piper_file_id(voice, kind, text):
    return hashlib.sha1(f"{PIPER_TAG}|{voice}|{piper_length(kind)}|{text}".encode()).hexdigest()[:12]
