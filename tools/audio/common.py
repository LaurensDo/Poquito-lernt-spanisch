"""Gemeinsame Einstellungen für make_audio.py und build_manifest.py (ohne schwere Abhängigkeiten)."""
import hashlib

VOICE_NAMES = {"f": "ef_dora", "m": "em_alex"}
SPEEDS = {"vocab": 0.9, "form": 0.9, "verb": 0.9, "number": 0.9, "sound": 0.88, "sentence": 0.9, "example": 0.9, "podcast": 0.82, "sample": 0.9, "misc": 0.9}
CARRIERS = ["Ahora", "Mira:", "Dime:", "Escucha:", "Dice:", "Oye:"]
TAG = "kokoro-v1.0-es-4"


def file_id(voice, speed, text):
    return hashlib.sha1(f"{TAG}|{voice}|{speed}|{text}".encode()).hexdigest()[:12]
