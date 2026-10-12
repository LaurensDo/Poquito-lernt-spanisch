#!/usr/bin/env python3
"""Baut audio/manifest.json aus den Erzeugungsprotokollen (nur vorhandene, geprüfte Dateien)."""
import glob, json, os
root = os.path.join(os.path.dirname(__file__), "..", "..", "audio")
clips = {}
for path in sorted(glob.glob(os.path.join(root, ".meta-*.jsonl"))):
    for line in open(path):
        e = json.loads(line)
        if os.path.exists(os.path.join(root, e["voice"], e["id"] + ".mp3")):
            clips.setdefault(e["text"], {})[e["voice"]] = [e["id"], e["ms"]]
json.dump({"v": 1, "clips": clips}, open(os.path.join(root, "manifest.json"), "w"), ensure_ascii=False, separators=(",", ":"))
n = sum(len(v) for v in clips.values())
size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(root, "*", "*.mp3"))) / 1e6
print(f"{n} Aufnahmen für {len(clips)} Texte, {size:.1f} MB")
