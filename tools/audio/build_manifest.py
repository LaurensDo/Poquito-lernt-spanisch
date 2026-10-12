#!/usr/bin/env python3
"""Baut audio/manifest.json aus den Erzeugungsprotokollen.
Aufgenommen wird nur, was zum aktuellen Stand von texts.json und den Einstellungen in common.py passt und als Datei vorliegt."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import SPEEDS, file_id

root = os.path.join(os.path.dirname(__file__), "..", "..", "audio")
texts = json.load(open(os.path.join(os.path.dirname(__file__), "texts.json")))
wanted = {file_id(t["voice"], SPEEDS.get(t["kind"], 0.9), t["text"]): t for t in texts}
ms = {}
for path in sorted(glob.glob(os.path.join(root, ".meta-*.jsonl"))):
    for line in open(path):
        e = json.loads(line)
        ms[e["id"]] = e["ms"]
clips, missing = {}, 0
for fid, t in wanted.items():
    if fid in ms and os.path.exists(os.path.join(root, t["voice"], fid + ".mp3")):
        clips.setdefault(t["text"], {})[t["voice"]] = [fid, ms[fid]]
    else:
        missing += 1
# Dateien, die nicht mehr zum Manifest gehören, entfernen
keep = {v[0] for c in clips.values() for v in c.values()}
for f in glob.glob(os.path.join(root, "*", "*.mp3")):
    if os.path.basename(f)[:-4] not in keep:
        os.remove(f)
json.dump({"v": 1, "clips": clips}, open(os.path.join(root, "manifest.json"), "w"), ensure_ascii=False, separators=(",", ":"))
size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(root, "*", "*.mp3"))) / 1e6
print(f"{sum(len(v) for v in clips.values())} Aufnahmen für {len(clips)} Texte, {size:.1f} MB; {missing} fehlen noch")
