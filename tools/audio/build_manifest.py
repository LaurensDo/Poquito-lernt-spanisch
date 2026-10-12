#!/usr/bin/env python3
"""Baut audio/manifest.json (Stimme „Natürlich“, Kokoro) oder audio/manifest-k.json (Stimme „Kastilisch“, Piper)
aus den Erzeugungsprotokollen.

  python build_manifest.py              # Kokoro  (audio/f, audio/m)
  python build_manifest.py --engine k   # Piper   (audio/kf, audio/km)

Aufgenommen wird nur, was zum aktuellen Stand von texts.json und den Einstellungen in common.py passt und als Datei vorliegt.
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import SPEEDS, PIPER_DIR, file_id, piper_file_id

ap = argparse.ArgumentParser()
ap.add_argument("--engine", choices=["n", "k"], default="n")
engine = ap.parse_args().engine

root = os.path.join(os.path.dirname(__file__), "..", "..", "audio")
texts = json.load(open(os.path.join(os.path.dirname(__file__), "texts.json")))
if engine == "k":
    dirs = {"f": PIPER_DIR["f"], "m": PIPER_DIR["m"]}
    wanted = {piper_file_id(t["voice"], t["kind"], t["text"]): t for t in texts}
    meta_glob, out_name = ".meta-k*.jsonl", "manifest-k.json"
else:
    dirs = {"f": "f", "m": "m"}
    wanted = {file_id(t["voice"], SPEEDS.get(t["kind"], 0.9), t["text"]): t for t in texts}
    meta_glob, out_name = ".meta-[0-9].jsonl", "manifest.json"

ms = {}
for path in sorted(glob.glob(os.path.join(root, meta_glob))):
    for line in open(path):
        e = json.loads(line)
        ms[e["id"]] = e["ms"]
clips, missing = {}, 0
for fid, t in wanted.items():
    if fid in ms and os.path.exists(os.path.join(root, dirs[t["voice"]], fid + ".mp3")):
        clips.setdefault(t["text"], {})[t["voice"]] = [fid, ms[fid]]
    else:
        missing += 1
# Dateien, die nicht mehr zum Manifest gehören, entfernen (nur im Verzeichnis dieser Stimme)
keep = {v[0] for c in clips.values() for v in c.values()}
for d in dirs.values():
    for f in glob.glob(os.path.join(root, d, "*.mp3")):
        if os.path.basename(f)[:-4] not in keep:
            os.remove(f)
json.dump({"v": 1, "clips": clips}, open(os.path.join(root, out_name), "w"), ensure_ascii=False, separators=(",", ":"))
size = sum(os.path.getsize(f) for d in dirs.values() for f in glob.glob(os.path.join(root, d, "*.mp3"))) / 1e6
print(f"{out_name}: {sum(len(v) for v in clips.values())} Aufnahmen für {len(clips)} Texte, {size:.1f} MB; {missing} fehlen noch")
