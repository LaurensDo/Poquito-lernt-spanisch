#!/usr/bin/env python3
"""Prüft die Konjugationen von Poquito gegen die unabhängige Bibliothek verbecc.
   pip install verbecc && node tools/dump_verbs.js > /tmp/verbs.json && python tools/verify_verbs.py /tmp/verbs.json"""
import json, sys, logging, unicodedata
logging.disable(logging.CRITICAL)
from verbecc import CompleteConjugator, LangCodeISO639_1 as Lang
cg = CompleteConjugator(lang=Lang.es)
ours = json.load(open(sys.argv[1]))
PRON = {"yo": 0, "tú": 1, "él": 2, "nosotros": 3, "vosotros": 4, "ellos": 5}
MAP = {  # unser Name -> (Modus, Zeit)
    "presente": ("indicativo", "presente"), "perfecto": ("indicativo", "pretérito-perfecto-compuesto"), "indefinido": ("indicativo", "pretérito-perfecto-simple"),
    "imperfecto": ("indicativo", "pretérito-imperfecto"), "pluscuamperfecto": ("indicativo", "pretérito-pluscuamperfecto"), "futuro": ("indicativo", "futuro"),
    "futuroPerf": ("indicativo", "futuro-perfecto"), "condicional": ("condicional", "presente"), "condPerf": ("condicional", "perfecto"),
    "subjuntivo": ("subjuntivo", "presente"), "subjPerfecto": ("subjuntivo", "pretérito-perfecto"), "subjImperfecto": ("subjuntivo", "pretérito-imperfecto"),
    "subjPluscuam": ("subjuntivo", "pretérito-pluscuamperfecto"),
}
def forms(moods, mood, tense):
    try: items = moods[mood][tense]
    except KeyError: return None
    out = [None] * 6
    for it in items:
        pr = it.get("pr", "")
        if pr in PRON and out[PRON[pr]] is None:
            c = it["c"][0]; out[PRON[pr]] = c[len(pr) + 1:] if c.startswith(pr + " ") else c
        elif pr == "ellos" or pr == "ustedes": pass
    return out
def imp(moods, mood, tense):
    try: items = moods[mood][tense]
    except KeyError: return None
    d = {}
    for it in items:
        pr = it.get("pr", ""); c = it["c"][0]
        d[pr] = c[len(pr) + 1:] if pr and c.startswith(pr + " ") else c
    return d
COMPOUND = {"perfecto", "pluscuamperfecto", "futuroPerf", "condPerf", "subjPerfecto", "subjPluscuam"}
# Die Bibliothek kennt bei einigen Verben nur seltene Adjektiv-Partizipien (atento, presunto …) oder weicht von der RAE ab.
import re as _re
def strip(x): return "".join(c for c in unicodedata.normalize("NFD", x) if unicodedata.category(c) != "Mn")
KNOWN = {("enfriar", "*"): "RAE: enfrío (mit Akzent)", ("esparcir", "*"): "RAE: esparzo", ("gobernar", "*"): "RAE: gobierno",
         ("haber", "presente"): "hay ist die unpersönliche Form", ("valer", "imperativo"): "vale und val sind beide gültig",
         ("reír", "indefinido"): "RAE 2010: rio ohne Tilde", ("freír", "indefinido"): "RAE 2010: frio ohne Tilde", ("sonreír", "indefinido"): "RAE 2010: sonrio ohne Tilde",
         ("satisfacer", "*"): "beide Formen gültig", ("ir", "imperativo"): "vamos ist die übliche Form"}
bad = 0; checked = 0; missing = []; ignored = []
for inf, o in sorted(ours.items()):
    try: c = json.loads(cg.conjugate(inf).to_json())
    except Exception as e: missing.append(inf); continue
    m = c["moods"]
    for t, (mood, tense) in MAP.items():
        want = forms(m, mood, tense)
        if want is None or None in want: continue
        checked += 1
        mine = [x.split(" ")[0] for x in o[t]] if t in COMPOUND else o[t]
        theirs = [x.split(" ")[0] for x in want] if t in COMPOUND else want
        if mine != theirs:
            if (inf, "*") in KNOWN or (inf, t) in KNOWN: ignored.append((inf, t, KNOWN.get((inf, "*")) or KNOWN[(inf, t)])); continue
            bad += 1; print(f"{inf:14s} {t:16s} unser: {o[t]} | erwartet: {want}")
    # Imperativ
    ia = imp(m, "imperativo", "afirmativo"); inn = imp(m, "imperativo", "negativo")
    if ia and not o["noImp"]:
        mine = o["imperativo"]; want = [ia.get("tú"), ia.get("usted"), ia.get("nosotros"), ia.get("vosotros"), ia.get("ustedes")]
        checked += 1
        if mine != want:
            if (inf, "*") in KNOWN or (inf, "imperativo") in KNOWN: ignored.append((inf, "imperativo", "")); continue
            if _re.search(r"(tener|venir|poner)$", inf) and [strip(x) for x in mine] == [strip(x) for x in want]: ignored.append((inf, "imperativo", "RAE: mantén, convén, compón …")); continue
            bad += 1; print(f"{inf:14s} imperativo       unser: {mine} | erwartet: {want}")
    if inn and not o["noImp"]:
        want = [inn.get("tú"), inn.get("usted"), inn.get("nosotros"), inn.get("vosotros"), inn.get("ustedes")]
        want = ["no " + w if w and not w.startswith("no ") else w for w in want]
        checked += 1
        if o["impNeg"] != want:
            if (inf, "*") in KNOWN: ignored.append((inf, "impNeg", "")); continue
            bad += 1; print(f"{inf:14s} impNeg           unser: {o['impNeg']} | erwartet: {want}")
    for key, mood, tense in (("participio", "participio", "participio"), ("gerundio", "gerundio", "gerundio")):
        try:
            items = m[mood][tense]; w = items[0]["c"][0] if isinstance(items, list) else None
        except Exception: w = None
        if w:
            checked += 1
            if o[key] != w and w.split()[-1] != o[key]:
                if key == "participio" and w.endswith(("to", "so", "cho")) and o[key].endswith(("ado", "ido")): ignored.append((inf, key, f"Bibliothek: {w}")); continue
                bad += 1; print(f"{inf:14s} {key:16s} unser: {o[key]} | erwartet: {w}")
print(f"\nIgnoriert (Eigenheiten der Bibliothek): {len(ignored)} – {sorted({i[0] for i in ignored})}")
print(f"{len(ours)} Verben, {checked} Vergleiche, {bad} Abweichungen; nicht in der Bibliothek: {missing}")
