// Gibt die Konjugationen aller Verben der Datenbank aus (für tools/verify_verbs.py)
const fs = require("fs");
const path = require("path");
const html = fs.readFileSync(path.join(__dirname, "..", "index.html"), "utf8");
const js = html.split("<script>")[1].split("</script>")[0];
const data = js.slice(js.indexOf("{") + 1, js.indexOf("const app = document.getElementById"));
const conjSrc = js.slice(js.indexOf("  // ---------- Konjugation ----------"), js.indexOf("  function personsFor"));
const api = new Function(data + conjSrc + "; return { VERB_DB, conj, participio, gerundio };")();
const TENSES = ["presente", "perfecto", "indefinido", "imperfecto", "pluscuamperfecto", "futuro", "futuroPerf", "condicional", "condPerf", "subjuntivo", "subjPerfecto", "subjImperfecto", "subjPluscuam", "imperativo", "impNeg"];
const out = {};
api.VERB_DB.forEach((v) => {
  out[v.inf] = { group: v.group || "", noImp: !!v.noImp, participio: api.participio(v), gerundio: api.gerundio(v) };
  TENSES.forEach((t) => { out[v.inf][t] = api.conj(v, t); });
});
console.log(JSON.stringify(out));
