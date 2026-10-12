// Sammelt alle Texte, die Poquito vorliest, aus index.html -> texts.json
// Aufruf: node tools/audio/extract_texts.js > tools/audio/texts.json
const fs = require("fs");
const path = require("path");
const html = fs.readFileSync(path.join(__dirname, "..", "..", "index.html"), "utf8");
const js = html.split("<script>")[1].split("</script>")[0];
const data = js.slice(js.indexOf("{") + 1, js.indexOf("const app = document.getElementById"));
const conjSrc = js.slice(js.indexOf("  // ---------- Konjugation ----------"), js.indexOf("  function personsFor"));
const numSrc = js.slice(js.indexOf("  function numberWord(n)"), js.indexOf("  // Zahlen, die man beim Hören"));
const api = new Function(data + conjSrc + numSrc + "; return { VOCAB, SENTENCES, PODCASTS, TENSES, SOUNDS, VERBS, VOICE_SAMPLE, conj, numberWord };")();

// voice: f = weibliche Hauptstimme, m = männliche Stimme
const out = new Map();
const add = (text, voice = "f", kind = "misc") => {
  const t = String(text).trim();
  if (t && !out.has(t + "|" + voice)) out.set(t + "|" + voice, { text: t, voice, kind });
};
api.VOCAB.forEach((w) => add(w.es, "f", "vocab"));
api.SENTENCES.forEach((s) => add(s.es, "f", "sentence"));
for (let n = 0; n <= 100; n++) add(api.numberWord(n), "f", "number");
api.VERBS.forEach((v) => {
  add(v.inf, "f", "verb");
  api.TENSES.forEach((t) => api.conj(v, t.id).forEach((form) => add(form, "f", "form")));
});
api.PODCASTS.forEach((ep) => ep.lines.forEach((l) => {
  const gender = ep.speakers ? ep.speakers[l.s].gender : ep.narrator;
  add(l.es, gender === "m" ? "m" : "f", "podcast");
}));
api.SOUNDS.forEach((snd) => {
  snd.words.forEach((w) => add(w.es, "f", "sound"));
  add(snd.words.map((w) => w.es).join(", "), "f", "sound");
});
api.TENSES.forEach((t) => t.use.forEach((u) => add(u.es, "f", "example")));
add(api.VOICE_SAMPLE, "f", "sample");
console.log(JSON.stringify([...out.values()], null, 0));
