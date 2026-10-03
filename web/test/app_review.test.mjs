// Review state (src/app/review.ts): items a person can switch off, masked Aadhaar, hide everywhere, undo.
// With everything on, the final redaction must equal the verified pipeline (= Python's Redactor.analyse).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { analysePage } from "../build/app/pipeline.js";
import { buildItems, finalRegions, maskedEnd, occurrences, Review } from "../build/app/review.js";
import * as core from "../build/core/index.js";

const G = JSON.parse(readFileSync(new URL("./goldens/core.json", import.meta.url), "utf8"));
const DETS = [{ xyxy: [10, 10, 60, 80], cls: 0, label: "FACE", conf: 0.9 }, { xyxy: [5, 5, 9, 9], cls: 2, label: "QR_CODE", conf: 0.2 }];

function stream(segs, source) {
  const { text, ordered, offs } = core.buildOcrText(segs);
  return { source, text, ordered, offs, spans: core.findRules(text) };
}

test("everything on = the verified pipeline's regions, on every real OCR page (two streams + visual)", async () => {
  for (const segs of G.inputs.pages) {
    const streams = [stream(segs, "tesseract"), stream(segs, "easyocr")];
    const page = { width: 1000, height: 1400, streams, items: buildItems(streams, DETS) };
    const ref = await analysePage({ ocr: [{ name: "tesseract", run: async () => segs }, { name: "easyocr", run: async () => segs }],
      yolo: async () => DETS }, "en");
    assert.deepEqual(finalRegions(page), ref.regions);
  }
});

test("the same text found by both OCR streams is one item (with both parts); visual below 0.25 dropped", () => {
  const segs = G.inputs.pages.find((p) => core.findRules(core.buildOcrText(p).text).length > 1);
  const streams = [stream(segs, "tesseract"), stream(segs, "easyocr")];
  const items = buildItems(streams, DETS);
  const textItems = items.filter((i) => i.kind === "text");
  assert.equal(textItems.length, streams[0].spans.length);
  assert.ok(textItems.every((i) => i.parts.length === 2 && i.parts[0].source === "tesseract" && i.parts[1].source === "easyocr"));
  assert.deepEqual(items.filter((i) => i.kind === "visual").map((i) => i.label), ["FACE"]);
});

test("switching items off removes exactly their bars; groups switch together; undo / redo", () => {
  const segs = G.inputs.pages.find((p) => core.findRules(core.buildOcrText(p).text).length > 1);
  const streams = [stream(segs, "tesseract")];
  const r = new Review({ width: 1000, height: 1400, streams, items: buildItems(streams, DETS) });
  const all = finalRegions(r.page).length;
  const first = r.page.items[0];
  r.toggle(first.id);
  assert.ok(finalRegions(r.page).length < all);
  r.setGroup("visual", false);
  assert.ok(!finalRegions(r.page).some((x) => x.kind === "visual"));
  r.undo(); r.undo();
  assert.equal(finalRegions(r.page).length, all);
  r.redo();
  assert.ok(finalRegions(r.page).length < all);
  assert.ok(r.canUndo() && r.canRedo());
  const box = r.addBox([100, 100, 200, 150]);
  assert.equal(box.label, "MANUAL");
  assert.ok(finalRegions(r.page).some((x) => x.label === "MANUAL" && x.bbox[0] === 100));
  assert.ok(!r.canRedo()); // a new edit clears redo
});

test("masked Aadhaar keeps the last 4 digits visible (any script's digits)", () => {
  assert.equal(maskedEnd("Aadhaar 6197 5241 0246 ok", 8, 22), 17);       // bar covers "6197 5241"
  assert.equal(maskedEnd("६१९७ ५२४१ ०२४६", 0, 14), 9);                    // Devanagari digits
  assert.equal(maskedEnd("123", 0, 3), 3);                                 // fewer than 4 digits: whole span
  const segs = [{ text: "Aadhaar", bbox: [10, 10, 90, 30] }, { text: "6197", bbox: [100, 10, 150, 30] },
    { text: "5241", bbox: [160, 10, 210, 30] }, { text: "0246", bbox: [220, 10, 270, 30] }];
  const streams = [stream(segs, "tesseract")];
  const page = { width: 400, height: 100, streams, items: buildItems(streams, []) };
  const full = finalRegions(page), masked = finalRegions(page, { maskAadhaar: true });
  assert.equal(full[0].label, "AADHAAR");
  assert.equal(Math.max(...full.map((x) => x.bbox[2])), 270);
  assert.equal(Math.max(...masked.map((x) => x.bbox[2])), 210);            // the last group stays visible
});

test("hide everywhere finds the other occurrences of the same text and hides them too", () => {
  const segs = [{ text: "Name:", bbox: [10, 10, 60, 30] }, { text: "Meena", bbox: [70, 10, 130, 30] }, { text: "Sharma", bbox: [140, 10, 210, 30] },
    { text: "I,", bbox: [10, 60, 25, 80] }, { text: "Meena", bbox: [30, 60, 90, 80] }, { text: "Sharma", bbox: [100, 60, 170, 80] }, { text: "declare", bbox: [180, 60, 260, 80] }];
  const st = stream(segs, "tesseract");
  st.spans = [{ label: "PERSON_NAME", start: st.text.indexOf("Meena"), end: st.text.indexOf("Meena") + 12, score: 0.9 }];
  const r = new Review({ width: 400, height: 100, streams: [st], items: buildItems([st], []) });
  assert.equal(occurrences(r.page, r.page.items[0]).length, 1);
  assert.equal(r.hideEverywhere(r.page.items[0].id), 1);
  const regs = finalRegions(r.page);
  assert.ok(regs.some((x) => x.bbox[1] === 60 && x.label === "PERSON_NAME"));  // the second line is hidden now
  assert.equal(r.hideEverywhere(r.page.items[0].id), 0);                       // nothing left to add
});

test("one label per item: the confident model label wins over a checksum hit, a weak one does not; bars unchanged", async () => {
  const { genderSpans } = await import("../build/app/gender.js");
  const segs = [{ text: "Bank", bbox: [10, 10, 60, 30] }, { text: "A/C:", bbox: [65, 10, 110, 30] }, { text: "575183312946", bbox: [120, 10, 300, 30] },
    { text: "Aadhaar:", bbox: [10, 50, 100, 70] }, { text: "6197", bbox: [120, 50, 170, 70] }, { text: "5241", bbox: [180, 50, 230, 70] }, { text: "0246", bbox: [240, 50, 290, 70] }];
  const st = stream(segs, "tesseract");
  const at = (s) => st.text.indexOf(s);
  st.spans = core.mergeSpans([
    { label: "AADHAAR", start: at("575183312946"), end: at("575183312946") + 12, score: 1.0 },      // checksum hit by chance
    { label: "BANK_ACCOUNT", start: at("575183312946"), end: at("575183312946") + 12, score: 0.995 }, // the model saw "Bank A/C"
    { label: "AADHAAR", start: at("6197"), end: at("6197") + 14, score: 1.0 },
    { label: "UAN", start: at("6197"), end: at("6197") + 14, score: 0.48 },                          // weak model guess
  ]);
  const items = buildItems([st], []);
  assert.deepEqual(items.map((i) => i.label), ["BANK_ACCOUNT", "AADHAAR"]);
  const ref = await analysePage({ ocr: [{ name: "tesseract", run: async () => segs }] }, "en");
  // what the pipeline draws for the same spans (rules-only analysePage would not have them, so compare with textRegions)
  const { regions } = await core.textRegions(segs, async () => st.spans, "tesseract");
  assert.deepEqual(finalRegions({ width: 400, height: 100, streams: [st], items }), regions);
  assert.ok(ref.regions.length > 0);
  assert.deepEqual(genderSpans("Gender: Female"), [{ label: "GENDER", start: 8, end: 14, score: 1 }]);
});

test("also hide gender: only values next to a gender label (English, Hindi, Kannada), undoable", async () => {
  const { genderSpans } = await import("../build/app/gender.js");
  const t = "Gender: Female\nलिंग / Gender: महिला / Female\nಲಿಂಗ: ಮಹಿಳೆ\nThe female applicant and Sex: M\nMale nurse";
  const vals = genderSpans(t).map((s) => t.slice(s.start, s.end));
  assert.deepEqual(vals, ["Female", "महिला", "Female", "ಮಹಿಳೆ", "M"]);   // "female applicant" and "Male nurse" left alone
  const segs = [{ text: "Gender:", bbox: [10, 10, 90, 30] }, { text: "Female", bbox: [100, 10, 170, 30] }];
  const st = stream(segs, "tesseract");
  const r = new Review({ width: 300, height: 100, streams: [st], items: buildItems([st], []) });
  assert.equal(finalRegions(r.page).length, 0);
  r.setGender(true, genderSpans);
  assert.deepEqual(finalRegions(r.page).map((x) => [x.label, x.bbox[0]]), [["GENDER", 100]]);
  r.setGender(false, genderSpans);
  assert.equal(finalRegions(r.page).length, 0);
  r.undo();
  assert.equal(finalRegions(r.page).length, 1);
});
