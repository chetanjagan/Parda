/** Parda in the browser: start -> working -> review (switch items off, draw bars, hide everywhere, undo) -> export. */
import type { BBox } from "../core/spans.js";
import { TesseractOcr } from "../ocr/tesseract.js";
import { Analyser } from "./client.js";
import { auditOf, pdfOf, pngOf, redactedCanvas, save } from "./export.js";
import { lib, LIBS } from "./libs.js";
import { clearModelCache } from "./models.js";
import { loadFile, type PageImage } from "./pages.js";
import { genderSpans } from "./gender.js";
import { blackouts } from "./pipeline.js";
import { buildItems, finalRegions, type Group, groupOf, type Item, occurrences, Review } from "./review.js";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const SIZES: Record<string, number> = { "en": 93, "hi-en": 284, "kn-en": 94 };
const NAMES: Record<string, string> = {
  AADHAAR: "Aadhaar number", PAN: "PAN", PHONE: "Phone number", EMAIL: "Email", PERSON_NAME: "Name", ADDRESS: "Address",
  DOB: "Date of birth", BANK_ACCOUNT: "Bank account", IFSC: "IFSC code", UPI_ID: "UPI ID", GSTIN: "GSTIN",
  VOTER_ID: "Voter ID", PASSPORT: "Passport number", VEHICLE_REG: "Vehicle number", UAN: "UAN", ABHA: "ABHA ID",
  EMPLOYEE_ID: "Employee ID", MRN: "Medical record number", FACE: "Face", SIGNATURE: "Signature", QR_CODE: "QR code",
  STAMP: "Stamp", MANUAL: "Area you marked", GENDER: "Gender",
};
const GROUPS: Array<{ g: Group; title: string; css: string }> = [
  { g: "ids", title: "ID numbers", css: "--hl-ids" }, { g: "people", title: "People", css: "--hl-people" },
  { g: "contact", title: "Contact and address", css: "--hl-contact" }, { g: "visual", title: "Faces and marks", css: "--hl-visual" },
  { g: "manual", title: "Drawn by you", css: "--hl-manual" },
];
const css = (v: string) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

interface Doc { name: string; device?: { gpu: boolean; threads: number }; pages: Array<{ img: PageImage; review: Review; timings: Record<string, number> }> }
let doc: Doc | null = null;
let current = 0;
let view: "found" | "final" = "found";
const analyser = new Analyser();
let tess: TesseractOcr | null = null;

// ------------------------------------------------------------------ screens
function show(screen: "start" | "work" | "review") {
  for (const s of ["start", "work", "review"]) $(s).hidden = s !== screen;
}
const lang = () => (document.querySelector<HTMLInputElement>('input[name="lang"]:checked')?.value ?? "en");

function sizeNote() {
  $("easysize").textContent = `An extra ${SIZES[lang()]} MB download, once. Slower, but finds more on photos.`;
}

const bars = new Map<string, HTMLLIElement>();
function progress(label: string, done: number, total: number) {
  let li = bars.get(label);
  if (!li) {
    li = document.createElement("li");
    li.innerHTML = `<span class="name"></span><span class="mb"></span><span class="bar"><i></i></span>`;
    $("downloads").append(li);
    bars.set(label, li);
  }
  li.querySelector(".name")!.textContent = label;
  li.querySelector(".mb")!.textContent = total > 2 ** 20 ? `${(done / 2 ** 20).toFixed(0)} of ${(total / 2 ** 20).toFixed(0)} MB` : "ready";
  (li.querySelector(".bar i") as HTMLElement).style.width = `${total ? (100 * done) / total : 100}%`;
}

// ------------------------------------------------------------------ running
async function tesseract(): Promise<TesseractOcr> {
  if (!tess) {
    const t = await lib(LIBS.tesseract);
    const createWorker = t.createWorker ?? t.default?.createWorker;
    tess = new TesseractOcr((langs: string) => createWorker(langs, 1));
  }
  return tess;
}

async function run(file: File) {
  const l = lang(), easy = $<HTMLInputElement>("easyocr").checked;
  show("work");
  $("workfile").textContent = file.name;
  $("downloads").innerHTML = "";
  bars.clear();
  const step = (t: string) => { $("step").textContent = t; };
  try {
    step("Opening the file");
    const imgs = await loadFile(file);
    const d: Doc = { name: file.name, pages: [] };
    const ocr = await tesseract();
    for (const [i, img] of imgs.entries()) {
      const tag = imgs.length > 1 ? `Page ${i + 1} of ${imgs.length}: ` : "";
      step(`${tag}Reading the text`);
      const t0 = performance.now();
      const segs = await ocr.run(img.canvas, l);
      const tOcr = (performance.now() - t0) / 1000;
      const res = await analyser.analyse(img.rgba, img.width, img.height, l, easy, segs,
        { progress, step: (s) => step(tag + s) }, $<HTMLInputElement>("lowpower").checked);
      // context rules (core/context.ts) were measured and rejected: they hid 1.5 points less real PII (results/context_rules)
      const page = { width: img.width, height: img.height, streams: res.streams, items: buildItems(res.streams, res.dets) };
      d.pages.push({ img, review: new Review(page), timings: { ocr_tesseract: tOcr, ...res.timings } });
      d.device = res.device;
    }
    doc = d;
    current = 0;
    view = "found";
    if ($<HTMLInputElement>("gender").checked) for (const p of d.pages) p.review.setGender(true, genderSpans);
    show("review");
    render();
  } catch (err) {
    const msg = String((err as Error)?.message ?? err);
    $("step").innerHTML = "";
    const p = document.createElement("span");
    p.className = "error";
    p.textContent = msg === "Stopped." ? "Stopped. Choose a file to start again." : `This file could not be processed: ${msg}`;
    $("step").append(p);
    setTimeout(() => show("start"), msg === "Stopped." ? 800 : 6000);
  }
}

// ------------------------------------------------------------------ review
const maskOn = () => $<HTMLInputElement>("mask").checked;
const pageNow = () => doc!.pages[current];
const boxesOf = (it: Item): BBox[] => it.parts.flatMap((p) => p.boxes);

function draw() {
  const { img, review } = pageNow();
  const c = $<HTMLCanvasElement>("canvas");
  c.width = img.width;
  c.height = img.height;
  const ctx = c.getContext("2d")!;
  ctx.drawImage(img.canvas, 0, 0);
  if (view === "final") {
    ctx.fillStyle = "#000";
    for (const [x0, y0, x1, y1] of blackouts(finalRegions(review.page, { maskAadhaar: maskOn() }), img.width, img.height)) {
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    }
    return;
  }
  const lw = Math.max(2, img.width / 500);
  for (const it of review.page.items) {
    const g = GROUPS.find((x) => x.g === groupOf(it.label))!;
    for (const [x0, y0, x1, y1] of boxesOf(it)) {
      if (it.on) {
        ctx.fillStyle = css(g.css);
        ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      } else {
        ctx.setLineDash([lw * 3, lw * 2]);
        ctx.lineWidth = lw;
        ctx.strokeStyle = "rgba(24,32,58,.55)";
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
        ctx.setLineDash([]);
      }
    }
  }
}

function inspector() {
  const { review, timings } = pageNow();
  const items = review.page.items;
  const on = items.filter((i) => i.on).length;
  $("count").textContent = on === 1 ? "1 thing will be hidden" : `${on} things will be hidden`;
  const host = $("groups");
  host.innerHTML = "";
  for (const g of GROUPS) {
    const mine = items.filter((i) => groupOf(i.label) === g.g);
    if (!mine.length) continue;
    const sec = document.createElement("section");
    sec.className = "group";
    sec.innerHTML = `<header><h3><span class="swatch" style="background:var(${g.css})"></span>${g.title}</h3>
      <label class="switch small"><input type="checkbox" ${mine.some((i) => i.on) ? "checked" : ""}>
      <span class="track" aria-hidden="true"></span><span class="sr">Hide all</span></label></header><ul class="items"></ul>`;
    (sec.querySelector("header input") as HTMLInputElement).addEventListener("change", (e) => {
      review.setGroup(g.g, (e.target as HTMLInputElement).checked);
      render();
    });
    const ul = sec.querySelector("ul")!;
    for (const it of mine) {
      const li = document.createElement("li");
      if (!it.on) li.className = "off";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = it.on;
      box.setAttribute("aria-label", `Hide ${NAMES[it.label] ?? it.label}`);
      box.addEventListener("change", () => { review.toggle(it.id); render(); });
      const what = document.createElement("span");
      what.className = "what";
      what.innerHTML = `${NAMES[it.label] ?? it.label}${it.text ? " <em></em>" : ""}`;
      if (it.text) (what.querySelector("em") as HTMLElement).textContent = it.text;
      li.append(box, what);
      const more = it.text ? occurrences(review.page, it).length : 0;
      if (more) {
        const b = document.createElement("button");
        b.className = "everywhere";
        b.type = "button";
        b.textContent = `+${more} more`;
        b.title = "Hide every other place this appears";
        b.addEventListener("click", () => { review.hideEverywhere(it.id); render(); });
        li.append(b);
      } else li.append(document.createElement("span"));
      ul.append(li);
    }
    host.append(sec);
  }
  const read = $("readtext");
  read.innerHTML = "";
  for (const st of review.page.streams) {
    const h = document.createElement("h4");
    h.textContent = st.source === "easyocr" ? "Photo reader (EasyOCR)" : "Text reader (Tesseract)";
    const pre = document.createElement("pre");
    pre.textContent = st.text || "(nothing read)";
    read.append(h, pre);
  }
  const total = Object.values(timings).reduce((s, v) => s + v, 0);
  const dev = doc?.device;
  const how = dev ? (dev.gpu ? `graphics chip + ${dev.threads} CPU thread${dev.threads > 1 ? "s" : ""}` : `${dev.threads} CPU thread${dev.threads > 1 ? "s" : ""}`) : "";
  $("timing").textContent = total > 0 ? `Took ${total.toFixed(1)} s on this device${how ? ` (${how})` : ""}.` : "";
  $("maskrow").hidden = !items.some((i) => i.label === "AADHAAR");
  $<HTMLButtonElement>("undo").disabled = !review.canUndo();
  $<HTMLButtonElement>("redo").disabled = !review.canRedo();
}

function pager() {
  const nav = $("pagenav");
  nav.innerHTML = "";
  if (!doc || doc.pages.length < 2) { nav.textContent = "1 page"; return; }
  const prev = Object.assign(document.createElement("button"), { type: "button", className: "quiet", textContent: "Previous" });
  const next = Object.assign(document.createElement("button"), { type: "button", className: "quiet", textContent: "Next" });
  prev.disabled = current === 0;
  next.disabled = current === doc.pages.length - 1;
  prev.onclick = () => { current--; render(); };
  next.onclick = () => { current++; render(); };
  nav.append(prev, next, document.createTextNode(` Page ${current + 1} of ${doc.pages.length}`));
}

function render() {
  if (!doc) return;
  $("docname").textContent = doc.name;
  $("viewFound").setAttribute("aria-pressed", String(view === "found"));
  $("viewFinal").setAttribute("aria-pressed", String(view === "final"));
  pager();
  draw();
  inspector();
}

// pointer: a click on a highlight toggles it; a drag draws a new bar
function pagePoint(e: PointerEvent): [number, number] {
  const c = $<HTMLCanvasElement>("canvas"), r = c.getBoundingClientRect();
  return [((e.clientX - r.left) / r.width) * c.width, ((e.clientY - r.top) / r.height) * c.height];
}
let dragFrom: [number, number] | null = null;
let dragScreen: [number, number] = [0, 0];
const DRAG_MIN_PX = 12; // on screen: anything smaller is a click, whatever the zoom
function wirePointer() {
  const c = $<HTMLCanvasElement>("canvas");
  c.addEventListener("pointerdown", (e) => { dragFrom = pagePoint(e); dragScreen = [e.clientX, e.clientY]; c.setPointerCapture(e.pointerId); });
  c.addEventListener("pointermove", (e) => {
    if (!dragFrom || !doc) return;
    if (Math.abs(e.clientX - dragScreen[0]) < DRAG_MIN_PX && Math.abs(e.clientY - dragScreen[1]) < DRAG_MIN_PX) return;
    const [x, y] = pagePoint(e);
    draw();
    const ctx = c.getContext("2d")!;
    ctx.fillStyle = "rgba(0,0,0,.55)";
    ctx.fillRect(Math.min(dragFrom[0], x), Math.min(dragFrom[1], y), Math.abs(x - dragFrom[0]), Math.abs(y - dragFrom[1]));
  });
  c.addEventListener("pointerup", (e) => {
    if (!dragFrom || !doc) return;
    const [x, y] = pagePoint(e), [x0, y0] = dragFrom;
    dragFrom = null;
    const review = pageNow().review;
    if (Math.abs(e.clientX - dragScreen[0]) >= DRAG_MIN_PX && Math.abs(e.clientY - dragScreen[1]) >= DRAG_MIN_PX * 0.66) {
      review.addBox([Math.min(x0, x), Math.min(y0, y), Math.max(x0, x), Math.max(y0, y)]);
    } else {
      const hit = review.page.items.filter((it) => boxesOf(it).some(([a, b, cc, d]) => x >= a - 4 && x <= cc + 4 && y >= b - 4 && y <= d + 4))
        .sort((p, q) => area(p) - area(q))[0];
      if (hit) review.toggle(hit.id);
    }
    render();
  });
}
const area = (it: Item) => boxesOf(it).reduce((s, [a, b, c, d]) => s + (c - a) * (d - b), 0);

// ------------------------------------------------------------------ export
const base = () => (doc?.name ?? "document").replace(/\.[a-z0-9]+$/i, "").replace(/[^\w\- ]+/g, "_");
const finalCanvas = (i: number) => redactedCanvas(doc!.pages[i].img.canvas, finalRegions(doc!.pages[i].review.page, { maskAadhaar: maskOn() }));

async function exportPdf() {
  const b = $<HTMLButtonElement>("savePdf");
  b.disabled = true;
  b.textContent = "Preparing…";
  try { save(await pdfOf(doc!.pages.map((_, i) => finalCanvas(i))), `${base()}_redacted.pdf`); }
  finally { b.disabled = false; b.textContent = "Download PDF"; }
}

/** Open the review screen for already-analysed pages (used by run(); also handy for screenshots and tests). */
export function openReview(name: string, pages: Array<{ img: PageImage; streams: import("./review.js").Stream[];
  dets: import("../vision/yolo.js").Detection[]; timings?: Record<string, number> }>) {
  doc = { name, pages: pages.map((p) => ({ img: p.img, timings: p.timings ?? {},
    review: new Review({ width: p.img.width, height: p.img.height, streams: p.streams, items: buildItems(p.streams, p.dets) }) })) };
  current = 0;
  view = "found";
  show("review");
  render();
}

// ------------------------------------------------------------------ start
function init() {
  sizeNote();
  document.querySelectorAll('input[name="lang"]').forEach((r) => r.addEventListener("change", sizeNote));
  const input = $<HTMLInputElement>("file");
  $("pick").addEventListener("click", (e) => { e.stopPropagation(); input.click(); });
  const drop = $("drop");
  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.addEventListener("change", () => { if (input.files?.[0]) run(input.files[0]); input.value = ""; });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); const f = e.dataTransfer?.files?.[0]; if (f) run(f); });
  $("clear").addEventListener("click", async () => { await clearModelCache(); $("clear").textContent = "Removed"; });
  $("stop").addEventListener("click", () => analyser.cancel());
  $("viewFound").addEventListener("click", () => { view = "found"; render(); });
  $("viewFinal").addEventListener("click", () => { view = "final"; render(); });
  $("undo").addEventListener("click", () => { pageNow().review.undo(); render(); });
  $("redo").addEventListener("click", () => { pageNow().review.redo(); render(); });
  $("mask").addEventListener("change", () => render());
  $("gender").addEventListener("change", () => {
    if (!doc) return;
    for (const p of doc.pages) p.review.setGender($<HTMLInputElement>("gender").checked, genderSpans);
    render();
  });
  for (const id of ["easyocr", "lowpower", "gender"]) { // remembered in this browser
    const el = $<HTMLInputElement>(id);
    try { el.checked = localStorage.getItem(`parda-${id}`) === "1"; } catch { /* storage off */ }
    el.addEventListener("change", () => { try { localStorage.setItem(`parda-${id}`, el.checked ? "1" : "0"); } catch { /* storage off */ } });
  }
  $("savePdf").addEventListener("click", exportPdf);
  $("savePng").addEventListener("click", async () => save(await pngOf(finalCanvas(current)), `${base()}_page${current + 1}_redacted.png`));
  $("saveAudit").addEventListener("click", () => save(auditOf(doc!.name, { language: lang(), photoReader: $<HTMLInputElement>("easyocr").checked,
    maskedAadhaar: maskOn() }, doc!.pages.map((p) => finalRegions(p.review.page, { maskAadhaar: maskOn() }))), `${base()}_audit.json`));
  $("restart").addEventListener("click", () => { doc = null; show("start"); });
  document.addEventListener("keydown", (e) => {
    if (!doc || $("review").hidden) return;
    const z = e.key.toLowerCase() === "z" && (e.ctrlKey || e.metaKey);
    if (z) { e.preventDefault(); if (e.shiftKey) pageNow().review.redo(); else pageNow().review.undo(); render(); }
  });
  wirePointer();
}

init();
