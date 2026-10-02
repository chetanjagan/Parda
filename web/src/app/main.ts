/** Parda in the browser, session-1 UI: choose a file -> every page analysed -> boxes, redacted preview, PNG download. */
import type { Region } from "../core/boxes.js";
import { Models, SIZES_MB } from "./engines.js";
import { clearModelCache } from "./models.js";
import { loadFile, type PageImage } from "./pages.js";
import { analysePage, blackouts } from "./pipeline.js";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const COLORS: Record<string, string> = {
  AADHAAR: "#d81b60", PAN: "#8e24aa", PHONE: "#3949ab", EMAIL: "#1e88e5", PERSON_NAME: "#00897b", ADDRESS: "#43a047",
  DOB: "#7cb342", BANK_ACCOUNT: "#f4511e", IFSC: "#fb8c00", UPI_ID: "#6d4c41", GSTIN: "#546e7a", VOTER_ID: "#c0ca33",
  PASSPORT: "#5e35b1", VEHICLE_REG: "#039be5", UAN: "#00acc1", ABHA: "#ec407a", EMPLOYEE_ID: "#8d6e63", MRN: "#26a69a",
  FACE: "#e53935", SIGNATURE: "#3949ab", QR_CODE: "#212121", STAMP: "#6a1b9a",
};
const color = (label: string) => COLORS[label] ?? "#757575";

const bars = new Map<string, HTMLLIElement>();
const models = new Models((label, done, total) => {
  $("status").classList.remove("hidden");
  let li = bars.get(label);
  if (!li) {
    li = document.createElement("li");
    li.innerHTML = `<span></span><progress max="1" value="0"></progress><span></span>`;
    $("downloads").append(li);
    bars.set(label, li);
  }
  const [name, bar, mb] = li.children as unknown as [HTMLElement, HTMLProgressElement, HTMLElement];
  name.textContent = label;
  bar.value = total ? done / total : 0;
  mb.textContent = `${(done / 2 ** 20).toFixed(0)} / ${(total / 2 ** 20).toFixed(0)} MB`;
});

function step(text: string) {
  $("status").classList.remove("hidden");
  $("step").textContent = text;
}

function updateEasySize() {
  const lang = $<HTMLSelectElement>("lang").value;
  $("easysize").textContent = `(EasyOCR, +${SIZES_MB.easyBase + (SIZES_MB.easy[lang] ?? 14)} MB once; slower)`;
}

function draw(page: PageImage, regions: Region[], canvas: HTMLCanvasElement, redacted: boolean) {
  canvas.width = page.width;
  canvas.height = page.height;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(page.canvas, 0, 0);
  if (redacted) {
    ctx.fillStyle = "#000";
    for (const [x0, y0, x1, y1] of blackouts(regions, page.width, page.height)) ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    return;
  }
  ctx.lineWidth = Math.max(2, page.width / 600);
  for (const r of regions) {
    const [x0, y0, x1, y1] = r.bbox;
    ctx.strokeStyle = color(r.label);
    ctx.fillStyle = color(r.label) + "33";
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  }
}

function showPage(page: PageImage, regions: Region[], timings: Record<string, number>) {
  const card = document.createElement("section");
  card.className = "card page";
  const canvas = document.createElement("canvas");
  const side = document.createElement("div");
  const counts = new Map<string, number>();
  for (const r of regions) counts.set(r.label, (counts.get(r.label) ?? 0) + 1);
  const chips = [...counts].sort((a, b) => b[1] - a[1])
    .map(([l, n]) => `<span class="chip" style="background:${color(l)}">${l.replace("_", " ").toLowerCase()} × ${n}</span>`).join("");
  const total = Object.values(timings).reduce((s, v) => s + v, 0);
  side.innerHTML = `<h3></h3><div>${regions.length} areas to hide</div><div class="chips">${chips || "nothing found"}</div>
    <div class="actions"><button type="button" class="ghost" data-a="toggle">Preview redacted</button>
    <button type="button" data-a="png">Download redacted PNG</button></div>
    <p class="timings">${total.toFixed(1)} s on this device: ${Object.entries(timings).map(([k, v]) => `${k} ${v.toFixed(1)} s`).join(", ")}</p>`;
  side.querySelector("h3")!.textContent = page.name;
  let redacted = false;
  draw(page, regions, canvas, redacted);
  side.addEventListener("click", (e) => {
    const a = (e.target as HTMLElement).dataset.a;
    if (a === "toggle") {
      redacted = !redacted;
      (e.target as HTMLElement).textContent = redacted ? "Show what was found" : "Preview redacted";
      draw(page, regions, canvas, redacted);
    } else if (a === "png") {
      const out = document.createElement("canvas");
      draw(page, regions, out, true);
      out.toBlob((b) => {
        const link = document.createElement("a");
        link.href = URL.createObjectURL(b!);
        link.download = page.name.replace(/\.[a-z]+$/i, "").replace(/[^\w\- ]+/g, "_") + "_redacted.png";
        link.click();
        setTimeout(() => URL.revokeObjectURL(link.href), 5000);
      }, "image/png");
    }
  });
  card.append(canvas, side);
  $("pages").append(card);
}

async function run(file: File) {
  $("pages").innerHTML = "";
  const lang = $<HTMLSelectElement>("lang").value;
  const easy = $<HTMLInputElement>("easyocr").checked;
  try {
    step("Opening the file…");
    const pages = await loadFile(file);
    step("Loading the models (first time: one download, then cached)…");
    await models.loadCore();
    for (const [i, page] of pages.entries()) {
      const engines = await models.engines(page, lang, easy);
      const res = await analysePage(engines, lang, (s) => step(`Page ${i + 1} of ${pages.length}: ${s}…`));
      showPage(page, res.regions, res.timings);
    }
    step(`Done: ${pages.length} page${pages.length > 1 ? "s" : ""}. Check the boxes before downloading.`);
  } catch (err) {
    console.error(err);
    $("step").innerHTML = `<span class="error">Something went wrong: ${String((err as Error)?.message ?? err)}</span>`;
  }
}

function init() {
  updateEasySize();
  $("lang").addEventListener("change", updateEasySize);
  const input = $<HTMLInputElement>("file");
  $("pick").addEventListener("click", () => input.click());
  input.addEventListener("change", () => input.files?.[0] && run(input.files[0]));
  const drop = $("drop");
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    const f = e.dataTransfer?.files?.[0];
    if (f) run(f);
  });
  $("clear").addEventListener("click", async () => {
    await clearModelCache();
    step("Downloaded models removed from this browser.");
  });
}

init();
