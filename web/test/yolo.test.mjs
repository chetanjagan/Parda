// YOLO in the browser = YOLO through ultralytics (8 real benchmark pages recorded from the shipped ONNX model:
// tools/make_yolo_goldens.py on Kaggle -> tools/pack_yolo_goldens.py -> test/goldens/yolo.json + yolo/*.png).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { decodeYolo, letterbox, visualRegions, YoloDetector } from "../build/vision/index.js";
import { readPng } from "./png.mjs";

const G = JSON.parse(readFileSync(new URL("./goldens/yolo.json", import.meta.url), "utf8"));
const NAMES = Object.fromEntries(Object.entries(G.meta.settings.names).map(([k, v]) => [Number(k), v]));
const golden = (rel) => new URL(`./goldens/${rel}`, import.meta.url).pathname;

function rawOf(c) {  // rebuild the full (1, 8, N) output: candidates far below the threshold are zeros
  const [, F, N] = c.raw_shape;
  const full = new Float32Array(F * N);
  const b = Buffer.from(c.raw_vals, "base64");
  const vals = new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  const n = c.raw_cols.length;
  c.raw_cols.forEach((col, k) => { for (let f = 0; f < F; f++) full[f * N + col] = vals[f * n + k]; });
  return { data: full, dims: c.raw_shape };
}

function sameBoxes(got, exp, msg, px = 1e-3) {
  assert.deepEqual(got.map((d) => [d.cls, d.label]), exp.map((d) => [d.cls, d.label]), msg);
  got.forEach((d, i) => {
    d.xyxy.forEach((v, k) => assert.ok(Math.abs(v - exp[i].xyxy[k]) <= px, `${msg}: box ${i} coord ${k}: ${v} vs ${exp[i].xyxy[k]}`));
    assert.ok(Math.abs(d.conf - exp[i].conf) < 1e-5, `${msg}: conf`);
  });
}

test("decoding the recorded network output gives ultralytics' boxes on all 8 pages", () => {
  for (const c of G.cases) {
    const { data, dims } = rawOf(c);
    sameBoxes(decodeYolo(data, dims, c.orig_hw[0], c.orig_hw[1], NAMES), c.boxes, c.image);
  }
  assert.ok(G.cases.reduce((n, c) => n + c.boxes.length, 0) >= 20);
});

test("letterbox: the browser's 1024x1024 input = ultralytics' input (OpenCV integer resize)", () => {
  const pages = G.cases.filter((c) => c.page_png);
  assert.ok(pages.length >= 3);
  for (const c of pages) {
    const page = readPng(golden(c.page_png));
    const gold = readPng(golden(c.input_png));
    const lb = letterbox(page.data, page.width, page.height, page.channels);
    assert.equal(lb.gain, c.gain);
    assert.deepEqual([lb.padX, lb.padY], [150, 0]);
    const plane = lb.size * lb.size;
    let differ = 0, worst = 0;
    for (let i = 0; i < plane; i++) {
      for (let ch = 0; ch < 3; ch++) {
        const mine = Math.round(lb.tensor[ch * plane + i] * 255);
        const d = Math.abs(mine - gold.data[i * gold.channels + ch]);
        if (d) differ++;
        worst = Math.max(worst, d);
      }
    }
    assert.ok(worst <= 1, `${c.image}: a pixel differs by ${worst}`);
    assert.ok(differ / (3 * plane) < 0.0005, `${c.image}: ${(100 * differ / (3 * plane)).toFixed(3)}% of values differ`);
  }
});

test("YoloDetector end to end (recorded network) + regions above the pipeline's 0.25", async () => {
  const c = G.cases[0];
  const page = readPng(golden(c.page_png));
  const det = new YoloDetector(async (input) => {
    assert.deepEqual(input.dims, [1, 3, 1024, 1024]);
    return rawOf(c);
  });
  const boxes = await det.detect(page.data, page.width, page.height, page.channels);
  sameBoxes(boxes, c.boxes, c.image);
  const regs = visualRegions(boxes);
  assert.deepEqual(regs.map((r) => [r.kind, r.label, r.source]), c.boxes.filter((b) => b.conf >= 0.25).map((b) => ["visual", b.label, "yolo"]));
});

test("overlap removal: same class IoU > 0.5 removed, exactly 0.5 kept, other classes never suppressed, max_det", () => {
  // N = 4 candidates, 2 classes; boxes as centre x, y, w, h in letterbox pixels on a 1024x1024 page
  const mk = (cands) => {
    const N = cands.length, raw = new Float32Array(6 * N);
    cands.forEach(([x, y, w, h, s0, s1], i) => { raw[i] = x; raw[N + i] = y; raw[2 * N + i] = w; raw[3 * N + i] = h; raw[4 * N + i] = s0; raw[5 * N + i] = s1; });
    return [raw, [1, 6, N]];
  };
  const names = { 0: "A", 1: "B" };
  // 100x100 boxes; shifted by 40 -> IoU 60/140 = 0.43 kept; shifted by 20 -> IoU 80/120 = 0.67 removed
  let [raw, dims] = mk([[100, 100, 100, 100, 0.9, 0], [120, 100, 100, 100, 0.8, 0], [140, 100, 100, 100, 0.7, 0], [100, 100, 100, 100, 0, 0.6]]);
  let d = decodeYolo(raw, dims, 1024, 1024, names);
  assert.deepEqual(d.map((x) => [x.label, x.conf.toFixed(1)]), [["A", "0.9"], ["A", "0.7"], ["B", "0.6"]]);
  // IoU exactly 0.5 (whole numbers, exact even in float32): [0,0,30,10] vs [10,0,40,10] -> 200 / 400
  [raw, dims] = mk([[15, 5, 30, 10, 0.9, 0], [25, 5, 30, 10, 0.8, 0]]);
  assert.equal(decodeYolo(raw, dims, 1024, 1024, names).length, 2); // not > 0.5: both kept
  [raw, dims] = mk([[15, 5, 30, 10, 0.9, 0], [24, 5, 30, 10, 0.8, 0]]); // 21 / 39 = 0.54: removed
  assert.equal(decodeYolo(raw, dims, 1024, 1024, names).length, 1);
  [raw, dims] = mk([[100, 100, 10, 10, 0.9, 0], [300, 300, 10, 10, 0.8, 0], [500, 500, 10, 10, 0.06, 0], [700, 700, 10, 10, 0.05, 0]]);
  assert.equal(decodeYolo(raw, dims, 1024, 1024, names).length, 3);              // 0.05 is not > 0.05
  assert.equal(decodeYolo(raw, dims, 1024, 1024, names, 1024, { maxDet: 2 }).length, 2);
});

test("landscape images (ID cards) get gray padding above and below, and boxes map back through it", () => {
  // 300 x 100 black image: ultralytics -> resized to 1024 x 341, top 341, bottom 342, no side padding
  const w = 300, h = 100;
  const lb = letterbox(new Uint8Array(w * h * 3), w, h, 3);
  assert.deepEqual([lb.padX, lb.padY], [0, 341]);
  const plane = 1024 * 1024, at = (x, y) => lb.tensor[y * 1024 + x];
  assert.equal(Math.round(at(500, 340) * 255), 114);   // last gray row above the image
  assert.equal(Math.round(at(500, 341) * 255), 0);     // first image row
  assert.equal(Math.round(at(500, 681) * 255), 0);     // last image row (341 + 341 - 1)
  assert.equal(Math.round(at(500, 682) * 255), 114);   // gray below
  assert.equal(lb.tensor.length, 3 * plane);
  // a detection covering the whole image area in letterbox pixels maps back to the whole image
  const N = 1, raw = new Float32Array(8 * N);
  raw.set([512, 341 + 341 / 2, 1024, 341, 0.9, 0, 0, 0]);
  const [d] = decodeYolo(raw, [1, 8, N], h, w, NAMES);
  d.xyxy.forEach((v, k) => assert.ok(Math.abs(v - [0, 0, 300, 100][k]) < 0.5, `coord ${k}: ${v}`));
});
