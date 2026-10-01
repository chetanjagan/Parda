// The real network: public/yolo/parda-yolo.onnx in onnxruntime-node, fed by the browser letterbox, must give
// ultralytics' boxes on the recorded pages (within 1 px). Needs `npm install` and the model file (CI sets REQUIRE_YOLO=1).
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";
import { YoloDetector } from "../build/vision/index.js";
import { readPng } from "./png.mjs";

const G = JSON.parse(readFileSync(new URL("./goldens/yolo.json", import.meta.url), "utf8"));
const MODEL = new URL("../public/yolo/parda-yolo.onnx", import.meta.url).pathname;
let ort = null;
let why = "";
try {
  if (!existsSync(MODEL)) throw new Error(`model file missing: ${MODEL}`);
  ort = await import("onnxruntime-node");
} catch (e) {
  why = `real YOLO not available (${e.message.split("\n")[0]})`;
  if (process.env.REQUIRE_YOLO) throw new Error(`CI requires the real YOLO: ${why}`);
}

test("the real ONNX YOLO on browser-letterboxed pages gives ultralytics' boxes", { skip: !ort && why }, async () => {
  const session = await ort.InferenceSession.create(MODEL);
  const input = session.inputNames[0];
  const det = new YoloDetector(async ({ data, dims }) => {
    const out = await session.run({ [input]: new ort.Tensor("float32", data, dims) });
    const t = out[session.outputNames[0]];
    return { data: t.data, dims: t.dims };
  });
  for (const c of G.cases.filter((c) => c.page_png)) {
    const page = readPng(new URL(`./goldens/${c.page_png}`, import.meta.url).pathname);
    const got = await det.detect(page.data, page.width, page.height, page.channels);
    assert.deepEqual(got.map((d) => d.label), c.boxes.map((b) => b.label), c.image);
    got.forEach((d, i) => d.xyxy.forEach((v, k) => assert.ok(Math.abs(v - c.boxes[i].xyxy[k]) <= 1.0,
      `${c.image} box ${i}: ${v} vs ${c.boxes[i].xyxy[k]}`)));
  }
});
