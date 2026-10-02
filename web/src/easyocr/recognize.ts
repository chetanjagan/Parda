/** EasyOCR's recognition around its CRNN networks (easyocr 1.7.2: get_image_list, AlignCollate, recognizer_predict). */
import type { FreeBox, HBox } from "./detect.js";
import { adjustContrastGrey, type Gray, perspectiveTransform, resizeBicubicPil, resizeLinear, warpPerspective } from "./imgops.js";

const f32 = Math.fround;
const MODEL_H = 64;

export interface Crop { box: number[][]; img: Gray; maxWidth: number }

const calcRatio = (w: number, h: number) => { const r = w / h; return r < 1 ? 1 / r : r; };

/** compute_ratio_and_resize (EasyOCR passes PIL's LANCZOS constant to OpenCV, which means bilinear). */
function ratioResize(img: Gray, w: number, h: number): { img: Gray; ratio: number } {
  let ratio = w / h;
  let nw: number, nh: number;
  if (ratio < 1) { ratio = calcRatio(w, h); nw = MODEL_H; nh = Math.trunc(MODEL_H * ratio); }
  else { nw = Math.trunc(MODEL_H * ratio); nh = MODEL_H; }
  return { img: { data: resizeLinear(img.data, img.width, img.height, nw, nh, 1), width: nw, height: nh }, ratio };
}

/** get_image_list for ONE box (EasyOCR recognises each box on its own with batch_size 1). */
export function cropHorizontal(grey: Gray, box: HBox): Crop | null {
  const x0 = Math.max(0, box[0]), x1 = Math.min(box[1], grey.width), y0 = Math.max(0, box[2]), y1 = Math.min(box[3], grey.height);
  const w = x1 - x0, h = y1 - y0;
  if (w <= 0 || h <= 0 || Math.trunc(MODEL_H * calcRatio(w, h)) === 0) return null;
  const data = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) data.set(grey.data.subarray((y0 + y) * grey.width + x0, (y0 + y) * grey.width + x1), y * w);
  const { img, ratio } = ratioResize({ data, width: w, height: h }, w, h);
  return { box: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], img, maxWidth: Math.ceil(Math.max(1, Math.ceil(ratio))) * MODEL_H };
}

export function cropFree(grey: Gray, box: FreeBox): Crop | null {
  const rect = box.map((p) => p.map(Math.fround)) as number[][];
  const [tl, tr, br, bl] = rect;
  const W = Math.max(Math.trunc(Math.hypot(br[0] - bl[0], br[1] - bl[1])), Math.trunc(Math.hypot(tr[0] - tl[0], tr[1] - tl[1])));
  const H = Math.max(Math.trunc(Math.hypot(tr[0] - br[0], tr[1] - br[1])), Math.trunc(Math.hypot(tl[0] - bl[0], tl[1] - bl[1])));
  if (W <= 0 || H <= 0) return null;
  const M = perspectiveTransform(rect, [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]);
  const warped = warpPerspective(grey, M, W, H);
  if (Math.trunc(MODEL_H * calcRatio(W, H)) === 0) return null;
  const { img, ratio } = ratioResize(warped, W, H);
  return { box: box.map((p) => [...p]), img, maxWidth: Math.ceil(Math.max(1, Math.ceil(ratio))) * MODEL_H };
}

/** AlignCollate + NormalizePAD: contrast (2nd pass), PIL bicubic to height 64, [-1, 1], right-pad with the last column. */
export function alignCollate(crop: Gray, imgW: number, adjust = 0): Float32Array {
  let img = crop;
  if (adjust > 0) img = adjustContrastGrey(img, adjust);
  const ratio = img.width / img.height;
  const rw = Math.ceil(MODEL_H * ratio) > imgW ? imgW : Math.ceil(MODEL_H * ratio);
  const r = resizeBicubicPil(img, rw, MODEL_H);
  const out = new Float32Array(MODEL_H * imgW);
  for (let y = 0; y < MODEL_H; y++) {
    for (let x = 0; x < imgW; x++) {
      const v = r.data[y * rw + Math.min(x, rw - 1)];
      out[y * imgW + x] = f32(f32(f32(v / 255) - 0.5) / 0.5);
    }
  }
  return out;
}

/** recognizer_predict for one crop: softmax, ignored characters zeroed + renormalised, greedy CTC, confidence. */
export function decodeCtc(preds: Float32Array | number[], T: number, C: number, character: string[], ignoreIdx: Set<number>) {
  const idx: number[] = [], maxp: number[] = [];
  const p = new Float64Array(C);
  for (let t = 0; t < T; t++) {
    let m = -Infinity;
    for (let c = 0; c < C; c++) m = Math.max(m, preds[t * C + c]);
    let s = 0;
    for (let c = 0; c < C; c++) { p[c] = f32(Math.exp(preds[t * C + c] - m)); s += p[c]; }
    s = f32(s);
    let s2 = 0;
    for (let c = 0; c < C; c++) { p[c] = ignoreIdx.has(c) ? 0 : f32(p[c] / s); s2 += p[c]; }
    s2 = f32(s2);
    let best = 0, bv = -1;
    for (let c = 0; c < C; c++) { const v = f32(p[c] / s2); if (v > bv) { bv = v; best = c; } }
    idx.push(best);
    if (best !== 0) maxp.push(bv);
  }
  let text = "";
  for (let t = 0; t < T; t++) if (idx[t] !== 0 && (t === 0 || idx[t] !== idx[t - 1])) text += character[idx[t] - 1];
  const probs = maxp.length ? maxp : [0];
  let prod = 1;
  for (const v of probs) prod = f32(prod * v);
  return { text, conf: Math.pow(prod, 2.0 / Math.sqrt(probs.length)) };
}
