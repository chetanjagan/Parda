/**
 * ultralytics' detection post-processing (8.4.x non_max_suppression + scale_boxes), verified on recorded outputs:
 * raw (1, 4 + nc, N): keep candidates whose best class score > conf, xywh -> xyxy, best class only, per-class NMS
 * (boxes offset by class x 7680; drop IoU > iou, highest score first), max_det, then undo padding and scale, clip.
 */
import { roundHalfEven } from "./letterbox.js";

export interface Detection { xyxy: [number, number, number, number]; cls: number; label: string; conf: number }

export interface DecodeOptions { conf?: number; iou?: number; maxDet?: number; maxNms?: number; maxWh?: number }

export function decodeYolo(raw: Float32Array | number[], dims: number[], origH: number, origW: number,
  names: Record<number, string>, size = 1024, opt: DecodeOptions = {}): Detection[] {
  const { conf = 0.05, iou = 0.5, maxDet = 300, maxNms = 30000, maxWh = 7680 } = opt;
  const [, F, N] = dims;
  const nc = F - 4;
  const confF32 = Math.fround(conf); // PyTorch compares the float32 scores with the threshold in float32
  type C = { box: [number, number, number, number]; score: number; cls: number };
  let cands: C[] = [];
  for (let i = 0; i < N; i++) {
    let best = -Infinity;
    let j = 0;
    for (let c = 0; c < nc; c++) {
      const s = raw[(4 + c) * N + i];
      if (s > best) {
        best = s;
        j = c;
      }
    }
    if (!(best > confF32)) continue;
    const x = raw[i], y = raw[N + i], w = raw[2 * N + i], h = raw[3 * N + i];
    cands.push({ box: [x - w / 2, y - h / 2, x + w / 2, y + h / 2], score: best, cls: j });
  }
  if (cands.length > maxNms) cands = [...cands].sort((a, b) => b.score - a.score).slice(0, maxNms);
  // NMS on class-offset boxes (classes never suppress each other)
  const order = cands.map((_, i) => i).sort((a, b) => cands[b].score - cands[a].score || a - b);
  const off = (c: C) => c.box.map((v) => v + c.cls * maxWh) as [number, number, number, number];
  const area = (b: number[]) => (b[2] - b[0]) * (b[3] - b[1]);
  const keep: number[] = [];
  const dead = new Uint8Array(cands.length);
  for (const i of order) {
    if (dead[i]) continue;
    keep.push(i);
    const bi = off(cands[i]);
    const ai = area(bi);
    for (const k of order) {
      if (dead[k] || k === i) continue;
      const bk = off(cands[k]);
      const iw = Math.max(0, Math.min(bi[2], bk[2]) - Math.max(bi[0], bk[0]));
      const ih = Math.max(0, Math.min(bi[3], bk[3]) - Math.max(bi[1], bk[1]));
      const inter = iw * ih;
      if (inter / (ai + area(bk) - inter) > iou) dead[k] = 1;
    }
  }
  const gain = Math.min(size / origH, size / origW);
  const padX = roundHalfEven((size - roundHalfEven(origW * gain)) / 2 - 0.1);
  const padY = roundHalfEven((size - roundHalfEven(origH * gain)) / 2 - 0.1);
  const clip = (v: number, hi: number) => Math.min(Math.max(v, 0), hi);
  return keep.slice(0, maxDet).map((i) => {
    const c = cands[i];
    const [x1, y1, x2, y2] = c.box;
    return {
      xyxy: [clip((x1 - padX) / gain, origW), clip((y1 - padY) / gain, origH), clip((x2 - padX) / gain, origW),
        clip((y2 - padY) / gain, origH)],
      cls: c.cls, label: names[c.cls] ?? String(c.cls), conf: c.score,
    };
  });
}
