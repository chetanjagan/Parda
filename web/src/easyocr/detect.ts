/** EasyOCR's text detection around the CRAFT network (easyocr 1.7.2, readtext defaults). */
import { resizeLinear } from "./imgops.js";

const f32 = Math.fround;
export type Poly = number[]; // 8 ints: x1 y1 x2 y2 x3 y3 x4 y4 (clockwise from top-left)

export interface DetectorInput { tensor: Float32Array; dims: number[]; ratio: number }

/** resize_aspect_ratio (canvas 2560, mag_ratio 1) + normalizeMeanVariance -> 1 x 3 x H32 x W32 (RGB). */
export function detectorInput(pixels: Uint8Array | Uint8ClampedArray, width: number, height: number, channels = 4,
  canvas = 2560, mag = 1.0): DetectorInput {
  let target = mag * Math.max(height, width);
  if (target > canvas) target = canvas;
  const ratio = target / Math.max(height, width);
  const th = Math.trunc(height * ratio), tw = Math.trunc(width * ratio);
  let rgb: Uint8Array = new Uint8Array(width * height * 3);
  for (let i = 0, p = 0; i < width * height; i++, p += channels) {
    rgb[i * 3] = pixels[p]; rgb[i * 3 + 1] = pixels[p + 1]; rgb[i * 3 + 2] = pixels[p + 2];
  }
  if (tw !== width || th !== height) rgb = resizeLinear(rgb, width, height, tw, th, 3);
  const H = th % 32 ? th + 32 - (th % 32) : th, W = tw % 32 ? tw + 32 - (tw % 32) : tw;
  const mean = [0.485, 0.456, 0.406].map((m) => f32(m * 255)), std = [0.229, 0.224, 0.225].map((s) => f32(s * 255));
  const t = new Float32Array(3 * H * W);
  for (let c = 0; c < 3; c++) {
    const base = c * H * W, fill = f32(f32(0 - mean[c]) / std[c]); // the zero padding, normalised
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const v = y < th && x < tw ? rgb[(y * tw + x) * 3 + c] : -1;
        t[base + y * W + x] = v < 0 ? fill : f32(f32(v - mean[c]) / std[c]);
      }
    }
  }
  return { tensor: t, dims: [1, 3, H, W], ratio };
}

// ------------------------------------------------------------------ score maps -> boxes (getDetBoxes_core)
/** connectedComponentsWithStats(connectivity 4): labels in raster order of each component's first pixel. */
export function components4(bin: Uint8Array, w: number, h: number) {
  const labels = new Int32Array(w * h);
  const parent: number[] = [0];
  const find = (a: number): number => {
    while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; }
    return a;
  };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!bin[i]) continue;
      const up = y && bin[i - w] ? labels[i - w] : 0, left = x && bin[i - 1] ? labels[i - 1] : 0;
      if (up && left) {
        const a = find(up), b = find(left), r = Math.min(a, b);
        parent[a] = r; parent[b] = r; labels[i] = r;
      } else if (up || left) labels[i] = find(up || left);
      else { parent.push(parent.length); labels[i] = parent.length - 1; }
    }
  }
  const remap = new Map<number, number>();
  for (let i = 0; i < labels.length; i++) {
    if (!labels[i]) continue;
    const r = find(labels[i]);
    if (!remap.has(r)) remap.set(r, remap.size + 1);
    labels[i] = remap.get(r)!;
  }
  const n = remap.size + 1;
  const stats = Array.from({ length: n }, () => ({ x0: Infinity, y0: Infinity, x1: -1, y1: -1, area: 0 }));
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const k = labels[y * w + x];
    if (!k) continue;
    const s = stats[k];
    s.area++; if (x < s.x0) s.x0 = x; if (x > s.x1) s.x1 = x; if (y < s.y0) s.y0 = y; if (y > s.y1) s.y1 = y;
  }
  return { n, labels, stats };
}

type Pt = [number, number];

function hull(points: Pt[]): Pt[] {
  const seen = new Set<string>(), pts: Pt[] = [];
  for (const p of points) { const k = p[0] + "," + p[1]; if (!seen.has(k)) { seen.add(k); pts.push(p); } }
  pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length <= 2) return pts;
  const cross = (o: Pt, a: Pt, b: Pt) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo: Pt[] = [], up: Pt[] = [];
  for (const p of pts) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); }
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], p) <= 0) up.pop();
    up.push(p);
  }
  return [...lo.slice(0, -1), ...up.slice(0, -1)];
}

/** Smallest-area enclosing rectangle (one side on a hull edge), corners in order. */
export function minAreaRectCorners(points: Pt[]): Pt[] {
  const H = hull(points);
  if (H.length === 1) return [H[0], H[0], H[0], H[0]];
  let best: number[] | null = null;
  for (let i = 0; i < H.length; i++) {
    const ex = H[(i + 1) % H.length][0] - H[i][0], ey = H[(i + 1) % H.length][1] - H[i][1];
    const L = Math.sqrt(ex * ex + ey * ey);
    if (L === 0) continue;
    const ux = ex / L, uy = ey / L, vx = -uy, vy = ux;
    let a0 = Infinity, a1 = -Infinity, b0 = Infinity, b1 = -Infinity;
    for (const [x, y] of H) {
      const pu = x * ux + y * uy, pv = x * vx + y * vy;
      if (pu < a0) a0 = pu; if (pu > a1) a1 = pu; if (pv < b0) b0 = pv; if (pv > b1) b1 = pv;
    }
    const area = (a1 - a0) * (b1 - b0);
    if (!best || area < best[0] - 1e-9) best = [area, ux, uy, vx, vy, a0, a1, b0, b1];
  }
  const [, ux, uy, vx, vy, a0, a1, b0, b1] = best!;
  const P = (a: number, b: number): Pt => [a * ux + b * vx, a * uy + b * vy];
  return [P(a0, b0), P(a1, b0), P(a1, b1), P(a0, b1)];
}

/** getDetBoxes (+ adjustResultCoordinates and get_textbox's int32 cast): CRAFT maps (h2 x w2 x 2) -> text boxes. */
export function detBoxes(maps: Float32Array | number[], h2: number, w2: number, ratio: number, textThreshold = 0.7,
  linkThreshold = 0.4, lowText = 0.4): Poly[] {
  const n = h2 * w2, text = new Float32Array(n), link = new Float32Array(n);
  for (let i = 0; i < n; i++) { text[i] = maps[i * 2]; link[i] = maps[i * 2 + 1]; }
  const lowT = f32(lowText), linkT = f32(linkThreshold), textT = f32(textThreshold);
  const textScore = new Uint8Array(n), linkScore = new Uint8Array(n), comb = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    textScore[i] = text[i] > lowT ? 1 : 0;
    linkScore[i] = link[i] > linkT ? 1 : 0;
    comb[i] = textScore[i] | linkScore[i];
  }
  const { n: nLabels, labels, stats } = components4(comb, w2, h2);
  const boxes: Poly[] = [];
  const scale = (1 / ratio) * 2;
  const segmap = new Uint8Array(n);
  for (let k = 1; k < nLabels; k++) {
    const s = stats[k];
    const size = s.area;
    if (size < 10) continue;
    let mx = -Infinity;
    for (let i = 0; i < n; i++) if (labels[i] === k && text[i] > mx) mx = text[i];
    if (mx < textT) continue;
    segmap.fill(0);
    for (let i = 0; i < n; i++) if (labels[i] === k && !(linkScore[i] === 1 && textScore[i] === 0)) segmap[i] = 255;
    const x = s.x0, y = s.y0, w = s.x1 - s.x0 + 1, h = s.y1 - s.y0 + 1;
    const niter = Math.trunc(Math.sqrt((size * Math.min(w, h)) / (w * h)) * 2);
    const sx = Math.max(0, x - niter), ex = Math.min(w2, x + w + niter + 1);
    const sy = Math.max(0, y - niter), ey = Math.min(h2, y + h + niter + 1);
    const kk = 1 + niter, a = Math.trunc(kk / 2);
    if (kk > 1) { // dilate the window with a kk x kk rectangle (anchor at kk // 2)
      const ww = ex - sx, hh = ey - sy, src = new Uint8Array(ww * hh);
      for (let yy = 0; yy < hh; yy++) for (let xx = 0; xx < ww; xx++) src[yy * ww + xx] = segmap[(sy + yy) * w2 + sx + xx];
      for (let yy = 0; yy < hh; yy++) for (let xx = 0; xx < ww; xx++) {
        let m = 0;
        for (let dy = -a; dy < kk - a && !m; dy++) for (let dx = -a; dx < kk - a; dx++) {
          const y2 = yy + dy, x2 = xx + dx;
          if (y2 >= 0 && y2 < hh && x2 >= 0 && x2 < ww && src[y2 * ww + x2]) { m = 255; break; }
        }
        segmap[(sy + yy) * w2 + sx + xx] = m;
      }
    }
    const pts: Pt[] = [];
    for (let yy = 0; yy < h2; yy++) for (let xx = 0; xx < w2; xx++) if (segmap[yy * w2 + xx]) pts.push([xx, yy]);
    let box = minAreaRectCorners(pts);
    const bw = Math.hypot(box[0][0] - box[1][0], box[0][1] - box[1][1]), bh = Math.hypot(box[1][0] - box[2][0], box[1][1] - box[2][1]);
    if (Math.abs(1 - Math.max(bw, bh) / (Math.min(bw, bh) + 1e-5)) <= 0.1) {
      let l = Infinity, r = -Infinity, t = Infinity, b = -Infinity;
      for (const [px, py] of pts) { if (px < l) l = px; if (px > r) r = px; if (py < t) t = py; if (py > b) b = py; }
      box = [[l, t], [r, t], [r, b], [l, b]];
    }
    let start = 0;
    for (let i = 1; i < 4; i++) if (box[i][0] + box[i][1] < box[start][0] + box[start][1]) start = i;
    box = [...box.slice(start), ...box.slice(0, start)];
    boxes.push(box.flatMap(([px, py]) => [Math.trunc(px * scale), Math.trunc(py * scale)]));
  }
  return boxes;
}

// ------------------------------------------------------------------ grouping into lines (utils.group_text_box)
export type HBox = [number, number, number, number]; // x_min, x_max, y_min, y_max
export type FreeBox = [Pt, Pt, Pt, Pt];

const mean = (xs: number[]) => xs.reduce((s, v) => s + v, 0) / xs.length;

export function groupTextBox(polys: Poly[], slopeThs = 0.1, ycenterThs = 0.5, heightThs = 0.5, widthThs = 0.5,
  addMargin = 0.1): { horizontal: HBox[]; free: FreeBox[] } {
  let hl: number[][] = [];
  const free: FreeBox[] = [];
  for (const p of polys) {
    const slopeUp = (p[3] - p[1]) / Math.max(10, p[2] - p[0]);
    const slopeDown = (p[5] - p[7]) / Math.max(10, p[4] - p[6]);
    if (Math.max(Math.abs(slopeUp), Math.abs(slopeDown)) < slopeThs) {
      const xs = [p[0], p[2], p[4], p[6]], ys = [p[1], p[3], p[5], p[7]];
      const xMax = Math.max(...xs), xMin = Math.min(...xs), yMax = Math.max(...ys), yMin = Math.min(...ys);
      hl.push([xMin, xMax, yMin, yMax, 0.5 * (yMin + yMax), yMax - yMin]);
    } else {
      const height = Math.hypot(p[6] - p[0], p[7] - p[1]), width = Math.hypot(p[2] - p[0], p[3] - p[1]);
      const margin = Math.trunc(1.44 * addMargin * Math.min(width, height));
      const t13 = Math.abs(Math.atan((p[1] - p[5]) / Math.max(10, p[0] - p[4])));
      const t24 = Math.abs(Math.atan((p[3] - p[7]) / Math.max(10, p[2] - p[6])));
      free.push([[p[0] - Math.cos(t13) * margin, p[1] - Math.sin(t13) * margin],
        [p[2] + Math.cos(t24) * margin, p[3] - Math.sin(t24) * margin],
        [p[4] + Math.cos(t13) * margin, p[5] + Math.sin(t13) * margin],
        [p[6] - Math.cos(t24) * margin, p[7] + Math.sin(t24) * margin]]);
    }
  }
  hl = hl.map((b, i) => ({ b, i })).sort((a, b) => a.b[4] - b.b[4] || a.i - b.i).map((x) => x.b);
  const combined: number[][][] = [];
  let newBox: number[][] = [], bHeight: number[] = [], bYcenter: number[] = [];
  for (const poly of hl) {
    if (!newBox.length) { bHeight = [poly[5]]; bYcenter = [poly[4]]; newBox.push(poly); }
    else if (Math.abs(mean(bYcenter) - poly[4]) < ycenterThs * mean(bHeight)) {
      bHeight.push(poly[5]); bYcenter.push(poly[4]); newBox.push(poly);
    } else { bHeight = [poly[5]]; bYcenter = [poly[4]]; combined.push(newBox); newBox = [poly]; }
  }
  combined.push(newBox);
  const merged: HBox[] = [];
  for (const boxes0 of combined) {
    if (boxes0.length === 1) {
      const box = boxes0[0];
      const margin = Math.trunc(addMargin * Math.min(box[1] - box[0], box[5]));
      merged.push([box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin]);
    } else if (boxes0.length > 1) {
      const boxes = boxes0.map((b, i) => ({ b, i })).sort((a, b) => a.b[0] - b.b[0] || a.i - b.i).map((x) => x.b);
      const mergedBox: number[][][] = [];
      let nb: number[][] = [], bh: number[] = [], xMax = 0;
      for (const box of boxes) {
        if (!nb.length) { bh = [box[5]]; xMax = box[1]; nb.push(box); }
        else if (Math.abs(mean(bh) - box[5]) < heightThs * mean(bh) && box[0] - xMax < widthThs * (box[3] - box[2])) {
          bh.push(box[5]); xMax = box[1]; nb.push(box);
        } else { bh = [box[5]]; xMax = box[1]; mergedBox.push(nb); nb = [box]; }
      }
      if (nb.length) mergedBox.push(nb);
      for (const mbox of mergedBox) {
        if (mbox.length !== 1) {
          const pick = (k: number, lo: boolean) => mbox.reduce((m, b) => (lo ? b[k] < m[k] : b[k] > m[k]) ? b : m)[k];
          const x0 = pick(0, true), x1 = pick(1, false), y0 = pick(2, true), y1 = pick(3, false);
          const margin = Math.trunc(addMargin * Math.min(x1 - x0, y1 - y0));
          merged.push([x0 - margin, x1 + margin, y0 - margin, y1 + margin]);
        } else {
          const box = mbox[0];
          const margin = Math.trunc(addMargin * Math.min(box[1] - box[0], box[3] - box[2]));
          merged.push([box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin]);
        }
      }
    }
  }
  return { horizontal: merged, free };
}

/** detect()'s min_size filter. */
export function filterMinSize(g: { horizontal: HBox[]; free: FreeBox[] }, minSize = 20) {
  const diff = (xs: number[]) => Math.max(...xs) - Math.min(...xs);
  return {
    horizontal: g.horizontal.filter((i) => Math.max(i[1] - i[0], i[3] - i[2]) > minSize),
    free: g.free.filter((i) => Math.max(diff(i.map((c) => c[0])), diff(i.map((c) => c[1]))) > minSize),
  };
}
