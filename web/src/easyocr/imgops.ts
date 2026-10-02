/** Image operations EasyOCR uses, reproduced for the browser (8-bit gray images unless noted). */

export interface Gray { data: Uint8Array; width: number; height: number }

const f32 = Math.fround;

/** cv2.cvtColor(RGB -> GRAY) for 8-bit: (R*9798 + G*19235 + B*3735 + 16384) >> 15 (bit-identical to OpenCV). */
export function rgbToGray(pixels: Uint8Array | Uint8ClampedArray, width: number, height: number, channels = 4): Gray {
  const out = new Uint8Array(width * height);
  for (let i = 0, p = 0; i < out.length; i++, p += channels) {
    out[i] = (pixels[p] * 9798 + pixels[p + 1] * 19235 + pixels[p + 2] * 3735 + 16384) >> 15;
  }
  return { data: out, width, height };
}

/** Python round() of a float: halves to even (for the 11-bit weights, like OpenCV's cvRound). */
function rint(x: number): number {
  const f = Math.floor(x), d = x - f;
  return d > 0.5 ? f + 1 : d < 0.5 ? f : f % 2 === 0 ? f : f + 1;
}

function linAxis(nOut: number, nIn: number) {
  const scale = nIn / nOut;
  const i0 = new Int32Array(nOut), i1 = new Int32Array(nOut), c0 = new Int32Array(nOut), c1 = new Int32Array(nOut);
  for (let d = 0; d < nOut; d++) {
    const f = f32((d + 0.5) * scale - 0.5);
    let s = Math.floor(f);
    let a = f32(f - s);
    if (s < 0) { s = 0; a = 0; }
    if (s >= nIn - 1) { s = nIn - 1; a = 0; }
    i0[d] = s;
    i1[d] = Math.min(s + 1, nIn - 1);
    c0[d] = rint(f32(f32(1 - a) * 2048));
    c1[d] = 2048 - c0[d];
  }
  return { i0, i1, c0, c1 };
}

/** cv2.resize(INTER_LINEAR) for 8-bit images with `channels` channels (OpenCV's 11-bit fixed point; equal to OpenCV
 *  when shrinking, within 1 level when enlarging in both directions). */
export function resizeLinear(src: Uint8Array, w0: number, h0: number, nw: number, nh: number, channels = 1): Uint8Array {
  if (nw === w0 && nh === h0) return src.slice();
  const X = linAxis(nw, w0), Y = linAxis(nh, h0);
  const out = new Uint8Array(nw * nh * channels);
  const row = (y: number) => {
    const r = new Int32Array(nw * channels), base = y * w0 * channels;
    for (let x = 0; x < nw; x++) {
      const p0 = base + X.i0[x] * channels, p1 = base + X.i1[x] * channels;
      for (let c = 0; c < channels; c++) r[x * channels + c] = src[p0 + c] * X.c0[x] + src[p1 + c] * X.c1[x];
    }
    return r;
  };
  let lastY = -1, lastRow = new Int32Array(0);
  const cached = (y: number) => (y === lastY ? lastRow : ((lastY = y), (lastRow = row(y))));
  for (let y = 0; y < nh; y++) {
    const a = cached(Y.i0[y]).slice();
    const b = cached(Y.i1[y]);
    const w0y = Y.c0[y], w1y = Y.c1[y], o = y * nw * channels;
    for (let i = 0; i < nw * channels; i++) {
      const v = ((((a[i] >> 4) * w0y) >> 16) + (((b[i] >> 4) * w1y) >> 16) + 2) >> 2;
      out[o + i] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  }
  return out;
}

// ------------------------------------------------------------------ Pillow's BICUBIC resize (mode "L")
const PRECISION_BITS = 32 - 8 - 2;

function bicubic(x: number): number {
  const a = -0.5;
  x = Math.abs(x);
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

function pilCoeffs(inSize: number, outSize: number) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = 2.0 * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Int32Array(outSize * ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1.0 / filterscale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    const k: number[] = [];
    let ww = 0.0;
    for (let x = 0; x < xmax; x++) {
      const w = bicubic((x + xmin - center + 0.5) * ss);
      k.push(w);
      ww += w;
    }
    for (let x = 0; x < xmax; x++) {
      const v = ww !== 0 ? k[x] / ww : k[x];
      kk[xx * ksize + x] = v < 0 ? Math.trunc(-0.5 + v * (1 << PRECISION_BITS)) : Math.trunc(0.5 + v * (1 << PRECISION_BITS));
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { ksize, bounds, kk };
}

function clip8(ss: number): number {
  const v = Math.floor(ss / (1 << PRECISION_BITS));
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

/** PIL Image.resize((nw, nh), BICUBIC) for an 8-bit gray image (Pillow's fixed-point resampling). */
export function resizeBicubicPil(img: Gray, nw: number, nh: number): Gray {
  if (nw === img.width && nh === img.height) return { data: img.data.slice(), width: nw, height: nh };
  let cur = img;
  if (nw !== img.width) {
    const { ksize, bounds, kk } = pilCoeffs(img.width, nw);
    const out = new Uint8Array(nw * img.height);
    for (let y = 0; y < img.height; y++) {
      for (let xx = 0; xx < nw; xx++) {
        const xmin = bounds[xx * 2], xmax = bounds[xx * 2 + 1];
        let ss = 1 << (PRECISION_BITS - 1);
        for (let x = 0; x < xmax; x++) ss += img.data[y * img.width + x + xmin] * kk[xx * ksize + x];
        out[y * nw + xx] = clip8(ss);
      }
    }
    cur = { data: out, width: nw, height: img.height };
  }
  if (nh !== cur.height) {
    const { ksize, bounds, kk } = pilCoeffs(cur.height, nh);
    const out = new Uint8Array(cur.width * nh);
    for (let yy = 0; yy < nh; yy++) {
      const ymin = bounds[yy * 2], ymax = bounds[yy * 2 + 1];
      for (let x = 0; x < cur.width; x++) {
        let ss = 1 << (PRECISION_BITS - 1);
        for (let y = 0; y < ymax; y++) ss += cur.data[(y + ymin) * cur.width + x] * kk[yy * ksize + y];
        out[yy * cur.width + x] = clip8(ss);
      }
    }
    cur = { data: out, width: cur.width, height: nh };
  }
  return cur;
}

// ------------------------------------------------------------------ perspective warp (tilted text)
/** cv2.getPerspectiveTransform: 3x3 matrix mapping src[i] -> dst[i] (row-major, m22 = 1). */
export function perspectiveTransform(src: number[][], dst: number[][]): number[] {
  const A: number[][] = [], b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i].map(f32), [u, v] = dst[i].map(f32);
    A.push([x, y, 1, 0, 0, 0, -x * u, -y * u]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -x * v, -y * v]); b.push(v);
  }
  for (let c = 0; c < 8; c++) { // Gaussian elimination with partial pivoting
    let p = c;
    for (let r = c + 1; r < 8; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    [b[c], b[p]] = [b[p], b[c]];
    for (let r = c + 1; r < 8; r++) {
      const f = A[r][c] / A[c][c];
      for (let k = c; k < 8; k++) A[r][k] -= f * A[c][k];
      b[r] -= f * b[c];
    }
  }
  const m = new Array(8).fill(0);
  for (let r = 7; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < 8; k++) s -= A[r][k] * m[k];
    m[r] = s / A[r][r];
  }
  return [...m, 1];
}

/** cv2.warpPerspective(INTER_LINEAR, BORDER_CONSTANT 0) of a gray image into a w x h output: OpenCV's 1/32-pixel
 *  position grid and 15-bit bilinear weights. */
export function invert3(m: number[]): number[] {
  const [a, b, c, d, e, f, g, h, i] = m;
  let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  det = det ? 1 / det : 0;
  return [(e * i - f * h) * det, (c * h - b * i) * det, (b * f - c * e) * det,
    (f * g - d * i) * det, (a * i - c * g) * det, (c * d - a * f) * det,
    (d * h - e * g) * det, (b * g - a * h) * det, (a * e - b * d) * det];
}

export function warpPerspective(src: Gray, forward: number[], w: number, h: number): Gray {
  const M = invert3(forward); // OpenCV maps each output pixel back into the source (inverse matrix)
  const out = new Uint8Array(w * h);
  const TAB = 32;
  const at = (x: number, y: number) => (x >= 0 && y >= 0 && x < src.width && y < src.height ? src.data[y * src.width + x] : 0);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const X0 = M[0] * x + M[1] * y + M[2], Y0 = M[3] * x + M[4] * y + M[5];
      let W = M[6] * x + M[7] * y + M[8];
      W = W ? TAB / W : 0;
      const X = Math.round(Math.max(-2147483648, Math.min(2147483647, X0 * W)));
      const Y = Math.round(Math.max(-2147483648, Math.min(2147483647, Y0 * W)));
      const sx = X >> 5, sy = Y >> 5, ax = (X & 31) / TAB, ay = (Y & 31) / TAB;
      const wts = [(1 - ay) * (1 - ax), (1 - ay) * ax, ay * (1 - ax), ay * ax].map((v) => Math.round(v * 32768));
      const fix = 32768 - wts.reduce((s, v) => s + v, 0); // OpenCV makes the 4 weights sum to exactly 32768
      if (fix) { let k = 0; for (let i = 1; i < 4; i++) if (wts[i] > wts[k]) k = i; wts[k] += fix; }
      const v = at(sx, sy) * wts[0] + at(sx + 1, sy) * wts[1] + at(sx, sy + 1) * wts[2] + at(sx + 1, sy + 1) * wts[3];
      const r = (v + (1 << 14)) >> 15;
      out[y * w + x] = r < 0 ? 0 : r > 255 ? 255 : r;
    }
  }
  return { data: out, width: w, height: h };
}

// ------------------------------------------------------------------ contrast (EasyOCR's second pass)
/** numpy.percentile (linear interpolation). */
export function percentile(values: Uint8Array, q: number): number {
  const s = Float64Array.from(values).sort();
  const pos = (s.length - 1) * (q / 100);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  const t = pos - lo, d = s[hi] - s[lo];
  return t >= 0.5 ? s[hi] - d * (1 - t) : s[lo] + d * t; // numpy's _lerp
}

/** easyocr.recognition.adjust_contrast_grey. */
export function adjustContrastGrey(img: Gray, target = 0.4): Gray {
  const high = percentile(img.data, 90), low = percentile(img.data, 10);
  const contrast = (high - low) / Math.max(10, high + low);
  if (!(contrast < target)) return img;
  const ratio = 200.0 / Math.max(10, high - low);
  const out = new Uint8Array(img.data.length);
  for (let i = 0; i < out.length; i++) {
    const v = (img.data[i] - low + 25) * ratio;
    out[i] = Math.trunc(Math.max(0, Math.min(255, v)));
  }
  return { data: out, width: img.width, height: img.height };
}
