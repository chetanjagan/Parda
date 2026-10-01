/**
 * ultralytics' LetterBox for one image (8.4.x, auto=False, center=True, gray 114), with OpenCV's INTER_LINEAR resize in
 * its own integer arithmetic (11-bit weights, the vectorised row mix), so the network sees the same pixels as in
 * Python: verified within 1 brightness level on <= 0.012% of pixels.
 */
export interface Letterboxed {
  /** 1 x 3 x S x S, RGB, 0..1 (what the network takes) */
  tensor: Float32Array;
  size: number;
  gain: number;
  padX: number;
  padY: number;
}

/** Python round(): halves go to the even neighbour (OpenCV's cvRound does the same). */
export function roundHalfEven(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

function axis(nOut: number, nIn: number, scale: number) {
  const i0 = new Int32Array(nOut);
  const i1 = new Int32Array(nOut);
  const c0 = new Int32Array(nOut);
  const c1 = new Int32Array(nOut);
  for (let d = 0; d < nOut; d++) {
    const f = (d + 0.5) * scale - 0.5;
    let s = Math.floor(f);
    let a = f - s;
    if (s < 0) {
      a = 0;
      s = 0;
    }
    if (s >= nIn - 1) {
      a = 0;
      s = nIn - 1;
    }
    i0[d] = s;
    i1[d] = Math.min(s + 1, nIn - 1);
    c0[d] = roundHalfEven((1 - a) * 2048);
    c1[d] = 2048 - c0[d];
  }
  return { i0, i1, c0, c1 };
}

/**
 * pixels: RGB or RGBA bytes, row-major (e.g. canvas ImageData.data with channels = 4).
 * Returns the 1024x1024 network input and the numbers needed to map boxes back to the page.
 */
export function letterbox(pixels: Uint8Array | Uint8ClampedArray, w0: number, h0: number, channels = 4,
  size = 1024, pad = 114): Letterboxed {
  const r = Math.min(size / h0, size / w0);
  const nw = roundHalfEven(w0 * r);
  const nh = roundHalfEven(h0 * r);
  const dw = (size - nw) / 2;
  const dh = (size - nh) / 2;
  const top = roundHalfEven(dh - 0.1);
  const left = roundHalfEven(dw - 0.1);
  const X = axis(nw, w0, w0 / nw);
  const Y = axis(nh, h0, h0 / nh);
  const plane = size * size;
  const t = new Float32Array(3 * plane).fill(pad / 255);
  // horizontal pass of one source row into integer sums (x 2048)
  const rowCache = new Map<number, Int32Array>();
  const hrow = (y: number): Int32Array => {
    let row = rowCache.get(y);
    if (row) return row;
    row = new Int32Array(nw * 3);
    const base = y * w0 * channels;
    for (let x = 0; x < nw; x++) {
      const p0 = base + X.i0[x] * channels;
      const p1 = base + X.i1[x] * channels;
      for (let c = 0; c < 3; c++) row[x * 3 + c] = pixels[p0 + c] * X.c0[x] + pixels[p1 + c] * X.c1[x];
    }
    if (rowCache.size > 4) rowCache.delete(rowCache.keys().next().value as number);
    rowCache.set(y, row);
    return row;
  };
  for (let y = 0; y < nh; y++) {
    const a = hrow(Y.i0[y]);
    const b = hrow(Y.i1[y]);
    const w0y = Y.c0[y];
    const w1y = Y.c1[y];
    const oy = (y + top) * size + left;
    for (let x = 0; x < nw; x++) {
      for (let c = 0; c < 3; c++) {
        // OpenCV's vectorised mix: (row >> 4) * w >> 16, both rows, + 2 >> 2
        let v = ((((a[x * 3 + c] >> 4) * w0y) >> 16) + (((b[x * 3 + c] >> 4) * w1y) >> 16) + 2) >> 2;
        v = v < 0 ? 0 : v > 255 ? 255 : v;
        t[c * plane + oy + x] = v / 255;
      }
    }
  }
  return { tensor: t, size, gain: r, padX: left, padY: top };
}
