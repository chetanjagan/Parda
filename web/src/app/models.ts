/** Model files: downloaded once from the public Hugging Face repo, cached in the browser (Cache Storage), and turned
 *  into onnxruntime-web sessions. onnxruntime-web, Tesseract.js and transformers.js come from the import map. */
import type { Feed, OnnxMeta } from "../gliner/processor.js";

export const MODEL_REPO = "https://huggingface.co/chetan-0804/parda-web-models/resolve/main";
export const CACHE_NAME = "parda-models-v1";

export type Progress = (label: string, done: number, total: number) => void;

/** Bytes of a file, from the browser cache or downloaded (with progress) and then cached. */
export async function fetchCached(url: string, label: string, progress?: Progress): Promise<Uint8Array> {
  const cache = typeof caches !== "undefined" ? await caches.open(CACHE_NAME) : null;
  const hit = cache ? await cache.match(url) : undefined;
  if (hit) {
    const b = new Uint8Array(await hit.arrayBuffer());
    progress?.(label, b.length, b.length);
    return b;
  }
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`download failed (${res.status}): ${url}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const parts: Uint8Array[] = [];
  let done = 0;
  for (;;) {
    const { value, done: end } = await reader.read();
    if (end) break;
    parts.push(value);
    done += value.length;
    progress?.(label, done, total || done);
  }
  const bytes = new Uint8Array(done);
  let o = 0;
  for (const p of parts) { bytes.set(p, o); o += p.length; }
  try {
    await cache?.put(url, new Response(bytes, { headers: { "content-type": "application/octet-stream" } }));
  } catch { /* storage full: keep working without the cache */ }
  return bytes;
}

export async function fetchJson<T>(url: string, label: string, progress?: Progress): Promise<T> {
  return JSON.parse(new TextDecoder().decode(await fetchCached(url, label, progress))) as T;
}

/** Remove every cached model (a button in the app). */
export async function clearModelCache(): Promise<void> {
  if (typeof caches !== "undefined") await caches.delete(CACHE_NAME);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Ort = any;
let ortPromise: Promise<Ort> | null = null;

export function ort(): Promise<Ort> {
  ortPromise ??= (async () => {
    const name = "onnxruntime-web";
    const m: Ort = await import(/* @vite-ignore */ name);
    const o = m.default ?? m;
    o.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    o.env.wasm.proxy = false; // a cross-origin proxy worker is fragile; the app yields between steps instead
    o.env.wasm.numThreads = 1; // threads need cross-origin isolation, which static hosting cannot set
    return o;
  })();
  return ortPromise;
}

export async function session(bytes: Uint8Array): Promise<Ort> {
  const o = await ort();
  return o.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
}

/** GLiNER's feeds (plain arrays) -> onnxruntime tensors. */
export async function toTensors(feeds: Record<string, Feed>): Promise<Record<string, unknown>> {
  const o = await ort();
  const out: Record<string, unknown> = {};
  for (const [k, f] of Object.entries(feeds)) {
    out[k] = f.type === "bool"
      ? new o.Tensor("bool", Uint8Array.from(f.data as boolean[], (v) => (v ? 1 : 0)), f.dims)
      : new o.Tensor("int64", BigInt64Array.from(f.data as number[], (v) => BigInt(v)), f.dims);
  }
  return out;
}

export interface Runner { (input: { data: Float32Array; dims: number[] }): Promise<{ data: Float32Array; dims: number[] }> }

/** A single-input, single-output float model (YOLO, EasyOCR) as a runner. */
export async function floatRunner(s: Ort): Promise<Runner> {
  const o = await ort();
  const inName = s.inputNames[0], outName = s.outputNames[0];
  return async ({ data, dims }) => {
    const r = await s.run({ [inName]: new o.Tensor("float32", data, dims) });
    return { data: r[outName].data as Float32Array, dims: r[outName].dims as number[] };
  };
}

export type { OnnxMeta };
