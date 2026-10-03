/** The page side of the background worker: one request at a time, progress callbacks, cancel = restart the worker. */
import type { Segment } from "../core/spans.js";
import type { Detection } from "../vision/yolo.js";
import type { Stream } from "./review.js";

export interface AnalyseResult { streams: Stream[]; dets: Detection[]; timings: Record<string, number>; device?: { gpu: boolean; threads: number } }
export interface Callbacks { progress?(label: string, done: number, total: number): void; step?(text: string): void }

export class Analyser {
  private worker: Worker;
  private seq = 0;
  private pending = new Map<number, { resolve(r: AnalyseResult): void; reject(e: Error): void; cb: Callbacks }>();

  constructor() { this.worker = this.spawn(); }

  private spawn(): Worker {
    const w = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    w.onmessage = (e: MessageEvent) => {
      const m = e.data, p = this.pending.get(m.id);
      if (!p) return;
      if (m.type === "progress") p.cb.progress?.(m.label, m.done, m.total);
      else if (m.type === "step") p.cb.step?.(m.text);
      else if (m.type === "result") { this.pending.delete(m.id); p.resolve({ streams: m.streams, dets: m.dets, timings: m.timings, device: m.device }); }
      else if (m.type === "error") { this.pending.delete(m.id); p.reject(new Error(m.message)); }
    };
    w.onerror = (e) => { for (const [, p] of this.pending) p.reject(new Error(e.message || "the background worker stopped")); this.pending.clear(); };
    return w;
  }

  private lowPower: boolean | null = null;

  analyse(rgba: Uint8ClampedArray, width: number, height: number, lang: string, easyocr: boolean, tesseract: Segment[],
    cb: Callbacks = {}, lowPower = false): Promise<AnalyseResult> {
    if (this.lowPower !== null && this.lowPower !== lowPower) this.cancel(); // the setting applies when models load
    this.lowPower = lowPower;
    const id = ++this.seq;
    const copy = rgba.slice(); // the page keeps its own pixels
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, cb });
      this.worker.postMessage({ type: "analyse", id, rgba: copy.buffer, width, height, lang, easyocr, tesseract, lowPower,
        base: new URL("./", document.baseURI).href }, [copy.buffer]);
    });
  }

  /** Stop whatever is running (the next request reloads the models from the browser's storage). */
  cancel(): void {
    this.worker.terminate();
    for (const [, p] of this.pending) p.reject(new Error("Stopped."));
    this.pending.clear();
    this.worker = this.spawn();
  }
}
