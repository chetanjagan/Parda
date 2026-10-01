/** YOLO in the browser: page pixels -> letterbox -> ONNX -> boxes in page pixels. The runner is injected. */
import type { Region } from "../core/boxes.js";
import { letterbox } from "./letterbox.js";
import { type DecodeOptions, decodeYolo, type Detection } from "./yolo.js";

export type ImageRunner = (input: { data: Float32Array; dims: number[] }) => Promise<{ data: Float32Array | number[]; dims: number[] }>;

export const VISUAL_NAMES: Record<number, string> = { 0: "FACE", 1: "SIGNATURE", 2: "QR_CODE", 3: "STAMP" };

export class YoloDetector {
  constructor(private readonly run: ImageRunner, private readonly names: Record<number, string> = VISUAL_NAMES,
    private readonly size = 1024, private readonly opt: DecodeOptions = {}) {}

  async detect(pixels: Uint8Array | Uint8ClampedArray, w: number, h: number, channels = 4): Promise<Detection[]> {
    const lb = letterbox(pixels, w, h, channels, this.size);
    const out = await this.run({ data: lb.tensor, dims: [1, 3, this.size, this.size] });
    return decodeYolo(out.data, out.dims, h, w, this.names, this.size, this.opt);
  }
}

/** Detections -> regions to redact (kept above the pipeline's visual threshold, like redactor.visual_regions). */
export function visualRegions(dets: Detection[], conf = 0.25): Region[] {
  return dets.filter((d) => d.conf >= conf).map((d) => ({
    kind: "visual" as const, label: d.label, bbox: [...d.xyxy] as [number, number, number, number], score: d.conf, source: "yolo",
  }));
}
