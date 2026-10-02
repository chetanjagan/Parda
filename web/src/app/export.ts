/** Export: pages flattened to images with the bars burnt in (no hidden text survives), as PNG or one PDF,
 *  plus an audit file of what was hidden where — never the text itself. */
import { auditEntry } from "../core/audit.js";
import type { Region } from "../core/boxes.js";
import { lib, LIBS } from "./libs.js";
import { blackouts } from "./pipeline.js";

export function redactedCanvas(src: HTMLCanvasElement, regions: Region[]): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = src.width;
  c.height = src.height;
  const ctx = c.getContext("2d")!;
  ctx.drawImage(src, 0, 0);
  ctx.fillStyle = "#000";
  for (const [x0, y0, x1, y1] of blackouts(regions, c.width, c.height)) ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  return c;
}

const blobOf = (c: HTMLCanvasElement, type = "image/png", q?: number) =>
  new Promise<Blob>((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error("could not encode the page"))), type, q));

export function save(blob: Blob, name: string): void {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}

export async function pngOf(c: HTMLCanvasElement): Promise<Blob> { return blobOf(c); }

/** One PDF, one flattened image per page (JPEG at high quality keeps the file small). */
export async function pdfOf(pages: HTMLCanvasElement[]): Promise<Blob> {
  const { PDFDocument } = await lib(LIBS.pdfLib);
  const doc = await PDFDocument.create();
  for (const c of pages) {
    const img = await doc.embedJpg(new Uint8Array(await (await blobOf(c, "image/jpeg", 0.92)).arrayBuffer()));
    const w = c.width * 0.75, h = c.height * 0.75; // 96 dpi pixels -> PDF points
    doc.addPage([w, h]).drawImage(img, { x: 0, y: 0, width: w, height: h });
  }
  return new Blob([await doc.save()], { type: "application/pdf" });
}

export function auditOf(file: string, settings: Record<string, unknown>, pages: Region[][]): Blob {
  const audit = { tool: "Parda (browser)", file, created: new Date().toISOString(), settings,
    pages: pages.map((regions, i) => ({ page: i + 1, hidden: regions.length, regions: auditEntry(regions) })) };
  return new Blob([JSON.stringify(audit, null, 2)], { type: "application/json" });
}
