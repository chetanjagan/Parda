/** Files -> page images (RGBA pixels + a canvas). PDFs are rendered with pdf.js; photos keep their EXIF rotation. */
export interface PageImage { canvas: HTMLCanvasElement; rgba: Uint8ClampedArray; width: number; height: number; name: string }

const PDF_SCALE = 2; // ~144 dpi for a typical A4 PDF
const MAX_SIDE = 2560; // EasyOCR's canvas; larger photos are scaled down first

function fromSource(src: CanvasImageSource, w: number, h: number, name: string): PageImage {
  const s = Math.min(1, MAX_SIDE / Math.max(w, h));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(w * s);
  canvas.height = Math.round(h * s);
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(src, 0, 0, canvas.width, canvas.height);
  const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  return { canvas, rgba, width: canvas.width, height: canvas.height, name };
}

export async function loadFile(file: File): Promise<PageImage[]> {
  if (file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf")) {
    const name = "pdfjs-dist";
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const pdfjs: any = await import(/* @vite-ignore */ name);
    pdfjs.GlobalWorkerOptions.workerSrc = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.worker.min.mjs";
    const doc = await pdfjs.getDocument({ data: new Uint8Array(await file.arrayBuffer()) }).promise;
    const pages: PageImage[] = [];
    for (let i = 1; i <= doc.numPages; i++) {
      const page = await doc.getPage(i);
      const vp = page.getViewport({ scale: PDF_SCALE });
      const c = document.createElement("canvas");
      c.width = Math.round(vp.width);
      c.height = Math.round(vp.height);
      await page.render({ canvasContext: c.getContext("2d")!, viewport: vp }).promise;
      pages.push(fromSource(c, c.width, c.height, `${file.name} — page ${i}`));
    }
    return pages;
  }
  const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
  return [fromSource(bmp, bmp.width, bmp.height, file.name)];
}
