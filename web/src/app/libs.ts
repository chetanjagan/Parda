/** Third-party libraries, by full CDN address (import maps do not apply inside workers). Pinned versions. */
export const LIBS = {
  ort: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.mjs",
  ortWasm: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/",
  transformers: "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.0/+esm",
  tesseract: "https://cdn.jsdelivr.net/npm/tesseract.js@6/dist/tesseract.esm.min.js",
  pdfjs: "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.min.mjs",
  pdfjsWorker: "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.worker.min.mjs",
  pdfLib: "https://cdn.jsdelivr.net/npm/pdf-lib@1.17.1/+esm",
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function lib(url: string): Promise<any> {
  const m = await import(/* @vite-ignore */ url);
  return m.default && Object.keys(m).length === 1 ? m.default : m;
}
