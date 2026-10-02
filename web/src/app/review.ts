/** The review state: what was found on each page, as items a person can switch off, add to and undo.
 *  With every item on, finalRegions() equals parda's Redactor.analyse (per OCR source: spans -> boxes -> gap filling;
 *  then visual regions). DOM-free, so it is tested in Node. */
import { fillGaps, type Region, spanBoxes } from "../core/boxes.js";
import { asciiDigits } from "../core/pycompat.js";
import type { Span } from "../core/rules.js";
import type { BBox, Segment } from "../core/spans.js";
import type { Detection } from "../vision/yolo.js";

/** One OCR stream of a page: its joined text, segments in reading order, and the PII spans found in the text. */
export interface Stream { source: string; text: string; ordered: Segment[]; offs: Array<[number, number]>; spans: Span[] }

export type Group = "ids" | "people" | "contact" | "visual" | "manual";
export const GROUP_OF: Record<string, Group> = {
  AADHAAR: "ids", PAN: "ids", BANK_ACCOUNT: "ids", IFSC: "ids", UPI_ID: "ids", GSTIN: "ids", VOTER_ID: "ids",
  PASSPORT: "ids", VEHICLE_REG: "ids", UAN: "ids", ABHA: "ids", EMPLOYEE_ID: "ids", MRN: "ids",
  PERSON_NAME: "people", DOB: "people",
  PHONE: "contact", EMAIL: "contact", ADDRESS: "contact",
  FACE: "visual", SIGNATURE: "visual", QR_CODE: "visual", STAMP: "visual",
};
export const groupOf = (label: string): Group => GROUP_OF[label] ?? "manual";

/** Where an item came from in one OCR stream (or a visual / hand-drawn box). */
export interface Part { source: string; order: number; start?: number; end?: number; boxes: BBox[]; score: number; added?: boolean }

export interface Item {
  id: number;
  label: string;
  kind: "text" | "visual";
  text?: string;   // what OCR read (stays on this device; never in the audit)
  parts: Part[];
  on: boolean;
  added?: boolean; // drawn by hand or via "hide everywhere"
}

export interface PageState { width: number; height: number; streams: Stream[]; items: Item[] }

const norm = (s: string) => asciiDigits(s).toLowerCase().replace(/[\s\-.,:;]+/g, "");

/** Spans of every stream + YOLO detections -> items. The same text and label found by two OCR streams is ONE item. */
export function buildItems(streams: Stream[], dets: Detection[], visualConf = 0.25): Item[] {
  const items: Item[] = [];
  const byKey = new Map<string, Item>();
  let id = 1;
  streams.forEach((st) => {
    const seen = new Map<string, number>(); // the k-th occurrence of a text in one stream matches the k-th in another
    st.spans.forEach((sp, order) => {
      const text = st.text.slice(sp.start, sp.end);
      const part: Part = { source: st.source, order, start: sp.start, end: sp.end, score: sp.score,
        boxes: spanBoxes(sp.start, sp.end, st.ordered, st.offs) };
      const base = `${sp.label}|${norm(text)}`;
      const k = seen.get(base) ?? 0;
      seen.set(base, k + 1);
      const key = `${base}|${k}`;
      const same = byKey.get(key);
      if (same && !same.parts.some((p) => p.source === st.source)) same.parts.push(part);
      else {
        const it: Item = { id: id++, label: sp.label, kind: "text", text, parts: [part], on: true };
        items.push(it);
        byKey.set(key, it);
      }
    });
  });
  dets.filter((d) => d.conf >= visualConf).forEach((d, order) => {
    items.push({ id: id++, label: d.label, kind: "visual", on: true,
      parts: [{ source: "yolo", order, boxes: [[...d.xyxy] as BBox], score: d.conf }] });
  });
  return items;
}

/** Aadhaar shown masked (as UIDAI's "masked Aadhaar"): the bar stops before the last 4 digits. */
export function maskedEnd(text: string, start: number, end: number): number {
  const s = asciiDigits(text.slice(start, end));
  let seen = 0;
  for (let i = s.length - 1; i >= 0; i--) {
    if (/[0-9]/.test(s[i]) && ++seen === 4) {
      let j = i; // also leave the space before the last group visible
      while (j > 0 && !/[0-9]/.test(s[j - 1])) j--;
      return start + j;
    }
  }
  return end;
}

/** The regions to black out: per OCR source, the boxes of every item that is on (in the original span order),
 *  gap-filled; then visual and hand-drawn boxes. */
export function finalRegions(page: PageState, opts: { maskAadhaar?: boolean } = {}): Region[] {
  const out: Region[] = [];
  for (const st of page.streams) {
    const parts = page.items.filter((it) => it.on && it.kind === "text")
      .flatMap((it) => it.parts.filter((p) => p.source === st.source).map((p) => ({ it, p })))
      .sort((a, b) => (a.p.added ? 1 : 0) - (b.p.added ? 1 : 0) || a.p.order - b.p.order);
    const regs: Region[] = [];
    for (const { it, p } of parts) {
      let boxes = p.boxes;
      if (opts.maskAadhaar && it.label === "AADHAAR" && p.start !== undefined && p.end !== undefined) {
        boxes = spanBoxes(p.start, maskedEnd(st.text, p.start, p.end), st.ordered, st.offs);
      }
      regs.push(...boxes.map((bbox) => ({ kind: "text" as const, label: it.label, bbox, score: p.score, source: st.source })));
    }
    out.push(...fillGaps(regs));
  }
  for (const it of page.items.filter((it) => it.on && (it.kind === "visual" || it.parts.some((p) => !page.streams.some((s) => s.source === p.source))))) {
    for (const p of it.parts.filter((p) => !page.streams.some((s) => s.source === p.source))) {
      out.push(...p.boxes.map((bbox) => ({ kind: it.kind, label: it.label, bbox, score: p.score, source: p.source })));
    }
  }
  return out;
}

/** Every other place the same text appears in any stream (for "hide everywhere"). */
export function occurrences(page: PageState, item: Item): Part[] {
  if (!item.text || norm(item.text).length < 3) return [];
  const want = item.text.trim();
  const found: Part[] = [];
  for (const st of page.streams) {
    let from = 0;
    for (;;) {
      const at = st.text.indexOf(want, from);
      if (at < 0) break;
      from = at + want.length;
      const taken = page.items.some((it) => it.parts.some((p) => p.source === st.source && p.start !== undefined
        && p.start < at + want.length && at < p.end!));
      if (!taken) found.push({ source: st.source, order: 1e6 + at, start: at, end: at + want.length, score: 1,
        boxes: spanBoxes(at, at + want.length, st.ordered, st.offs) });
    }
  }
  return found;
}

/** Edits with undo: every change is a new snapshot of the items. */
export class Review {
  private past: Item[][] = [];
  private future: Item[][] = [];
  private nextId: number;

  constructor(public page: PageState) {
    this.nextId = Math.max(0, ...page.items.map((i) => i.id)) + 1;
  }

  private commit(next: Item[]) {
    this.past.push(this.page.items);
    this.future = [];
    this.page = { ...this.page, items: next };
  }

  toggle(id: number) { this.commit(this.page.items.map((i) => (i.id === id ? { ...i, on: !i.on } : i))); }
  setGroup(group: Group, on: boolean) { this.commit(this.page.items.map((i) => (groupOf(i.label) === group ? { ...i, on } : i))); }

  /** A box drawn by hand (page pixels). */
  addBox(bbox: BBox): Item {
    const it: Item = { id: this.nextId++, label: "MANUAL", kind: "visual", on: true, added: true,
      parts: [{ source: "manual", order: this.nextId, boxes: [bbox], score: 1 }] };
    this.commit([...this.page.items, it]);
    return it;
  }

  /** Hide every other occurrence of this item's text. Returns how many were added. */
  hideEverywhere(id: number): number {
    const item = this.page.items.find((i) => i.id === id);
    const parts = item ? occurrences(this.page, item) : [];
    if (!item || !parts.length) return 0;
    const added = parts.map((p) => ({ id: this.nextId++, label: item.label, kind: "text" as const, text: item.text, on: true,
      added: true, parts: [{ ...p, added: true }] }));
    this.commit([...this.page.items, ...added]);
    return added.length;
  }

  canUndo() { return this.past.length > 0; }
  canRedo() { return this.future.length > 0; }
  undo() { if (this.past.length) { this.future.push(this.page.items); this.page = { ...this.page, items: this.past.pop()! }; } }
  redo() { if (this.future.length) { this.past.push(this.page.items); this.page = { ...this.page, items: this.future.pop()! }; } }
}
