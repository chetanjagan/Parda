/** The audit log entry: what was redacted, where, how confident, found by what. Never the text itself. */
import type { Region } from "./boxes.js";
import { pyRound } from "./pycompat.js";

export function auditEntry(regions: Region[]) {
  return regions.map((r) => ({
    kind: r.kind, label: r.label, bbox: r.bbox.map((v) => pyRound(Number(v), 1)),
    score: pyRound(Number(r.score), 3), source: r.source,
  }));
}
