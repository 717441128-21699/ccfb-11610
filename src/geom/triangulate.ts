// Ear-clipping triangulation for simple polygons (no holes).

import { Vec2, signedArea } from '../geom/curve';

function cross(o: Vec2, a: Vec2, b: Vec2): number {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

function pointInTriangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2): boolean {
  const d1 = cross(a, b, p);
  const d2 = cross(b, c, p);
  const d3 = cross(c, a, p);
  const hasNeg = d1 < 0 || d2 < 0 || d3 < 0;
  const hasPos = d1 > 0 || d2 > 0 || d3 > 0;
  return !(hasNeg && hasPos);
}

/** Returns triangle vertex indices into `poly`. Falls back to a fan if clipping stalls. */
export function triangulate(poly: Vec2[]): number[] {
  const n = poly.length;
  if (n < 3) return [];
  if (n === 3) return [0, 1, 2];

  const ccw = signedArea(poly) > 0;
  const idx: number[] = [];
  for (let i = 0; i < n; i++) idx.push(i);
  const out: number[] = [];

  let guard = 0;
  const maxIter = n * n;
  while (idx.length > 3 && guard++ < maxIter) {
    let clipped = false;
    for (let i = 0; i < idx.length; i++) {
      const i0 = idx[(i - 1 + idx.length) % idx.length];
      const i1 = idx[i];
      const i2 = idx[(i + 1) % idx.length];
      const a = poly[i0];
      const b = poly[i1];
      const c = poly[i2];
      const turn = cross(a, b, c);
      if (ccw ? turn <= 1e-12 : turn >= -1e-12) continue; // reflex or degenerate
      let ear = true;
      for (const j of idx) {
        if (j === i0 || j === i1 || j === i2) continue;
        if (pointInTriangle(poly[j], a, b, c)) {
          ear = false;
          break;
        }
      }
      if (!ear) continue;
      out.push(i0, i1, i2);
      idx.splice(i, 1);
      clipped = true;
      break;
    }
    if (!clipped) break; // numerically stuck — fall through to fan
  }
  if (idx.length === 3) {
    out.push(idx[0], idx[1], idx[2]);
  }
  // If we failed to decompose fully, top up with a fan (keeps mesh closed).
  const tris = out.length / 3;
  if (tris !== n - 2) {
    const fan: number[] = [];
    for (let i = 1; i < n - 1; i++) fan.push(0, i, i + 1);
    return fan;
  }
  return out;
}
