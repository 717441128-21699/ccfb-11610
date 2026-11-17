// 2D vector / polyline / curve utilities shared by puzzle generation and laser paths.

export interface Vec2 {
  x: number;
  y: number;
}

export const v = (x: number, y: number): Vec2 => ({ x, y });
export const add = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x + b.x, y: a.y + b.y });
export const sub = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x - b.x, y: a.y - b.y });
export const scale = (a: Vec2, s: number): Vec2 => ({ x: a.x * s, y: a.y * s });
export const lerp = (a: Vec2, b: Vec2, t: number): Vec2 => ({
  x: a.x + (b.x - a.x) * t,
  y: a.y + (b.y - a.y) * t,
});
export const len = (a: Vec2): number => Math.hypot(a.x, a.y);
export const dist = (a: Vec2, b: Vec2): number => Math.hypot(a.x - b.x, a.y - b.y);
export const norm = (a: Vec2): Vec2 => {
  const l = len(a) || 1;
  return { x: a.x / l, y: a.y / l };
};
/** Left perpendicular (in y-down screen space this points "outward" for CW wound polys). */
export const perp = (a: Vec2): Vec2 => ({ x: -a.y, y: a.x });

export function polylineLength(pts: Vec2[]): number {
  let s = 0;
  for (let i = 1; i < pts.length; i++) s += dist(pts[i - 1], pts[i]);
  return s;
}

export function signedArea(pts: Vec2[]): number {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const q = pts[(i + 1) % pts.length];
    a += p.x * q.y - q.x * p.y;
  }
  return a / 2;
}

/** Return a copy of the polygon whose signed area has the requested sign. */
export function ensureAreaSign(pts: Vec2[], sign: 1 | -1): Vec2[] {
  const a = signedArea(pts);
  if ((a >= 0 ? 1 : -1) === sign) return pts.slice();
  return pts.slice().reverse();
}

/**
 * Sample an open Catmull-Rom spline through `pts` (endpoints duplicated).
 * `samplesPerSeg` points are emitted per interior segment, endpoints included.
 */
export function catmullRomOpen(pts: Vec2[], samplesPerSeg: number): Vec2[] {
  if (pts.length < 2) return pts.slice();
  const P = [pts[0], ...pts, pts[pts.length - 1]];
  const out: Vec2[] = [];
  for (let i = 0; i < P.length - 3; i++) {
    const p0 = P[i], p1 = P[i + 1], p2 = P[i + 2], p3 = P[i + 3];
    for (let j = 0; j < samplesPerSeg; j++) {
      const t = j / samplesPerSeg;
      const t2 = t * t;
      const t3 = t2 * t;
      out.push({
        x:
          0.5 *
          (2 * p1.x +
            (-p0.x + p2.x) * t +
            (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 +
            (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3),
        y:
          0.5 *
          (2 * p1.y +
            (-p0.y + p2.y) * t +
            (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 +
            (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3),
      });
    }
  }
  out.push(pts[pts.length - 1]);
  return out;
}

/** Circumradius of the triangle through three consecutive samples. Infinity when collinear. */
export function circumRadius(a: Vec2, b: Vec2, c: Vec2): number {
  const ab = dist(a, b);
  const bc = dist(b, c);
  const ca = dist(c, a);
  const area2 = Math.abs(
    (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y),
  );
  if (area2 < 1e-12) return Infinity;
  return (ab * bc * ca) / (2 * area2);
}

/**
 * Smallest discrete curvature radius along a polyline. This is the number the
 * laser cares about: below it the beam dwells too long and scorches the wood.
 */
export function minCurveRadius(pts: Vec2[]): number {
  let min = Infinity;
  for (let i = 1; i < pts.length - 1; i++) {
    const r = circumRadius(pts[i - 1], pts[i], pts[i + 1]);
    if (r < min) min = r;
  }
  return min;
}

/**
 * Iteratively relax a polyline (endpoints fixed) until every discrete curve
 * radius is at least `minR`, or the iteration budget runs out. Only points
 * near a violation are smoothed, so the overall shape is preserved.
 */
export function enforceMinRadius(pts: Vec2[], minR: number, maxIter = 600): Vec2[] {
  const n = pts.length;
  if (n < 3) return pts.slice();
  let cur = pts.map((p) => ({ ...p }));
  for (let iter = 0; iter < maxIter; iter++) {
    const violate = new Array<boolean>(n).fill(false);
    let worst = Infinity;
    for (let i = 1; i < n - 1; i++) {
      const r = circumRadius(cur[i - 1], cur[i], cur[i + 1]);
      if (r < minR) {
        violate[i] = true;
        if (i > 1) violate[i - 1] = true;
        if (i < n - 2) violate[i + 1] = true;
      }
      if (r < worst) worst = r;
    }
    if (worst >= minR) return cur;
    const next = cur.map((p) => ({ ...p }));
    for (let i = 1; i < n - 1; i++) {
      if (!violate[i]) continue;
      next[i] = {
        x: cur[i].x * 0.5 + (cur[i - 1].x + cur[i + 1].x) * 0.25,
        y: cur[i].y * 0.5 + (cur[i - 1].y + cur[i + 1].y) * 0.25,
      };
    }
    cur = next;
  }
  return cur;
}

/**
 * Round every corner of a polygon with a circular arc of radius `radius`
 * (clamped per-corner so fillets never overlap). Returns a dense polyline.
 * Used to give star tips / cat ears a laser-safe minimum radius.
 */
export function roundedPolygon(verts: Vec2[], radius: number, arcSamples = 6): Vec2[] {
  const n = verts.length;
  const out: Vec2[] = [];
  for (let i = 0; i < n; i++) {
    const prev = verts[(i - 1 + n) % n];
    const cur = verts[i];
    const next = verts[(i + 1) % n];
    const d1 = norm(sub(prev, cur));
    const d2 = norm(sub(next, cur));
    const l1 = dist(prev, cur);
    const l2 = dist(next, cur);
    const cosA = Math.max(-1, Math.min(1, d1.x * d2.x + d1.y * d2.y));
    const half = Math.acos(cosA) / 2;
    const tanHalf = Math.tan(half);
    // Distance from the corner to each tangent point, limited by edge lengths.
    const t = Math.min(radius / Math.max(tanHalf, 1e-6), l1 * 0.45, l2 * 0.45);
    const p1 = add(cur, scale(d1, t));
    const p2 = add(cur, scale(d2, t));
    const r = t * tanHalf;
    if (r < 1e-9 || !isFinite(r)) {
      out.push(cur);
      continue;
    }
    // Arc centre: along the bisector, at distance r/sin(half) from the corner.
    const bis = norm(add(d1, d2));
    const centre = add(cur, scale(bis, r / Math.max(Math.sin(half), 1e-6)));
    const a1 = Math.atan2(p1.y - centre.y, p1.x - centre.x);
    let a2 = Math.atan2(p2.y - centre.y, p2.x - centre.x);
    // Choose the sweep direction that takes the short way around.
    let sweep = a2 - a1;
    while (sweep > Math.PI) sweep -= 2 * Math.PI;
    while (sweep < -Math.PI) sweep += 2 * Math.PI;
    a2 = a1 + sweep;
    out.push(p1);
    for (let k = 1; k < arcSamples; k++) {
      const a = a1 + (a2 - a1) * (k / arcSamples);
      out.push({ x: centre.x + Math.cos(a) * r, y: centre.y + Math.sin(a) * r });
    }
    out.push(p2);
  }
  return out;
}

/** Deterministic PRNG so a seed reproduces an identical puzzle. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
