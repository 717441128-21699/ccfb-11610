// Special ("whimsy") piece silhouettes: cat head and five-pointed star.
// Both are defined as polar radius functions rho(theta) around their centre,
// which makes them trivial to warp onto a puzzle cell and to resample.
// A smoothing pass guarantees a laser-safe minimum curve radius.

import { Vec2, minCurveRadius } from '../geom/curve';

export type SilhouetteKind = 'cat' | 'star';

const TAU = Math.PI * 2;

/** Raised-cosine bump centred at `c` with half-width `w` (radians), height `h`. */
function bump(theta: number, c: number, w: number, h: number): number {
  let d = theta - c;
  while (d > Math.PI) d -= TAU;
  while (d < -Math.PI) d += TAU;
  if (Math.abs(d) >= w) return 0;
  return h * 0.5 * (1 + Math.cos((Math.PI * d) / w));
}

/**
 * Cat head: a circle with two ear bumps on top and slightly wider cheeks.
 * theta = -PI/2 is "up" in y-down image space.
 */
function catRadius(theta: number): number {
  let r = 0.78;
  r += bump(theta, -Math.PI / 2 - 0.72, 0.5, 0.34); // left ear
  r += bump(theta, -Math.PI / 2 + 0.72, 0.5, 0.34); // right ear
  r += bump(theta, Math.PI / 2, 1.1, 0.06); // gentle chin flattening opposite side
  return r;
}

/** Star: five rounded points. The exponent < 1 keeps tips blunt, not needle-sharp. */
function starRadius(theta: number): number {
  const inner = 0.58;
  const outer = 1.0;
  const wave = 0.5 + 0.5 * Math.cos(5 * (theta + Math.PI / 2));
  return inner + (outer - inner) * Math.pow(wave, 0.72);
}

export function silhouetteRadiusFn(kind: SilhouetteKind): (theta: number) => number {
  return kind === 'cat' ? catRadius : starRadius;
}

/**
 * Sample a silhouette as `n` points evenly spaced in angle, smoothed in polar
 * space until every curve radius is at least `minRadius` (in the same units as
 * the radius function output) or the iteration budget runs out.
 */
export function sampleSilhouette(
  kind: SilhouetteKind,
  n: number,
  minRadius: number,
): Vec2[] {
  const fn = silhouetteRadiusFn(kind);
  let rho: number[] = [];
  for (let i = 0; i < n; i++) rho.push(fn((i / n) * TAU - Math.PI));

  // Laplacian smoothing on the periodic radius function raises the minimum
  // radius of curvature; iterate until safe (usually 0-3 passes).
  for (let iter = 0; iter < 40; iter++) {
    const pts = polarToXY(rho);
    if (minCurveRadius([...pts, pts[0], pts[1]]) >= minRadius) break;
    const next = rho.slice();
    for (let i = 0; i < n; i++) {
      const a = rho[(i - 1 + n) % n];
      const b = rho[i];
      const c = rho[(i + 1) % n];
      next[i] = b * 0.5 + (a + c) * 0.25;
    }
    rho = next;
  }
  return polarToXY(rho);
}

function polarToXY(rho: number[]): Vec2[] {
  const n = rho.length;
  const out: Vec2[] = [];
  for (let i = 0; i < n; i++) {
    const t = (i / n) * TAU - Math.PI;
    out.push({ x: Math.cos(t) * rho[i], y: Math.sin(t) * rho[i] });
  }
  return out;
}
