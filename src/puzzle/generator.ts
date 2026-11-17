// Puzzle generation: grid layout, randomised knob/socket edges, special
// ("whimsy") cat & star pieces, and automatic avoidance of face regions.
//
// Puzzle space = image pixel space (x right, y down). Callers convert to
// millimetres for laser export via a uniform scale.

import {
  Vec2,
  catmullRomOpen,
  enforceMinRadius,
  minCurveRadius,
  mulberry32,
  signedArea,
} from '../geom/curve';
import { SilhouetteKind, sampleSilhouette } from './silhouettes';

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface SpecialCell {
  row: number;
  col: number;
  kind: SilhouetteKind;
}

export interface PuzzlePiece {
  id: number;
  row: number;
  col: number;
  /** Closed outline in puzzle space, positive signed area, no duplicate joints. */
  outline: Vec2[];
  centroid: Vec2;
  special: SilhouetteKind | null;
}

export interface Puzzle {
  cols: number;
  rows: number;
  width: number;
  height: number;
  pieces: PuzzlePiece[];
  /** Every cut line exactly once — shared edges are stored a single time. */
  cutPaths: Vec2[][];
  specials: SpecialCell[];
  warnings: string[];
  /** Smallest curve radius across all cut paths (puzzle units). */
  minRadiusAchieved: number;
}

export interface GenerateOptions {
  width: number; // puzzle space width  (e.g. image px)
  height: number; // puzzle space height
  targetPieces: number;
  minRadius: number; // puzzle units — laser-safe minimum curve radius
  specialCount: number;
  seed: number;
  faceRects: Rect[];
}

const SILHOUETTE_SAMPLES = 160; // must be divisible by 8 (corner alignment)

export function gridDimensions(
  targetPieces: number,
  width: number,
  height: number,
): { cols: number; rows: number } {
  const aspect = width / height;
  let cols = Math.max(2, Math.round(Math.sqrt(targetPieces * aspect)));
  let rows = Math.max(2, Math.round(targetPieces / cols));
  return { cols, rows };
}

/** Classic jigsaw knob edge from a to b, relaxed to satisfy the laser minimum radius. */
function knobEdge(a: Vec2, b: Vec2, rng: () => number, minR: number): Vec2[] {
  const L = Math.hypot(b.x - a.x, b.y - a.y);
  const t = { x: (b.x - a.x) / L, y: (b.y - a.y) / L };
  const n = { x: -t.y, y: t.x };
  const tab = rng() < 0.5 ? 1 : -1;

  const neck = 0.1 + rng() * 0.06;
  const head = 0.22 + rng() * 0.08;
  const height = 0.17 + rng() * 0.06;
  const skew = (rng() - 0.5) * 0.1;

  const c = 0.5 + skew;
  const h = height * tab;
  // Control points in (along, across) fractions of L — a smooth knob silhouette.
  const ctrl: Array<[number, number]> = [
    [0, 0],
    [c - 0.5 * head - 0.14, 0],
    [c - 0.5 * neck - 0.04, 0.06 * h],
    [c - 0.5 * head, 0.42 * h],
    [c - 0.4 * head, 0.88 * h],
    [c, 1.0 * h],
    [c + 0.4 * head, 0.88 * h],
    [c + 0.5 * head, 0.42 * h],
    [c + 0.5 * neck + 0.04, 0.06 * h],
    [c + 0.5 * head + 0.14, 0],
    [1, 0],
  ];
  const pts2d = ctrl.map(([s, o]) => ({
    x: a.x + t.x * s * L + n.x * o * L,
    y: a.y + t.y * s * L + n.y * o * L,
  }));
  const sampled = catmullRomOpen(pts2d, 4);
  return enforceMinRadius(sampled, minR * 1.05);
}

/**
 * Warp a unit silhouette so the points at the four diagonal directions land
 * exactly on the cell corners, then extract the four side arcs.
 * Returns arcs in puzzle space, each running corner→corner.
 */
function specialCellEdges(
  kind: SilhouetteKind,
  cx: number,
  cy: number,
  cw: number,
  ch: number,
  minR: number,
): { top: Vec2[]; right: Vec2[]; bottom: Vec2[]; left: Vec2[] } {
  const cornerDist = Math.hypot(cw, ch) / 2;
  // Sample the silhouette with the radius constraint expressed in unit space
  // (unit space scales by ~cornerDist when mapped onto the cell).
  const unit = sampleSilhouette(kind, SILHOUETTE_SAMPLES, (minR / cornerDist) * 1.15);
  const N = unit.length;

  // Warp: scale rho(theta) so the four diagonal directions hit the corners.
  const cornerAngles = [(-3 * Math.PI) / 4, -Math.PI / 4, Math.PI / 4, (3 * Math.PI) / 4];
  const scales = cornerAngles.map((ca) => {
    // rho0 at the corner direction (sampled grid is aligned: corners fall on samples)
    const idx = Math.round(((ca + Math.PI) / (2 * Math.PI)) * N) % N;
    return cornerDist / Math.hypot(unit[idx].x, unit[idx].y);
  });
  const pts: Vec2[] = unit.map((p, i) => {
    const theta = (i / N) * 2 * Math.PI - Math.PI;
    // Piecewise-smooth interpolation of the corner scale factors.
    let k = scales[scales.length - 1];
    for (let j = 0; j < 4; j++) {
      const a0 = cornerAngles[j];
      const a1 = j < 3 ? cornerAngles[j + 1] : cornerAngles[0] + 2 * Math.PI;
      const th = theta < cornerAngles[0] ? theta + 2 * Math.PI : theta;
      if (th >= a0 && th <= a1) {
        const t = (th - a0) / (a1 - a0);
        const s0 = scales[j];
        const s1 = scales[(j + 1) % 4];
        const smooth = t * t * (3 - 2 * t);
        k = s0 + (s1 - s0) * smooth;
        break;
      }
    }
    return { x: cx + p.x * k, y: cy + p.y * k };
  });

  const arc = (i0: number, i1: number): Vec2[] => {
    const out: Vec2[] = [];
    for (let i = i0; i <= i1; i++) out.push(pts[i % N]);
    // The warp can re-sharpen features; relax each arc (corners stay fixed).
    return enforceMinRadius(out, minR * 1.05);
  };
  const eighth = N / 8;
  return {
    top: arc(eighth, 3 * eighth), // TL -> TR
    right: arc(3 * eighth, 5 * eighth), // TR -> BR
    bottom: arc(5 * eighth, 7 * eighth).reverse(), // stored BL -> BR
    left: arc(7 * eighth, 9 * eighth).reverse(), // stored TL -> BL
  };
}

function rectsIntersect(a: Rect, b: Rect): boolean {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}

function dedupePoints(pts: Vec2[]): Vec2[] {
  const out: Vec2[] = [];
  for (const p of pts) {
    const last = out[out.length - 1];
    if (!last || Math.hypot(p.x - last.x, p.y - last.y) > 1e-9) out.push(p);
  }
  // Drop closing duplicate.
  if (out.length > 1) {
    const f = out[0];
    const l = out[out.length - 1];
    if (Math.hypot(f.x - l.x, f.y - l.y) < 1e-9) out.pop();
  }
  return out;
}

function centroid(pts: Vec2[]): Vec2 {
  let cx = 0;
  let cy = 0;
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const q = pts[(i + 1) % pts.length];
    const w = p.x * q.y - q.x * p.y;
    a += w;
    cx += (p.x + q.x) * w;
    cy += (p.y + q.y) * w;
  }
  a /= 2;
  if (Math.abs(a) < 1e-12) return pts[0];
  return { x: cx / (6 * a), y: cy / (6 * a) };
}

export function generatePuzzle(opts: GenerateOptions): Puzzle {
  const rng = mulberry32(opts.seed);
  const warnings: string[] = [];
  const { cols, rows } = gridDimensions(opts.targetPieces, opts.width, opts.height);
  const cw = opts.width / cols;
  const ch = opts.height / rows;

  // ---- Choose special (whimsy) cells: interior ring, away from faces, non-adjacent.
  const inflateX = cw * 0.75;
  const inflateY = ch * 0.75;
  const faceRects = opts.faceRects.map((r) => ({
    x: r.x - inflateX,
    y: r.y - inflateY,
    w: r.w + 2 * inflateX,
    h: r.h + 2 * inflateY,
  }));
  const eligible: Array<{ row: number; col: number }> = [];
  for (let r = 1; r < rows - 1; r++) {
    for (let c = 1; c < cols - 1; c++) {
      const cell: Rect = { x: c * cw, y: r * ch, w: cw, h: ch };
      if (faceRects.some((f) => rectsIntersect(cell, f))) continue;
      eligible.push({ row: r, col: c });
    }
  }
  // Shuffle deterministically.
  for (let i = eligible.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [eligible[i], eligible[j]] = [eligible[j], eligible[i]];
  }
  const specials: SpecialCell[] = [];
  const occupied = new Set<string>();
  const key = (r: number, c: number) => `${r},${c}`;
  for (const cell of eligible) {
    if (specials.length >= opts.specialCount) break;
    let blocked = false;
    for (let dr = -1; dr <= 1 && !blocked; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        if (occupied.has(key(cell.row + dr, cell.col + dc))) {
          blocked = true;
          break;
        }
      }
    }
    if (blocked) continue;
    occupied.add(key(cell.row, cell.col));
    specials.push({
      row: cell.row,
      col: cell.col,
      kind: rng() < 0.5 ? 'cat' : 'star',
    });
  }
  if (specials.length < opts.specialCount) {
    warnings.push(
      `人脸区域或边界限制:只能放置 ${specials.length}/${opts.specialCount} 个异形块`,
    );
  }
  const specialAt = new Map<string, SpecialCell>();
  for (const s of specials) specialAt.set(key(s.row, s.col), s);

  // ---- Build all edges. vEdges[c][r]: vertical line x=c*cw between rows r and r+1.
  // hEdges[r][c]: horizontal line y=r*ch between cols c and c+1.
  const vEdges: (Vec2[] | null)[][] = [];
  for (let c = 0; c <= cols; c++) {
    vEdges.push(new Array<Vec2[] | null>(rows).fill(null));
  }
  const hEdges: (Vec2[] | null)[][] = [];
  for (let r = 0; r <= rows; r++) {
    hEdges.push(new Array<Vec2[] | null>(cols).fill(null));
  }

  // Special silhouettes first — their arcs become the shared edges.
  for (const s of specials) {
    const edges = specialCellEdges(
      s.kind,
      (s.col + 0.5) * cw,
      (s.row + 0.5) * ch,
      cw,
      ch,
      opts.minRadius,
    );
    hEdges[s.row][s.col] = edges.top;
    vEdges[s.col + 1][s.row] = edges.right;
    hEdges[s.row + 1][s.col] = edges.bottom;
    vEdges[s.col][s.row] = edges.left;
  }

  // Remaining edges: straight on the border, random knobs inside.
  for (let c = 0; c <= cols; c++) {
    for (let r = 0; r < rows; r++) {
      if (vEdges[c][r]) continue;
      const a = { x: c * cw, y: r * ch };
      const b = { x: c * cw, y: (r + 1) * ch };
      vEdges[c][r] =
        c === 0 || c === cols ? [a, b] : knobEdge(a, b, rng, opts.minRadius);
    }
  }
  for (let r = 0; r <= rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (hEdges[r][c]) continue;
      const a = { x: c * cw, y: r * ch };
      const b = { x: (c + 1) * cw, y: r * ch };
      hEdges[r][c] =
        r === 0 || r === rows ? [a, b] : knobEdge(a, b, rng, opts.minRadius);
    }
  }

  // ---- Assemble piece outlines (TL -> TR -> BR -> BL).
  const pieces: PuzzlePiece[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const top = hEdges[r][c]!;
      const right = vEdges[c + 1][r]!;
      const bottom = hEdges[r + 1][c]!.slice().reverse();
      const left = vEdges[c][r]!.slice().reverse();
      let outline = dedupePoints([
        ...top.slice(0, -1),
        ...right.slice(0, -1),
        ...bottom.slice(0, -1),
        ...left.slice(0, -1),
      ]);
      if (signedArea(outline) < 0) outline = outline.reverse();
      const sp = specialAt.get(key(r, c));
      pieces.push({
        id: r * cols + c,
        row: r,
        col: c,
        outline,
        centroid: centroid(outline),
        special: sp ? sp.kind : null,
      });
    }
  }

  // ---- Unique cut paths: every edge exactly once.
  const cutPaths: Vec2[][] = [];
  for (let c = 0; c <= cols; c++)
    for (let r = 0; r < rows; r++) cutPaths.push(vEdges[c][r]!);
  for (let r = 0; r <= rows; r++)
    for (let c = 0; c < cols; c++) cutPaths.push(hEdges[r][c]!);

  let minR = Infinity;
  for (const p of cutPaths) {
    if (p.length < 3) continue;
    const r = minCurveRadius(p);
    if (r < minR) minR = r;
  }
  if (minR < opts.minRadius * 0.95) {
    warnings.push(
      `最小圆角 ${minR.toFixed(2)} 低于目标 ${opts.minRadius.toFixed(2)}(单位:拼图坐标),可增大单块尺寸`,
    );
  }

  return {
    cols,
    rows,
    width: opts.width,
    height: opts.height,
    pieces,
    cutPaths,
    specials,
    warnings,
    minRadiusAchieved: minR,
  };
}
