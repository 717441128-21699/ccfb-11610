import { describe, it, expect } from 'vitest';
import { generatePuzzle, gridDimensions, Rect } from '../src/puzzle/generator';
import { minCurveRadius, signedArea, polylineLength } from '../src/geom/curve';
import { triangulate } from '../src/geom/triangulate';
import { orderCutPaths } from '../src/laser/optimize';
import { toSVG, toDXF } from '../src/laser/export';
import { sampleSilhouette } from '../src/puzzle/silhouettes';

const BASE = {
  width: 1600,
  height: 1200,
  targetPieces: 500,
  minRadius: 3, // puzzle units (px here)
  specialCount: 5,
  seed: 42,
  faceRects: [] as Rect[],
};

describe('puzzle generation', () => {
  const puzzle = generatePuzzle(BASE);

  it('produces the requested grid', () => {
    const { cols, rows } = gridDimensions(500, 1600, 1200);
    expect(cols * rows).toBeGreaterThanOrEqual(480);
    expect(puzzle.pieces.length).toBe(puzzle.cols * puzzle.rows);
  });

  it('every piece outline is closed, simple-sized and positively wound', () => {
    for (const p of puzzle.pieces) {
      expect(p.outline.length).toBeGreaterThan(10);
      expect(signedArea(p.outline)).toBeGreaterThan(0);
    }
  });

  it('stores each shared edge exactly once', () => {
    // vertical edges: (cols+1)*rows, horizontal edges: (rows+1)*cols
    const expected = (puzzle.cols + 1) * puzzle.rows + (puzzle.rows + 1) * puzzle.cols;
    expect(puzzle.cutPaths.length).toBe(expected);
  });

  it('respects the laser minimum curve radius', () => {
    expect(puzzle.minRadiusAchieved).toBeGreaterThanOrEqual(BASE.minRadius * 0.9);
    for (const path of puzzle.cutPaths) {
      if (path.length < 3) continue;
      expect(minCurveRadius(path)).toBeGreaterThanOrEqual(BASE.minRadius * 0.9);
    }
  });

  it('places the requested number of special pieces', () => {
    expect(puzzle.specials.length).toBe(5);
    const kinds = puzzle.specials.map((s) => s.kind);
    expect(kinds.every((k) => k === 'cat' || k === 'star')).toBe(true);
  });

  it('keeps special pieces off faces and apart from each other', () => {
    const face: Rect = { x: 600, y: 400, w: 300, h: 300 };
    const p2 = generatePuzzle({ ...BASE, faceRects: [face], seed: 7 });
    const cw = p2.width / p2.cols;
    const ch = p2.height / p2.rows;
    for (const s of p2.specials) {
      const cell = { x: s.col * cw, y: s.row * ch, w: cw, h: ch };
      const inflated = {
        x: face.x - cw * 0.75,
        y: face.y - ch * 0.75,
        w: face.w + cw * 1.5,
        h: face.h + ch * 1.5,
      };
      const overlap =
        cell.x < inflated.x + inflated.w &&
        cell.x + cell.w > inflated.x &&
        cell.y < inflated.y + inflated.h &&
        cell.y + cell.h > inflated.y;
      expect(overlap).toBe(false);
      // interior ring only
      expect(s.row).toBeGreaterThan(0);
      expect(s.row).toBeLessThan(p2.rows - 1);
      expect(s.col).toBeGreaterThan(0);
      expect(s.col).toBeLessThan(p2.cols - 1);
    }
    for (let i = 0; i < p2.specials.length; i++) {
      for (let j = i + 1; j < p2.specials.length; j++) {
        const a = p2.specials[i];
        const b = p2.specials[j];
        expect(Math.max(Math.abs(a.row - b.row), Math.abs(a.col - b.col))).toBeGreaterThan(1);
      }
    }
  });

  it('is deterministic for a fixed seed', () => {
    const a = generatePuzzle(BASE);
    const b = generatePuzzle(BASE);
    expect(a.pieces[123].outline).toEqual(b.pieces[123].outline);
    expect(a.specials).toEqual(b.specials);
  });
});

describe('triangulation', () => {
  it('preserves area', () => {
    const puzzle = generatePuzzle({ ...BASE, targetPieces: 60, specialCount: 2 });
    for (const p of puzzle.pieces) {
      const tris = triangulate(p.outline);
      expect(tris.length).toBe((p.outline.length - 2) * 3);
      let area = 0;
      for (let i = 0; i < tris.length; i += 3) {
        const a = p.outline[tris[i]];
        const b = p.outline[tris[i + 1]];
        const c = p.outline[tris[i + 2]];
        area += Math.abs(
          (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y),
        ) / 2;
      }
      expect(area).toBeCloseTo(Math.abs(signedArea(p.outline)), 0);
    }
  });
});

describe('silhouettes', () => {
  it('stay above the requested minimum radius', () => {
    for (const kind of ['cat', 'star'] as const) {
      const pts = sampleSilhouette(kind, 160, 0.05);
      const closed = [...pts, pts[0], pts[1]];
      expect(minCurveRadius(closed)).toBeGreaterThanOrEqual(0.045);
    }
  });
});

describe('laser path optimisation', () => {
  it('never does worse than the naive order and covers every path once', () => {
    const puzzle = generatePuzzle({ ...BASE, targetPieces: 200 });
    const paths = puzzle.cutPaths;
    const result = orderCutPaths(paths, 2000);
    expect(result.order.length).toBe(paths.length);
    expect(new Set(result.order).size).toBe(paths.length);
    expect(result.travel).toBeLessThanOrEqual(result.naiveTravel + 1e-6);
    expect(result.cutLength).toBeCloseTo(
      paths.reduce((s, p) => s + polylineLength(p), 0),
      6,
    );
  });

  it('generates + optimises 500 pieces within 10 seconds', () => {
    const t0 = performance.now();
    const puzzle = generatePuzzle(BASE);
    const result = orderCutPaths(puzzle.cutPaths, 8000);
    const elapsed = performance.now() - t0;
    expect(puzzle.pieces.length).toBeGreaterThanOrEqual(480);
    expect(elapsed).toBeLessThan(10_000);
    expect(result.travel).toBeLessThan(result.naiveTravel);
  }, 20_000);
});

describe('export', () => {
  it('emits valid-looking SVG and DXF', () => {
    const paths = [
      [
        { x: 0, y: 0 },
        { x: 10, y: 0 },
        { x: 10, y: 10 },
      ],
    ];
    const svg = toSVG(paths, 100, 50);
    expect(svg).toContain('width="100.000mm"');
    expect(svg).toContain('<path d="M0.000 0.000L10.000 0.000L10.000 10.000"/>');
    const dxf = toDXF(paths, 100, 50);
    expect(dxf).toContain('LWPOLYLINE');
    expect(dxf).toContain('CUT');
    expect(dxf).toContain('EOF');
    // y flipped: y=10 -> 40
    expect(dxf).toContain('40.000');
  });
});
