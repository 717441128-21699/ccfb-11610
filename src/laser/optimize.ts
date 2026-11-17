// Cut-path ordering: shared edges are already stored once by the generator;
// here we sequence the polylines (choosing a direction for each) so the
// laser head's rapid-travel distance is near-minimal.
//
// Strategy: greedy nearest-neighbour tour construction, then 2-opt
// improvement (reversing tour segments, which flips the orientation of every
// polyline inside the segment). Runs under a time budget.

import { Vec2, dist, polylineLength } from '../geom/curve';

export interface OrderedCuts {
  /** Indices into the input path array, in cutting order. */
  order: number[];
  /** reversed[i] === true → cut path order[i] from its end to its start. */
  reversed: boolean[];
  /** Total rapid (non-cutting) travel in the input's units. */
  travel: number;
  /** Travel of the naive input order, for comparison. */
  naiveTravel: number;
  /** Total cutting length. */
  cutLength: number;
  /** Milliseconds spent optimising. */
  timeMs: number;
}

interface Node {
  idx: number;
  rev: boolean;
}

function endpoint(paths: Vec2[][], node: Node, which: 'start' | 'end'): Vec2 {
  const p = paths[node.idx];
  const first = node.rev ? p[p.length - 1] : p[0];
  const last = node.rev ? p[0] : p[p.length - 1];
  return which === 'start' ? first : last;
}

function tourTravel(paths: Vec2[][], tour: Node[]): number {
  let t = 0;
  for (let i = 1; i < tour.length; i++) {
    t += dist(endpoint(paths, tour[i - 1], 'end'), endpoint(paths, tour[i], 'start'));
  }
  return t;
}

export function orderCutPaths(paths: Vec2[][], timeBudgetMs = 4000): OrderedCuts {
  const t0 = performance.now();
  const n = paths.length;
  const cutLength = paths.reduce((s, p) => s + polylineLength(p), 0);
  if (n === 0) {
    return { order: [], reversed: [], travel: 0, naiveTravel: 0, cutLength: 0, timeMs: 0 };
  }
  const naive: Node[] = paths.map((_, i) => ({ idx: i, rev: false }));
  const naiveTravel = tourTravel(paths, naive);

  // ---- Greedy nearest neighbour (each path can be approached from either end).
  const used = new Array<boolean>(n).fill(false);
  const tour: Node[] = [];
  let current: Node = { idx: 0, rev: false };
  tour.push(current);
  used[0] = true;
  for (let placed = 1; placed < n; placed++) {
    const tail = endpoint(paths, current, 'end');
    let bestIdx = -1;
    let bestRev = false;
    let bestD = Infinity;
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      const p = paths[i];
      const dStart = dist(tail, p[0]);
      if (dStart < bestD) {
        bestD = dStart;
        bestIdx = i;
        bestRev = false;
      }
      const dEnd = dist(tail, p[p.length - 1]);
      if (dEnd < bestD) {
        bestD = dEnd;
        bestIdx = i;
        bestRev = true;
      }
    }
    current = { idx: bestIdx, rev: bestRev };
    used[bestIdx] = true;
    tour.push(current);
  }

  // ---- 2-opt: reversing tour[i..j] flips orientation of every node inside.
  // Gap cost only involves the two boundaries, so delta evaluation is O(1).
  const endOf = (node: Node): Vec2 => endpoint(paths, node, 'end');
  const startOf = (node: Node): Vec2 => endpoint(paths, node, 'start');
  let improved = true;
  let travel = tourTravel(paths, tour);
  while (improved && performance.now() - t0 < timeBudgetMs) {
    improved = false;
    for (let i = 0; i < n - 1; i++) {
      const aEnd = endOf(tour[i]);
      for (let j = i + 1; j < n; j++) {
        const before =
          dist(aEnd, startOf(tour[i + 1])) +
          (j + 1 < n ? dist(endOf(tour[j]), startOf(tour[j + 1])) : 0);
        // After reversing [i+1..j]: a connects to (flipped) old tour[j],
        // old tour[i+1] flipped connects to tour[j+1].
        const flippedJ: Node = { idx: tour[j].idx, rev: !tour[j].rev };
        const flippedI1: Node = { idx: tour[i + 1].idx, rev: !tour[i + 1].rev };
        const after =
          dist(aEnd, startOf(flippedJ)) +
          (j + 1 < n ? dist(endOf(flippedI1), startOf(tour[j + 1])) : 0);
        if (after < before - 1e-9) {
          // Apply: reverse segment and flip orientations.
          let lo = i + 1;
          let hi = j;
          while (lo < hi) {
            const tmp = tour[lo];
            tour[lo] = { idx: tour[hi].idx, rev: !tour[hi].rev };
            tour[hi] = { idx: tmp.idx, rev: !tmp.rev };
            lo++;
            hi--;
          }
          if (lo === hi) tour[lo] = { idx: tour[lo].idx, rev: !tour[lo].rev };
          travel -= before - after;
          improved = true;
        }
      }
      if (performance.now() - t0 > timeBudgetMs) break;
    }
  }

  return {
    order: tour.map((t) => t.idx),
    reversed: tour.map((t) => t.rev),
    travel,
    naiveTravel,
    cutLength,
    timeMs: performance.now() - t0,
  };
}

/** Materialise the ordered, oriented polylines (e.g. for export). */
export function applyOrder(
  paths: Vec2[][],
  ordered: OrderedCuts,
): Vec2[][] {
  return ordered.order.map((idx, i) => {
    const p = paths[idx];
    return ordered.reversed[i] ? p.slice().reverse() : p;
  });
}
