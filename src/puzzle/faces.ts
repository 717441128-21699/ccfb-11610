// Face detection: use the browser FaceDetector API when available, otherwise
// fall back to a skin-tone blob heuristic. Results are editable in the UI.

import { Rect } from './generator';

interface FaceDetectorLike {
  detect(image: CanvasImageSource): Promise<Array<{ boundingBox: DOMRect }>>;
}

export async function detectFaces(
  img: HTMLImageElement | ImageBitmap,
  width: number,
  height: number,
): Promise<Rect[]> {
  const FD = (window as unknown as { FaceDetector?: new () => FaceDetectorLike })
    .FaceDetector;
  if (FD) {
    try {
      const faces = await new FD().detect(img);
      if (faces.length > 0) {
        return faces.map((f) => ({
          x: f.boundingBox.x,
          y: f.boundingBox.y,
          w: f.boundingBox.width,
          h: f.boundingBox.height,
        }));
      }
    } catch {
      // fall through to heuristic
    }
  }
  return skinToneHeuristic(img, width, height);
}

/**
 * Crude but offline: threshold skin tones in YCbCr, find connected blobs on a
 * downsampled grid, keep blob bounding boxes with face-like size/aspect.
 */
function skinToneHeuristic(
  img: HTMLImageElement | ImageBitmap,
  width: number,
  height: number,
): Rect[] {
  const GW = 96;
  const GH = Math.max(2, Math.round((GW * height) / width));
  const canvas = document.createElement('canvas');
  canvas.width = GW;
  canvas.height = GH;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return [];
  ctx.drawImage(img, 0, 0, GW, GH);
  const data = ctx.getImageData(0, 0, GW, GH).data;

  const skin = new Uint8Array(GW * GH);
  for (let i = 0; i < GW * GH; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
    const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
    if (cr > 133 && cr < 173 && cb > 77 && cb < 127 && r > 60) skin[i] = 1;
  }

  // Connected components (4-neighbour).
  const label = new Int32Array(GW * GH).fill(-1);
  const boxes: Rect[] = [];
  let nextLabel = 0;
  const stack: number[] = [];
  for (let i = 0; i < GW * GH; i++) {
    if (!skin[i] || label[i] >= 0) continue;
    let minX = GW;
    let maxX = 0;
    let minY = GH;
    let maxY = 0;
    let count = 0;
    stack.push(i);
    label[i] = nextLabel;
    while (stack.length) {
      const p = stack.pop()!;
      const px = p % GW;
      const py = (p / GW) | 0;
      count++;
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
      const neighbours = [
        px > 0 ? p - 1 : -1,
        px < GW - 1 ? p + 1 : -1,
        py > 0 ? p - GW : -1,
        py < GH - 1 ? p + GW : -1,
      ];
      for (const q of neighbours) {
        if (q >= 0 && skin[q] && label[q] < 0) {
          label[q] = nextLabel;
          stack.push(q);
        }
      }
    }
    const w = maxX - minX + 1;
    const h = maxY - minY + 1;
    const fill = count / (w * h);
    const aspect = w / h;
    // Face-like: big enough, not too sparse, plausible aspect.
    if (count >= GW * GH * 0.004 && fill > 0.35 && aspect > 0.35 && aspect < 2.2) {
      boxes.push({
        x: (minX / GW) * width,
        y: (minY / GH) * height,
        w: (w / GW) * width,
        h: (h / GH) * height,
      });
    }
    nextLabel++;
  }

  // Merge overlapping boxes, then shrink toward the upper 2/3 (faces sit high
  // in skin regions that include neck/hair).
  const merged: Rect[] = [];
  for (const b of boxes) {
    const hit = merged.find((m) => rectsOverlap(m, b));
    if (hit) {
      const x = Math.min(hit.x, b.x);
      const y = Math.min(hit.y, b.y);
      hit.w = Math.max(hit.x + hit.w, b.x + b.w) - x;
      hit.h = Math.max(hit.y + hit.h, b.y + b.h) - y;
      hit.x = x;
      hit.y = y;
    } else {
      merged.push({ ...b });
    }
  }
  return merged.map((b) => ({
    x: b.x + b.w * 0.12,
    y: b.y,
    w: b.w * 0.76,
    h: b.h * 0.72,
  }));
}

function rectsOverlap(a: Rect, b: Rect): boolean {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}
