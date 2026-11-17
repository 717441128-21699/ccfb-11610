// Application glue: UI state, puzzle generation, laser export, and the
// interactive 3D preview (orbit camera, piece dragging, snap-to-fit).

import { Vec2, mulberry32 } from './geom/curve';
import { generatePuzzle, Puzzle, Rect } from './puzzle/generator';
import { detectFaces } from './puzzle/faces';
import { orderCutPaths, applyOrder, OrderedCuts } from './laser/optimize';
import { toSVG, toDXF, downloadText } from './laser/export';
import { OrbitCamera, PieceMesh, PieceTransform, Renderer } from './gl/renderer';

interface PieceState extends PieceTransform {
  locked: boolean;
}

interface Tween {
  i: number;
  from: PieceTransform;
  to: PieceTransform;
  t0: number;
  dur: number;
}

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error('#' + id + ' missing');
  return el as T;
};

export class App {
  private renderer: Renderer;
  private camera = new OrbitCamera();
  private img: HTMLImageElement | null = null;
  private imgW = 1200;
  private imgH = 900;
  private faces: Rect[] = [];
  private puzzle: Puzzle | null = null;
  private meshes: PieceMesh[] = [];
  private pieces: PieceState[] = [];
  private ordered: OrderedCuts | null = null;
  private orderedPathsMm: Vec2[][] = [];
  private pxPerMm = 2.4;
  private thicknessPx = 7;
  private tweens: Tween[] = [];
  private dragging = -1;
  private grabOffset: Vec2 = { x: 0, y: 0 };
  private orbiting = false;
  private panning = false;
  private lastPointer: Vec2 = { x: 0, y: 0 };
  private hover = -1;
  private lastHoverPick = 0;
  private explode = 0;
  private explodeSeed: number[] = [];

  private glCanvas = $<HTMLCanvasElement>('glCanvas');
  private faceCanvas = $<HTMLCanvasElement>('faceCanvas');

  constructor() {
    this.renderer = new Renderer(this.glCanvas);
    this.bindUI();
    this.loadDefaultImage();
    requestAnimationFrame(() => this.frame());
  }

  // ------------------------------------------------------------- UI wiring

  private bindUI(): void {
    $<HTMLInputElement>('fileInput').addEventListener('change', (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) this.loadImage(URL.createObjectURL(file));
    });
    $('btnGenerate').addEventListener('click', () => this.generate());
    $('btnDetectFaces').addEventListener('click', () => this.runFaceDetection());
    $('btnScatter').addEventListener('click', () => this.scatter());
    $('btnAssemble').addEventListener('click', () => this.assembleAll());
    $('btnSvg').addEventListener('click', () => this.exportFile('svg'));
    $('btnDxf').addEventListener('click', () => this.exportFile('dxf'));
    $<HTMLInputElement>('explode').addEventListener('input', (e) => {
      this.explode = parseFloat((e.target as HTMLInputElement).value);
      this.applyExplode();
    });

    // 3D interaction
    this.glCanvas.addEventListener('pointerdown', (e) => this.onPointerDown(e));
    this.glCanvas.addEventListener('pointermove', (e) => this.onPointerMove(e));
    window.addEventListener('pointerup', (e) => this.onPointerUp(e));
    this.glCanvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.camera.dist *= e.deltaY > 0 ? 1.12 : 1 / 1.12;
      this.camera.dist = Math.max(50, Math.min(this.camera.dist, 20000));
    }, { passive: false });
    this.glCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
    window.addEventListener('keydown', (e) => {
      if (this.dragging < 0) return;
      if ((e.target as HTMLElement).tagName === 'INPUT') return;
      const step = e.shiftKey ? Math.PI / 180 : Math.PI / 12;
      if (e.key === 'q' || e.key === 'Q') this.pieces[this.dragging].rot += step;
      if (e.key === 'e' || e.key === 'E') this.pieces[this.dragging].rot -= step;
    });

    // Face-box editor
    this.faceCanvas.addEventListener('pointerdown', (e) => this.onFacePointerDown(e));
  }

  private num(id: string): number {
    return parseFloat($<HTMLInputElement>(id).value) || 0;
  }

  // ------------------------------------------------------------- images

  private loadDefaultImage(): void {
    const c = document.createElement('canvas');
    c.width = this.imgW;
    c.height = this.imgH;
    const ctx = c.getContext('2d')!;
    const grad = ctx.createLinearGradient(0, 0, 0, this.imgH);
    grad.addColorStop(0, '#2b4a6f');
    grad.addColorStop(0.55, '#7a4a6e');
    grad.addColorStop(1, '#c9722f');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, this.imgW, this.imgH);
    ctx.fillStyle = '#f5d67b';
    ctx.beginPath();
    ctx.arc(this.imgW * 0.7, this.imgH * 0.3, 90, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#31404f';
    for (let i = 0; i < 7; i++) {
      const x = (i / 6) * this.imgW;
      ctx.beginPath();
      ctx.moveTo(x - 260, this.imgH);
      ctx.lineTo(x, this.imgH * (0.45 + 0.12 * Math.sin(i * 2.3)));
      ctx.lineTo(x + 260, this.imgH);
      ctx.fill();
    }
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.font = 'bold 56px system-ui';
    ctx.fillText('示例照片 · 可上传顾客照片', 60, this.imgH - 60);
    const img = new Image();
    img.onload = () => {
      this.img = img;
      this.onImageReady();
    };
    img.src = c.toDataURL();
  }

  private loadImage(url: string): void {
    const img = new Image();
    img.onload = () => {
      this.img = img;
      this.imgW = img.naturalWidth;
      this.imgH = img.naturalHeight;
      this.onImageReady();
    };
    img.src = url;
  }

  private onImageReady(): void {
    if (!this.img) return;
    // Cap texture size for the GPU.
    const cap = 2048;
    const scale = Math.min(1, cap / Math.max(this.imgW, this.imgH));
    const texSource = document.createElement('canvas');
    texSource.width = Math.round(this.imgW * scale);
    texSource.height = Math.round(this.imgH * scale);
    texSource.getContext('2d')!.drawImage(this.img, 0, 0, texSource.width, texSource.height);
    this.renderer.setTexture(texSource);
    this.runFaceDetection();
  }

  private async runFaceDetection(): Promise<void> {
    if (!this.img) return;
    this.faces = await detectFaces(this.img, this.imgW, this.imgH);
    this.drawFaceEditor();
    this.generate();
  }

  // ------------------------------------------------------------- generation

  private generate(): void {
    if (!this.img) return;
    const boardWmm = Math.max(50, this.num('boardWidth'));
    this.pxPerMm = this.imgW / boardWmm;
    const minRadiusPx = this.num('minRadius') * this.pxPerMm;
    this.thicknessPx = this.num('thickness') * this.pxPerMm;

    const t0 = performance.now();
    const puzzle = generatePuzzle({
      width: this.imgW,
      height: this.imgH,
      targetPieces: Math.max(20, Math.round(this.num('pieceCount'))),
      minRadius: minRadiusPx,
      specialCount: Math.max(0, Math.round(this.num('specialCount'))),
      seed: Math.round(this.num('seed')),
      faceRects: this.faces,
    });
    const tGen = performance.now() - t0;

    // Rebuild GPU meshes.
    for (const m of this.meshes) this.renderer.deletePiece(m);
    this.meshes = puzzle.pieces.map((p) =>
      this.renderer.buildPiece(p.outline, p.centroid, this.thicknessPx, puzzle.width, puzzle.height),
    );
    this.pieces = puzzle.pieces.map(() => ({
      dx: 0, dy: 0, z: 0, rot: 0, highlight: 0, locked: false,
    }));
    const rng = mulberry32(Math.round(this.num('seed')) ^ 0x9e3779b9);
    this.explodeSeed = puzzle.pieces.map(() => rng());
    this.tweens = [];
    this.puzzle = puzzle;

    // Camera frames the board.
    this.camera.target = [puzzle.width / 2, puzzle.height / 2, 0];
    this.camera.dist = Math.max(puzzle.width, puzzle.height) * 1.35;

    // Laser path optimisation (paths converted to millimetres).
    const t1 = performance.now();
    const pathsMm = puzzle.cutPaths.map((p) =>
      p.map((q) => ({ x: q.x / this.pxPerMm, y: q.y / this.pxPerMm })),
    );
    this.ordered = orderCutPaths(pathsMm, 4000);
    this.orderedPathsMm = applyOrder(pathsMm, this.ordered);
    const tOpt = performance.now() - t1;

    const warn = $('warnings');
    warn.textContent = puzzle.warnings.join('\n');

    $('genStats').textContent =
      `${puzzle.cols}×${puzzle.rows} = ${puzzle.pieces.length} 块 · ` +
      `生成 ${tGen.toFixed(0)}ms · 路径优化 ${tOpt.toFixed(0)}ms`;

    const o = this.ordered;
    $('laserStats').textContent =
      `切割线 ${o.order.length} 条(共边只切一次)\n` +
      `切割总长 ${(o.cutLength / 1000).toFixed(2)} m\n` +
      `空走 ${o.naiveTravel.toFixed(0)} → ${o.travel.toFixed(0)} mm` +
      `(省 ${((1 - o.travel / Math.max(o.naiveTravel, 1e-9)) * 100).toFixed(0)}%)\n` +
      `最小圆角 ${(puzzle.minRadiusAchieved / this.pxPerMm).toFixed(2)} mm`;

    this.updateProgress();
  }

  // ------------------------------------------------------------- interaction

  private viewProj() {
    const w = this.glCanvas.clientWidth || 1;
    const h = this.glCanvas.clientHeight || 1;
    return this.camera.viewProj(w / h);
  }

  private onPointerDown(e: PointerEvent): void {
    this.glCanvas.setPointerCapture(e.pointerId);
    this.lastPointer = { x: e.offsetX, y: e.offsetY };
    if (e.button === 2 || e.shiftKey) {
      this.panning = true;
      return;
    }
    if (!this.puzzle) return;
    const id = this.renderer.pick(
      this.viewProj(), this.meshes, this.pieces, e.offsetX, e.offsetY,
    );
    if (id >= 0 && !this.pieces[id].locked) {
      this.dragging = id;
      const hit = this.renderer.screenToPlane(
        this.viewProj(), e.offsetX, e.offsetY, this.pieces[id].z,
      );
      if (hit) {
        const m = this.meshes[id];
        this.grabOffset = {
          x: m.cx + this.pieces[id].dx - hit.x,
          y: m.cy + this.pieces[id].dy - hit.y,
        };
      }
    } else {
      this.orbiting = true;
    }
  }

  private onPointerMove(e: PointerEvent): void {
    const dx = e.offsetX - this.lastPointer.x;
    const dy = e.offsetY - this.lastPointer.y;
    this.lastPointer = { x: e.offsetX, y: e.offsetY };

    if (this.dragging >= 0) {
      const p = this.pieces[this.dragging];
      const hit = this.renderer.screenToPlane(
        this.viewProj(), e.offsetX, e.offsetY, p.z,
      );
      if (hit) {
        const m = this.meshes[this.dragging];
        p.dx = hit.x + this.grabOffset.x - m.cx;
        p.dy = hit.y + this.grabOffset.y - m.cy;
        p.highlight = this.inSnapRange(p) ? 1 : 0.35;
      }
      return;
    }
    if (this.orbiting) {
      this.camera.theta += dx * 0.008;
      this.camera.phi = Math.max(0.06, Math.min(1.5, this.camera.phi + dy * 0.008));
      return;
    }
    if (this.panning) {
      const s = this.camera.dist * 0.0012;
      const st = Math.sin(this.camera.theta);
      const ct = Math.cos(this.camera.theta);
      const sf = Math.sin(this.camera.phi);
      const cf = Math.cos(this.camera.phi);
      // Screen-right and screen-up in world space (z-up world).
      const rx = -st, ry = ct;
      const ux = -cf * ct, uy = cf * st, uz = sf;
      this.camera.target[0] -= rx * dx * s - ux * dy * s;
      this.camera.target[1] -= ry * dx * s - uy * dy * s;
      this.camera.target[2] -= -uz * dy * s;
      return;
    }
    // Hover highlight (throttled — picking renders the scene offscreen).
    const now = performance.now();
    if (this.puzzle && now - this.lastHoverPick > 90) {
      this.lastHoverPick = now;
      const id = this.renderer.pick(
        this.viewProj(), this.meshes, this.pieces, e.offsetX, e.offsetY,
      );
      if (id !== this.hover) {
        if (this.hover >= 0 && this.hover !== this.dragging) {
          this.pieces[this.hover].highlight = 0;
        }
        this.hover = id;
      }
      if (this.hover >= 0 && !this.pieces[this.hover].locked) {
        this.pieces[this.hover].highlight = 0.3;
      }
    }
  }

  private onPointerUp(_e: PointerEvent): void {
    this.orbiting = false;
    this.panning = false;
    if (this.dragging >= 0) {
      const p = this.pieces[this.dragging];
      if (this.inSnapRange(p)) {
        // Snap: tween home and lock the piece.
        this.tweens.push({
          i: this.dragging,
          from: { ...p },
          to: { dx: 0, dy: 0, z: 0, rot: 0, highlight: 0 },
          t0: performance.now(),
          dur: 180,
        });
        p.locked = true;
        this.updateProgress();
      } else {
        p.highlight = 0;
        // Re-apply the explode height so a dropped piece rejoins its layer.
        if (this.puzzle && this.explode > 0) {
          const span = Math.max(this.puzzle.width, this.puzzle.height);
          p.z = this.explode * span * 0.25 * (0.25 + this.explodeSeed[this.dragging]);
        }
      }
      this.dragging = -1;
    }
  }

  private inSnapRange(p: PieceState): boolean {
    if (!this.puzzle) return false;
    const cell =
      (this.puzzle.width / this.puzzle.cols + this.puzzle.height / this.puzzle.rows) / 2;
    const distHome = Math.hypot(p.dx, p.dy);
    let rot = p.rot % (Math.PI * 2);
    if (rot > Math.PI) rot -= Math.PI * 2;
    if (rot < -Math.PI) rot += Math.PI * 2;
    return distHome < cell * 0.45 && Math.abs(rot) < Math.PI / 10;
  }

  // ------------------------------------------------------------- scatter / assemble

  private scatter(): void {
    if (!this.puzzle) return;
    const rng = mulberry32((Math.random() * 1e9) | 0);
    const cx = this.puzzle.width / 2;
    const cy = this.puzzle.height / 2;
    const R = Math.hypot(this.puzzle.width, this.puzzle.height);
    const now = performance.now();
    this.pieces.forEach((p, i) => {
      p.locked = false;
      const m = this.meshes[i];
      const ang = rng() * Math.PI * 2;
      const rad = R * (0.42 + rng() * 0.4);
      const targetX = cx + Math.cos(ang) * rad - m.cx;
      const targetY = cy + Math.sin(ang) * rad * 0.75 - m.cy;
      this.tweens.push({
        i,
        from: { ...p },
        to: {
          dx: targetX, dy: targetY, z: 0,
          rot: (rng() - 0.5) * Math.PI * 2, highlight: 0,
        },
        t0: now,
        dur: 700 + rng() * 300,
      });
    });
    this.updateProgress();
  }

  private assembleAll(): void {
    const now = performance.now();
    this.pieces.forEach((p, i) => {
      p.locked = true;
      this.tweens.push({
        i,
        from: { ...p },
        to: { dx: 0, dy: 0, z: 0, rot: 0, highlight: 0 },
        t0: now,
        dur: 500 + (i % 40) * 12,
      });
    });
    this.updateProgress();
  }

  private applyExplode(): void {
    if (!this.puzzle) return;
    const span = Math.max(this.puzzle.width, this.puzzle.height);
    this.pieces.forEach((p, i) => {
      if (this.dragging === i) return;
      p.z = this.explode * span * 0.25 * (0.25 + this.explodeSeed[i]);
    });
  }

  private updateProgress(): void {
    const placed = this.pieces.filter((p) => p.locked).length;
    const total = this.pieces.length;
    $('progress').textContent =
      total === 0 ? '' :
      placed === total ? `🎉 全部 ${total} 块已拼好!` : `已放置 ${placed} / ${total}`;
  }

  // ------------------------------------------------------------- face editor

  private faceScale(): number {
    return this.faceCanvas.width / this.imgW;
  }

  private drawFaceEditor(): void {
    const ctx = this.faceCanvas.getContext('2d')!;
    const s = this.faceScale();
    this.faceCanvas.height = Math.round(this.imgH * s);
    if (this.img) ctx.drawImage(this.img, 0, 0, this.faceCanvas.width, this.faceCanvas.height);
    ctx.strokeStyle = '#ff5544';
    ctx.lineWidth = 2;
    for (const f of this.faces) {
      ctx.strokeRect(f.x * s, f.y * s, f.w * s, f.h * s);
    }
    if (this.faces.length === 0) {
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.font = '12px system-ui';
      ctx.fillText('未检测到人脸,可拖框手动标记', 8, 18);
    }
  }

  private faceDragStart: Vec2 | null = null;

  private onFacePointerDown(e: PointerEvent): void {
    this.faceCanvas.setPointerCapture(e.pointerId);
    const s = this.faceScale();
    const x = e.offsetX / s;
    const y = e.offsetY / s;
    const hit = this.faces.findIndex(
      (f) => x >= f.x && x <= f.x + f.w && y >= f.y && y <= f.y + f.h,
    );
    if (hit >= 0) {
      this.faces.splice(hit, 1);
      this.drawFaceEditor();
      this.generate();
      return;
    }
    this.faceDragStart = { x, y };
    const move = (ev: PointerEvent) => {
      if (!this.faceDragStart) return;
      this.drawFaceEditor();
      const ctx = this.faceCanvas.getContext('2d')!;
      ctx.strokeStyle = '#ffdd44';
      ctx.lineWidth = 2;
      ctx.strokeRect(
        this.faceDragStart.x * s,
        this.faceDragStart.y * s,
        ev.offsetX - this.faceDragStart.x * s,
        ev.offsetY - this.faceDragStart.y * s,
      );
    };
    const up = (ev: PointerEvent) => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      if (!this.faceDragStart) return;
      const w = ev.offsetX / s - this.faceDragStart.x;
      const h = ev.offsetY / s - this.faceDragStart.y;
      if (Math.abs(w) > 8 / s && Math.abs(h) > 8 / s) {
        this.faces.push({
          x: Math.min(this.faceDragStart.x, this.faceDragStart.x + w),
          y: Math.min(this.faceDragStart.y, this.faceDragStart.y + h),
          w: Math.abs(w),
          h: Math.abs(h),
        });
      }
      this.faceDragStart = null;
      this.drawFaceEditor();
      this.generate();
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  }

  // ------------------------------------------------------------- export

  private exportFile(kind: 'svg' | 'dxf'): void {
    if (!this.puzzle || this.orderedPathsMm.length === 0) return;
    const wMm = this.puzzle.width / this.pxPerMm;
    const hMm = this.puzzle.height / this.pxPerMm;
    const seed = Math.round(this.num('seed'));
    if (kind === 'svg') {
      downloadText(`puzzle-${seed}.svg`, toSVG(this.orderedPathsMm, wMm, hMm), 'image/svg+xml');
    } else {
      downloadText(`puzzle-${seed}.dxf`, toDXF(this.orderedPathsMm, wMm, hMm), 'application/dxf');
    }
  }

  // ------------------------------------------------------------- render loop

  private frame(): void {
    requestAnimationFrame(() => this.frame());
    // Advance tweens.
    if (this.tweens.length) {
      const now = performance.now();
      this.tweens = this.tweens.filter((t) => {
        const p = this.pieces[t.i];
        if (!p) return false;
        let k = (now - t.t0) / t.dur;
        if (k >= 1) k = 1;
        const e = 1 - Math.pow(1 - k, 3);
        p.dx = t.from.dx + (t.to.dx - t.from.dx) * e;
        p.dy = t.from.dy + (t.to.dy - t.from.dy) * e;
        p.z = t.from.z + (t.to.z - t.from.z) * e;
        p.rot = t.from.rot + (t.to.rot - t.from.rot) * e;
        p.highlight = t.from.highlight + (t.to.highlight - t.from.highlight) * e;
        return k < 1;
      });
    }
    if (!this.puzzle) return;
    this.renderer.draw(this.viewProj(), this.meshes, this.pieces, {
      showReference: $<HTMLInputElement>('showRef').checked,
      boardW: this.puzzle.width,
      boardH: this.puzzle.height,
      refZ: -this.thicknessPx * 4,
    });
  }
}
