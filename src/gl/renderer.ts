// WebGL2 renderer: extruded puzzle pieces textured with the customer's photo,
// orbit camera, and colour-coded GPU picking for mouse interaction.

import { Vec2 } from '../geom/curve';
import { triangulate } from '../geom/triangulate';
import {
  Mat4,
  invert,
  lookAt,
  multiply,
  perspective,
  transformPoint,
  translationRotationZ,
} from './mat4';

export interface PieceMesh {
  vao: WebGLVertexArrayObject;
  count: number;
  cx: number; // centroid (puzzle space) — pieces rotate about this
  cy: number;
}

export interface PieceTransform {
  dx: number; // offset from home position
  dy: number;
  z: number;
  rot: number; // radians, about z through the centroid
  highlight: number; // 0..1
}

const VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
uniform mat4 uModel;
uniform mat4 uViewProj;
out vec3 vNormal;
out vec2 vUV;
void main() {
  vec4 wp = uModel * vec4(aPos, 1.0);
  gl_Position = uViewProj * wp;
  vNormal = mat3(uModel) * aNormal;
  vUV = aUV;
}`;

const FS = `#version 300 es
precision highp float;
in vec3 vNormal;
in vec2 vUV;
uniform sampler2D uTex;
uniform vec3 uLight;
uniform float uHighlight;
uniform float uUseTexture;
out vec4 frag;
void main() {
  vec3 n = normalize(vNormal);
  vec3 base;
  if (n.z > 0.5) {
    base = mix(vec3(0.75, 0.6, 0.4), texture(uTex, vUV).rgb, uUseTexture);
  } else if (n.z < -0.5) {
    base = vec3(0.4, 0.29, 0.17);
  } else {
    base = vec3(0.66, 0.5, 0.3);
  }
  float diff = max(dot(n, normalize(uLight)), 0.0);
  vec3 col = base * (0.4 + 0.6 * diff);
  col = mix(col, vec3(1.0, 0.85, 0.35), uHighlight * 0.45);
  frag = vec4(col, 1.0);
}`;

const PICK_VS = `#version 300 es
layout(location=0) in vec3 aPos;
uniform mat4 uModel;
uniform mat4 uViewProj;
void main() { gl_Position = uViewProj * uModel * vec4(aPos, 1.0); }`;

const PICK_FS = `#version 300 es
precision highp float;
uniform vec3 uIdColor;
out vec4 frag;
void main() { frag = vec4(uIdColor, 1.0); }`;

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error('shader: ' + gl.getShaderInfoLog(sh));
  }
  return sh;
}

function link(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error('link: ' + gl.getProgramInfoLog(p));
  }
  return p;
}

export class OrbitCamera {
  theta = -0.35; // around z
  phi = 0.9; // from vertical
  dist = 900;
  target: [number, number, number] = [0, 0, 0];

  viewProj(aspect: number): Mat4 {
    const eye: [number, number, number] = [
      this.target[0] + this.dist * Math.sin(this.phi) * Math.cos(this.theta),
      this.target[1] + this.dist * Math.sin(this.phi) * Math.sin(this.theta),
      this.target[2] + this.dist * Math.cos(this.phi),
    ];
    const view = lookAt(eye, this.target, [0, 0, 1]);
    const proj = perspective((42 * Math.PI) / 180, aspect, this.dist / 50, this.dist * 8);
    return multiply(proj, view);
  }
}

export class Renderer {
  readonly gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private pickProgram: WebGLProgram;
  private uModel: WebGLUniformLocation;
  private uViewProj: WebGLUniformLocation;
  private uLight: WebGLUniformLocation;
  private uHighlight: WebGLUniformLocation;
  private uUseTexture: WebGLUniformLocation;
  private pUModel: WebGLUniformLocation;
  private pUViewProj: WebGLUniformLocation;
  private pUIdColor: WebGLUniformLocation;
  private texture: WebGLTexture | null = null;
  hasTexture = false;
  private pickFbo: WebGLFramebuffer | null = null;
  private pickW = 0;
  private pickH = 0;
  private refVao: WebGLVertexArrayObject | null = null;
  private refCount = 0;

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { antialias: true });
    if (!gl) throw new Error('此浏览器不支持 WebGL2');
    this.gl = gl;
    this.program = link(gl, VS, FS);
    this.pickProgram = link(gl, PICK_VS, PICK_FS);
    this.uModel = gl.getUniformLocation(this.program, 'uModel')!;
    this.uViewProj = gl.getUniformLocation(this.program, 'uViewProj')!;
    this.uLight = gl.getUniformLocation(this.program, 'uLight')!;
    this.uHighlight = gl.getUniformLocation(this.program, 'uHighlight')!;
    this.uUseTexture = gl.getUniformLocation(this.program, 'uUseTexture')!;
    this.pUModel = gl.getUniformLocation(this.pickProgram, 'uModel')!;
    this.pUViewProj = gl.getUniformLocation(this.pickProgram, 'uViewProj')!;
    this.pUIdColor = gl.getUniformLocation(this.pickProgram, 'uIdColor')!;
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE); // outlines are user-generated; winding is not trusted
  }

  resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.floor(this.canvas.clientWidth * dpr);
    const h = Math.floor(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  setTexture(source: TexImageSource): void {
    const gl = this.gl;
    if (this.texture) gl.deleteTexture(this.texture);
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1); // image rows -> v axis
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);
    this.texture = tex;
    this.hasTexture = true;
  }

  /** Build an extruded mesh for one piece outline. UVs map the outline onto the photo. */
  buildPiece(
    outline: Vec2[],
    centroid: Vec2,
    thickness: number,
    boardW: number,
    boardH: number,
  ): PieceMesh {
    const gl = this.gl;
    const n = outline.length;
    const tris = triangulate(outline);

    const pos: number[] = [];
    const nrm: number[] = [];
    const uv: number[] = [];
    const idx: number[] = [];

    // Top face (z = thickness).
    const topBase = 0;
    for (const p of outline) {
      pos.push(p.x, p.y, thickness);
      nrm.push(0, 0, 1);
      uv.push(p.x / boardW, p.y / boardH);
    }
    for (const t of tris) idx.push(topBase + t);

    // Bottom face (z = 0), reversed winding.
    const botBase = n;
    for (const p of outline) {
      pos.push(p.x, p.y, 0);
      nrm.push(0, 0, -1);
      uv.push(p.x / boardW, p.y / boardH);
    }
    for (let i = 0; i < tris.length; i += 3) {
      idx.push(botBase + tris[i], botBase + tris[i + 2], botBase + tris[i + 1]);
    }

    // Side walls: outward normal = edge perpendicular pointing away from centroid.
    for (let i = 0; i < n; i++) {
      const p = outline[i];
      const q = outline[(i + 1) % n];
      const ex = q.x - p.x;
      const ey = q.y - p.y;
      const el = Math.hypot(ex, ey) || 1;
      let nx = -ey / el;
      let ny = ex / el;
      const mx = (p.x + q.x) / 2 - centroid.x;
      const my = (p.y + q.y) / 2 - centroid.y;
      if (nx * mx + ny * my < 0) {
        nx = -nx;
        ny = -ny;
      }
      const base = pos.length / 3;
      pos.push(p.x, p.y, 0, q.x, q.y, 0, q.x, q.y, thickness, p.x, p.y, thickness);
      for (let k = 0; k < 4; k++) {
        nrm.push(nx, ny, 0);
        uv.push(0, 0);
      }
      idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }

    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    const interleaved = new Float32Array((pos.length / 3) * 8);
    for (let i = 0; i < pos.length / 3; i++) {
      interleaved[i * 8 + 0] = pos[i * 3 + 0];
      interleaved[i * 8 + 1] = pos[i * 3 + 1];
      interleaved[i * 8 + 2] = pos[i * 3 + 2];
      interleaved[i * 8 + 3] = nrm[i * 3 + 0];
      interleaved[i * 8 + 4] = nrm[i * 3 + 1];
      interleaved[i * 8 + 5] = nrm[i * 3 + 2];
      interleaved[i * 8 + 6] = uv[i * 2 + 0];
      interleaved[i * 8 + 7] = uv[i * 2 + 1];
    }
    const vbo = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, interleaved, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 32, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 32, 12);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 32, 24);
    const ibo = gl.createBuffer()!;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(idx), gl.STATIC_DRAW);
    gl.bindVertexArray(null);

    return { vao, count: idx.length, cx: centroid.x, cy: centroid.y };
  }

  deletePiece(mesh: PieceMesh): void {
    this.gl.deleteVertexArray(mesh.vao);
  }

  /** Flat textured quad at z=0 spanning the board — used as the reference photo. */
  private ensureRefQuad(boardW: number, boardH: number): void {
    if (this.refVao) return;
    const gl = this.gl;
    const data = new Float32Array([
      0, 0, 0, 0, 0, 1, 0, 0,
      boardW, 0, 0, 0, 0, 1, 1, 0,
      boardW, boardH, 0, 0, 0, 1, 1, 1,
      0, boardH, 0, 0, 0, 1, 0, 1,
    ]);
    this.refVao = gl.createVertexArray()!;
    gl.bindVertexArray(this.refVao);
    const vbo = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 32, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 32, 12);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 32, 24);
    const ibo = gl.createBuffer()!;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    this.refCount = 6;
  }

  draw(
    viewProj: Mat4,
    meshes: PieceMesh[],
    transforms: PieceTransform[],
    opts: { showReference: boolean; boardW: number; boardH: number; refZ: number },
  ): void {
    const gl = this.gl;
    this.resize();
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.13, 0.14, 0.17, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.uViewProj, false, viewProj);
    gl.uniform3f(this.uLight, 0.35, -0.45, 1.0);
    gl.uniform1f(this.uUseTexture, this.hasTexture ? 1 : 0);
    if (this.texture) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
    }

    if (opts.showReference && this.hasTexture) {
      this.ensureRefQuad(opts.boardW, opts.boardH);
      gl.uniformMatrix4fv(this.uModel, false, translationRotationZ(0, 0, opts.refZ, 0));
      gl.uniform1f(this.uHighlight, 0);
      gl.bindVertexArray(this.refVao);
      gl.drawElements(gl.TRIANGLES, this.refCount, gl.UNSIGNED_INT, 0);
    }

    for (let i = 0; i < meshes.length; i++) {
      const m = meshes[i];
      const t = transforms[i];
      const model = multiply(
        translationRotationZ(m.cx + t.dx, m.cy + t.dy, t.z, t.rot),
        translationRotationZ(-m.cx, -m.cy, 0, 0),
      );
      gl.uniformMatrix4fv(this.uModel, false, model);
      gl.uniform1f(this.uHighlight, t.highlight);
      gl.bindVertexArray(m.vao);
      gl.drawElements(gl.TRIANGLES, m.count, gl.UNSIGNED_INT, 0);
    }
    gl.bindVertexArray(null);
  }

  /** Render piece ids to an offscreen buffer and read back the id at (cssX, cssY). */
  pick(
    viewProj: Mat4,
    meshes: PieceMesh[],
    transforms: PieceTransform[],
    cssX: number,
    cssY: number,
  ): number {
    const gl = this.gl;
    this.resize();
    const w = this.canvas.width;
    const h = this.canvas.height;
    if (!this.pickFbo || this.pickW !== w || this.pickH !== h) {
      if (this.pickFbo) gl.deleteFramebuffer(this.pickFbo);
      const fbo = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      const color = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, color);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, color, 0);
      const depth = gl.createRenderbuffer()!;
      gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, w, h);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      this.pickFbo = fbo;
      this.pickW = w;
      this.pickH = h;
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.pickFbo);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.pickProgram);
    gl.uniformMatrix4fv(this.pUViewProj, false, viewProj);
    for (let i = 0; i < meshes.length; i++) {
      const m = meshes[i];
      const t = transforms[i];
      const model = multiply(
        translationRotationZ(m.cx + t.dx, m.cy + t.dy, t.z, t.rot),
        translationRotationZ(-m.cx, -m.cy, 0, 0),
      );
      const id = i + 1;
      gl.uniformMatrix4fv(this.pUModel, false, model);
      gl.uniform3f(
        this.pUIdColor,
        (id & 255) / 255,
        ((id >> 8) & 255) / 255,
        ((id >> 16) & 255) / 255,
      );
      gl.bindVertexArray(m.vao);
      gl.drawElements(gl.TRIANGLES, m.count, gl.UNSIGNED_INT, 0);
    }
    const dprX = w / this.canvas.clientWidth;
    const dprY = h / this.canvas.clientHeight;
    const px = new Uint8Array(4);
    gl.readPixels(
      Math.floor(cssX * dprX),
      h - 1 - Math.floor(cssY * dprY),
      1,
      1,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      px,
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindVertexArray(null);
    const id = px[0] | (px[1] << 8) | (px[2] << 16);
    return id - 1;
  }

  /** Unproject a screen point to a ray, intersect with the plane z = planeZ. */
  screenToPlane(
    viewProj: Mat4,
    cssX: number,
    cssY: number,
    planeZ: number,
  ): Vec2 | null {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    const ndcX = (cssX / w) * 2 - 1;
    const ndcY = 1 - (cssY / h) * 2;
    const inv = invert(viewProj);
    const near = transformPoint(inv, ndcX, ndcY, -1);
    const far = transformPoint(inv, ndcX, ndcY, 1);
    const nx = near[0] / near[3];
    const ny = near[1] / near[3];
    const nz = near[2] / near[3];
    const fx = far[0] / far[3];
    const fy = far[1] / far[3];
    const fz = far[2] / far[3];
    const dz = fz - nz;
    if (Math.abs(dz) < 1e-9) return null;
    const t = (planeZ - nz) / dz;
    return { x: nx + (fx - nx) * t, y: ny + (fy - ny) * t };
  }
}
