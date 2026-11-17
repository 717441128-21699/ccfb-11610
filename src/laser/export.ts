// Laser-machine export: SVG (for LightBurn / RDWorks import) and DXF (R12-ish
// ASCII, LWPOLYLINE entities on a CUT layer). Input paths are in millimetres.

import { Vec2 } from '../geom/curve';

function fmt(x: number): string {
  return x.toFixed(3);
}

export function toSVG(paths: Vec2[][], widthMm: number, heightMm: number): string {
  const parts: string[] = [];
  parts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${fmt(widthMm)}mm" height="${fmt(
      heightMm,
    )}mm" viewBox="0 0 ${fmt(widthMm)} ${fmt(heightMm)}">`,
  );
  parts.push(
    `<g fill="none" stroke="#ff0000" stroke-width="0.1" stroke-linecap="round">`,
  );
  for (const p of paths) {
    if (p.length < 2) continue;
    let d = `M${fmt(p[0].x)} ${fmt(p[0].y)}`;
    for (let i = 1; i < p.length; i++) d += `L${fmt(p[i].x)} ${fmt(p[i].y)}`;
    parts.push(`<path d="${d}"/>`);
  }
  parts.push('</g></svg>');
  return parts.join('\n');
}

/**
 * Minimal ASCII DXF. Y is flipped (DXF is y-up, our space is y-down) so the
 * board comes out upright in CAD. Each polyline is one LWPOLYLINE on layer CUT.
 */
export function toDXF(paths: Vec2[][], _widthMm: number, heightMm: number): string {
  const out: string[] = [];
  const push = (code: number, value: string | number) => {
    out.push(String(code), String(value));
  };
  push(0, 'SECTION');
  push(2, 'HEADER');
  push(9, '$INSUNITS');
  push(70, 4); // millimetres
  push(0, 'ENDSEC');
  push(0, 'SECTION');
  push(2, 'TABLES');
  push(0, 'TABLE');
  push(2, 'LAYER');
  push(0, 'LAYER');
  push(2, 'CUT');
  push(70, 0);
  push(62, 1); // red
  push(6, 'CONTINUOUS');
  push(0, 'ENDTAB');
  push(0, 'ENDSEC');
  push(0, 'SECTION');
  push(2, 'ENTITIES');
  for (const p of paths) {
    if (p.length < 2) continue;
    push(0, 'LWPOLYLINE');
    push(8, 'CUT');
    push(90, p.length);
    push(70, 0);
    for (const pt of p) {
      push(10, fmt(pt.x));
      push(20, fmt(heightMm - pt.y)); // flip to y-up
    }
  }
  push(0, 'ENDSEC');
  push(0, 'EOF');
  return out.join('\n');
}

export function downloadText(filename: string, text: string, mime: string): void {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}
