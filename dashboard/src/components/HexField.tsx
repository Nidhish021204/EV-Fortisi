import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { C, ramp, makeRankScale } from "../palette";
import type { Bundle, LayerId, PlanSite } from "../data";
import { LAYERS } from "../data";

/**
 * The primary object. A canvas-rendered H3 field with no basemap: tiles would
 * compete with the data and imply a precision the model does not claim.
 *
 * Geometry is pre-projected integer metres, so drawing is a straight affine
 * transform. 8,249 cells x 6 vertices redraws well inside a frame, which is why
 * pan and zoom can repaint rather than pre-tile.
 */

interface View { scale: number; tx: number; ty: number; }

interface Props {
  bundle: Bundle;
  layer: LayerId;
  budget: number;
  showStations: boolean;
  showPlan: boolean;
  hovered: number | null;
  onHover: (i: number | null) => void;
  onPickSite: (s: PlanSite | null) => void;
}

export default function HexField({
  bundle, layer, budget, showStations, showPlan, hovered, onHover, onPickSite,
}: Props) {
  const { grid, stations, plan } = bundle;
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [view, setView] = useState<View | null>(null);
  const drag = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);
  // Once the planner has panned or zoomed, their framing is theirs to keep.
  // Until then the field refits on every resize -- otherwise a transform
  // fitted at an early layout size survives a pane resize and puts the whole
  // field off-screen.
  const userMoved = useRef(false);

  const field = LAYERS.find((l) => l.id === layer)!;
  const values = grid[field.field] as number[];

  // Rank scaling: demand spans four orders of magnitude, so a linear ramp
  // would paint the entire city its darkest step and Midtown its lightest.
  const scale = useMemo(() => makeRankScale(values), [values]);

  const bounds = useMemo(() => {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < grid.count; i++) {
      if (grid.cx[i] < x0) x0 = grid.cx[i];
      if (grid.cx[i] > x1) x1 = grid.cx[i];
      if (grid.cy[i] < y0) y0 = grid.cy[i];
      if (grid.cy[i] > y1) y1 = grid.cy[i];
    }
    return { x0, y0, x1, y1 };
  }, [grid]);

  // Spatial hash over centroids so hover hit-testing is O(1), not O(n).
  const hash = useMemo(() => {
    const cell = 400; // metres
    const m = new Map<string, number[]>();
    for (let i = 0; i < grid.count; i++) {
      const k = `${Math.floor(grid.cx[i] / cell)},${Math.floor(grid.cy[i] / cell)}`;
      const b = m.get(k);
      if (b) b.push(i); else m.set(k, [i]);
    }
    return { cell, m };
  }, [grid]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;

    // A hidden pane, a background tab, or a display:none ancestor makes
    // ResizeObserver report 0x0. Accepting that zero leaves `size` empty, and
    // both the fit and the draw early-return -- so the field stays blank even
    // after the element regains size, because RO may not fire again. Zero
    // readings are therefore ignored, and the element is re-measured directly
    // on mount and whenever the document becomes visible.
    const measure = () => {
      const r = el.getBoundingClientRect();
      const w = Math.round(r.width);
      const h = Math.round(r.height);
      if (w > 0 && h > 0) {
        setSize((prev) => (prev.w === w && prev.h === h ? prev : { w, h }));
      }
    };

    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    document.addEventListener("visibilitychange", measure);
    window.addEventListener("focus", measure);
    // One deferred re-measure covers the case where layout settles a frame
    // after mount and RO has already delivered its only 0x0 reading.
    const raf = requestAnimationFrame(measure);

    return () => {
      ro.disconnect();
      document.removeEventListener("visibilitychange", measure);
      window.removeEventListener("focus", measure);
      cancelAnimationFrame(raf);
    };
  }, []);

  // Fit whenever the viewport changes, until the user takes over.
  useEffect(() => {
    if (!size.w || !size.h) return;
    if (view && userMoved.current) return;
    const pad = 48;
    const sx = (size.w - pad * 2) / (bounds.x1 - bounds.x0);
    const sy = (size.h - pad * 2) / (bounds.y1 - bounds.y0);
    const s = Math.min(sx, sy);
    setView({
      scale: s,
      tx: size.w / 2 - ((bounds.x0 + bounds.x1) / 2) * s,
      ty: size.h / 2 + ((bounds.y0 + bounds.y1) / 2) * s,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size, bounds]);

  const toScreen = useCallback(
    (x: number, y: number, v: View): [number, number] => [x * v.scale + v.tx, -y * v.scale + v.ty],
    [],
  );

  const draw = useCallback(() => {
    const cv = canvasRef.current;
    if (!cv || !view || !size.w) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (cv.width !== size.w * dpr || cv.height !== size.h * dpr) {
      cv.width = size.w * dpr;
      cv.height = size.h * dpr;
    }
    const g = cv.getContext("2d");
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = C.ground;
    g.fillRect(0, 0, size.w, size.h);

    const { scale: s, tx, ty } = view;
    const hairline = s > 0.0016;

    // ── field ────────────────────────────────────────────────────
    for (let i = 0; i < grid.count; i++) {
      const [rx, ry] = grid.rings[i];
      const v = values[i];
      const t = scale(v);
      if (t <= 0 && v <= 0) continue;

      g.beginPath();
      for (let k = 0; k < rx.length; k++) {
        const px = rx[k] * s + tx;
        const py = -ry[k] * s + ty;
        k === 0 ? g.moveTo(px, py) : g.lineTo(px, py);
      }
      g.closePath();
      g.fillStyle = ramp(t);
      g.fill();
      if (hairline) {
        g.strokeStyle = "rgba(13,13,12,0.55)";
        g.lineWidth = 0.5;
        g.stroke();
      }
    }

    // ── existing network: neutral and recessive ──────────────────
    // 863 bright marks would out-shout the field they are meant to annotate,
    // so these are hollow hairlines. DC fast sites get one extra pixel and a
    // touch more opacity, which is the only distinction they need.
    if (showStations) {
      g.lineWidth = 0.8;
      for (let i = 0; i < stations.x.length; i++) {
        const [px, py] = toScreen(stations.x[i], stations.y[i], view);
        if (px < -20 || py < -20 || px > size.w + 20 || py > size.h + 20) continue;
        const fast = stations.dcfc[i] > 0;
        g.strokeStyle = fast ? "rgba(234,234,230,0.62)" : "rgba(234,234,230,0.34)";
        g.beginPath();
        g.arc(px, py, fast ? 2.4 : 1.5, 0, Math.PI * 2);
        g.stroke();
      }
    }

    // ── recommended sites: hazard red, reserved ──────────────────
    if (showPlan) {
      const active = plan.filter((p) => p.rank <= budget);
      for (const p of active) {
        const [px, py] = toScreen(p.x, p.y, view);
        if (px < -30 || py < -30 || px > size.w + 30 || py > size.h + 30) continue;
        const arm = 6 + Math.min(6, p.ports / 2);
        g.strokeStyle = C.hazard;
        g.lineWidth = 1.25;
        g.beginPath();
        g.moveTo(px - arm, py); g.lineTo(px - 2.5, py);
        g.moveTo(px + 2.5, py); g.lineTo(px + arm, py);
        g.moveTo(px, py - arm); g.lineTo(px, py - 2.5);
        g.moveTo(px, py + 2.5); g.lineTo(px, py + arm);
        g.stroke();
        g.beginPath();
        g.arc(px, py, 2.2, 0, Math.PI * 2);
        g.stroke();
      }
    }

    // ── hover readout ────────────────────────────────────────────
    if (hovered != null && hovered < grid.count) {
      const [rx, ry] = grid.rings[hovered];
      g.beginPath();
      for (let k = 0; k < rx.length; k++) {
        const px = rx[k] * s + tx;
        const py = -ry[k] * s + ty;
        k === 0 ? g.moveTo(px, py) : g.lineTo(px, py);
      }
      g.closePath();
      g.strokeStyle = C.ink;
      g.lineWidth = 1.5;
      g.stroke();
    }
  }, [view, size, grid, values, scale, stations, plan, budget, showStations,
      showPlan, hovered, toScreen]);

  useEffect(() => {
    const id = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(id);
  }, [draw]);

  const pick = useCallback(
    (cx: number, cy: number): number | null => {
      if (!view) return null;
      const wx = (cx - view.tx) / view.scale;
      const wy = -(cy - view.ty) / view.scale;
      const gx = Math.floor(wx / hash.cell);
      const gy = Math.floor(wy / hash.cell);
      let best: number | null = null;
      let bestD = Infinity;
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          const b = hash.m.get(`${gx + dx},${gy + dy}`);
          if (!b) continue;
          for (const i of b) {
            const d = (grid.cx[i] - wx) ** 2 + (grid.cy[i] - wy) ** 2;
            if (d < bestD) { bestD = d; best = i; }
          }
        }
      }
      return bestD < 200 * 200 ? best : null;
    },
    [view, hash, grid],
  );

  return (
    <div ref={wrapRef} className="field">
      <canvas
        ref={canvasRef}
        style={{ width: size.w, height: size.h, cursor: drag.current ? "grabbing" : "crosshair" }}
        onMouseDown={(e) => {
          if (!view) return;
          userMoved.current = true;
          drag.current = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty };
        }}
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          if (drag.current && view) {
            setView({
              ...view,
              tx: drag.current.tx + (e.clientX - drag.current.x),
              ty: drag.current.ty + (e.clientY - drag.current.y),
            });
            return;
          }
          onHover(pick(e.clientX - r.left, e.clientY - r.top));
        }}
        onMouseUp={() => { drag.current = null; }}
        onMouseLeave={() => { drag.current = null; onHover(null); }}
        onClick={(e) => {
          if (!view) return;
          const r = e.currentTarget.getBoundingClientRect();
          const mx = e.clientX - r.left, my = e.clientY - r.top;
          let hit: PlanSite | null = null;
          let bd = 14 * 14;
          for (const p of plan) {
            if (p.rank > budget) continue;
            const [px, py] = toScreen(p.x, p.y, view);
            const d = (px - mx) ** 2 + (py - my) ** 2;
            if (d < bd) { bd = d; hit = p; }
          }
          onPickSite(hit);
        }}
        onWheel={(e) => {
          if (!view) return;
          userMoved.current = true;
          const r = e.currentTarget.getBoundingClientRect();
          const mx = e.clientX - r.left, my = e.clientY - r.top;
          const f = Math.exp(-e.deltaY * 0.0014);
          const ns = Math.max(0.00018, Math.min(0.05, view.scale * f));
          const k = ns / view.scale;
          setView({ scale: ns, tx: mx - (mx - view.tx) * k, ty: my - (my - view.ty) * k });
        }}
      />
      <FieldLegend field={field} values={values} />
    </div>
  );
}

function FieldLegend({ field, values }: {
  field: (typeof LAYERS)[number]; values: number[];
}) {
  const nz = values.filter((v) => v > 0).sort((a, b) => a - b);
  const q = (p: number) => (nz.length ? nz[Math.floor(p * (nz.length - 1))] : 0);
  const f = (v: number) =>
    field.unit === "P" ? v.toFixed(2)
      : v >= 1e6 ? (v / 1e6).toFixed(1) + "M"
      : v >= 1e3 ? (v / 1e3).toFixed(0) + "k"
      : v.toFixed(0);

  return (
    <figure className="legend">
      <figcaption>
        <span className="lg-title">{field.label}</span>
        <span className="mono lg-unit">{field.unit}</span>
      </figcaption>
      <div className="lg-bar" aria-hidden>
        {Array.from({ length: 13 }, (_, i) => (
          <i key={i} style={{ background: ramp(i / 12) }} />
        ))}
      </div>
      <div className="lg-ticks mono">
        <span>{f(q(0))}</span><span>{f(q(0.5))}</span><span>{f(q(0.9))}</span>
        <span>{f(q(1))}</span>
      </div>
      <p className="lg-note">{field.note}</p>
      <p className="lg-note lg-dim">
        Rank-scaled: {nz.length.toLocaleString()} non-zero cells, quantile ramp.
      </p>
    </figure>
  );
}
