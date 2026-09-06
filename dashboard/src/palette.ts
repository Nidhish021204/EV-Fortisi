// Design tokens. Mirrors DESIGN.md; validated with the dataviz validator
// against surface #0d0d0c (lightness band, chroma floor, CVD separation,
// normal-vision floor and contrast all PASS).

export const C = {
  ground: "#0d0d0c",
  surface1: "#161614",
  surface2: "#1e1e1b",
  hairline: "#2a2a26",
  ink: "#eaeae6",
  ink2: "#a3a29a",
  ink3: "#6b6a63",
  hazard: "#e61919",
  good: "#0ca30c",
  warning: "#fab219",
  critical: "#d03b3b",
} as const;

// Single-hue sequential ramp, blue 700 -> 100. On this dark substrate the
// lightest step is the HIGHEST magnitude and near-zero recedes into the ground.
export const RAMP = [
  "#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6",
  "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb",
] as const;

// Fixed categorical order for supporting charts. Never cycled.
export const SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"] as const;

/** Perceptual ramp lookup on [0,1]. */
export function ramp(t: number): string {
  if (!Number.isFinite(t)) return RAMP[0];
  const c = Math.max(0, Math.min(1, t));
  return RAMP[Math.min(RAMP.length - 1, Math.round(c * (RAMP.length - 1)))];
}

/**
 * Demand is heavy-tailed by four orders of magnitude, so a linear ramp paints
 * the whole city as the darkest step and Midtown as the lightest. Rank
 * (quantile) scaling spends the ramp where the cells actually are.
 */
export function makeRankScale(values: number[]): (v: number) => number {
  const sorted = Float64Array.from(values.filter((v) => v > 0)).sort();
  if (sorted.length === 0) return () => 0;
  return (v: number) => {
    if (!(v > 0)) return 0;
    let lo = 0, hi = sorted.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (sorted[mid] < v) lo = mid + 1;
      else hi = mid;
    }
    return lo / sorted.length;
  };
}

export const fmt = {
  int: (n: number) => Math.round(n).toLocaleString("en-US"),
  kwh: (n: number) =>
    n >= 1e6 ? (n / 1e6).toFixed(2) + " GWh" :
    n >= 1e3 ? (n / 1e3).toFixed(1) + " MWh" : n.toFixed(0) + " kWh",
  usd: (n: number) =>
    n >= 1e6 ? "$" + (n / 1e6).toFixed(1) + "M" : "$" + Math.round(n / 1e3) + "k",
  pct: (n: number, d = 1) => n.toFixed(d) + "%",
  bytes: (n: number) => (n / 1e9).toFixed(1) + " GB",
};
