// Payload contract with src/evopt/export.py. Geometry arrives as true H3 cell
// boundaries already projected to Web Mercator and quantised to integer metres
// about a local origin, so nothing here needs an H3 library or a projection.

export interface Grid {
  origin: [number, number];
  count: number;
  cx: number[];
  cy: number[];
  rings: [number[], number[]][];
  h3: string[];
  borough: string[];
  demand: number[];
  served: number[];
  unmet: number[];
  p_viable: number[];
  aadt: number[];
  ports: number[];
  dwell: number[];
  built1921: number[];
}

export interface Stations {
  origin: [number, number];
  x: number[];
  y: number[];
  lat: number[];
  lon: number[];
  name: string[];
  city: string[];
  network: string[];
  ports: number[];
  dcfc: number[];
  kw: number[];
  year: number[];
}

export interface PlanSite {
  rank: number;
  h3: string;
  lat: number;
  lon: number;
  x: number;
  y: number;
  borough: string;
  captured_kwh_day: number;
  demand_kwh_day: number;
  p_viable: number;
  cells_covered: number;
  saturated: boolean;
  substation: string | null;
  headroom_kw_before: number | null;
  grid_limited: boolean;
  stability?: number;
  rank_median_boot?: number | null;
  capex_usd: number;
  ports: number;
  rho: number;
  p_wait: number;
  wait_min: number;
  sessions_day: number;
}

export interface Interval {
  mean: number; sd: number; lo95: number; hi95: number;
  n: number; supported_decimals: number;
}

export interface RankMetrics {
  pr_auc: number;
  roc_auc: number;
  base_rate: number;
  ndcg: number;
  [k: string]: number;
}

export interface Metrics {
  generated: string;
  cutoff: string;
  corpus: {
    taxi_trips: number;
    taxi_dropoffs_binned: number;
    traffic_segments: number;
    stations_nyc: number;
    stations_ports: number;
    stations_mw: number;
    hex_cells: number;
    raw_bytes: number;
  };
  demand: {
    spatial_block_cv: Record<string, number>;
    random_kfold_cv: Record<string, number>;
    ablation_no_landuse: Record<string, number>;
    leakage_inflation_r2: number;
    oracle_top_decile_capture: number;
    baseline_aadt_top_decile_capture: number;
    landuse_capture_gain: number;
    n_features: number;
    feature_importance: [string, number][];
  };
  suitability: {
    label_frequency_c: number;
    n_train_positives: number;
    n_forward_candidates: number;
    n_forward_positives: number;
    model_pu: RankMetrics;
    baseline_random: RankMetrics;
    baseline_demand: RankMetrics;
    baseline_traffic: RankMetrics;
    baseline_near_existing: RankMetrics;
    feature_importance: [string, number][];
  };
  transfer: {
    domains: Record<string, {
      frontier_cells: number; positives: number; base_rate: number;
      cutoff: string; forward_window: string[];
    }>;
    matrix: Record<string, {
      pr_auc: number; ndcg: number; "precision@100": number;
      "lift@100": number; base_rate: number;
    }>;
    matrices?: Record<string, Record<string, {
      pr_auc: number; ndcg: number; "precision@100": number;
      "lift@100": number; base_rate: number;
    }>>;
    feature_counts?: Record<string, number>;
    verdict_by_space?: Record<string, Record<string, {
      model_pr_auc: number; baseline: string; baseline_pr_auc: number;
      ratio: number; beats_baseline: boolean;
    } | { n_transfers: number; n_beating_baseline: number }>>;
    baselines: Record<string, Record<string, number>>;
    retention_vs_local: Record<string, number>;
    verdict: Record<string, {
      model_pr_auc: number; baseline: string; baseline_pr_auc: number;
      ratio: number; beats_baseline: boolean;
    } | { n_transfers: number; n_beating_baseline: number }>;
  };
  certification: {
    k: number;
    greedy_objective_kwh_day: number;
    milp_objective_kwh_day: number;
    milp_status: string;
    optimality_proven: boolean;
    greedy_ratio: number;
    ratio_is: string;
    theoretical_worst_case: number;
    headroom_kwh_day: number;
    headroom_pct: number;
    solver: { solve_seconds: number; n_binary: number; n_continuous: number;
              n_spacing_pairs: number; time_limit_s: number; n_sites: number };
  };
  feeder_impact: {
    constrained_kwh_day: number; unconstrained_kwh_day: number;
    cost_kwh_day: number; cost_pct: number;
    constrained_ports: number; unconstrained_ports: number;
    sites_displaced: number; sites_grid_limited: number;
    substations_used: number;
    sweep?: {
      k_max: number; sites_before_grid_exhausts: number;
      sites_without_grid_limit: number;
      curve: { k: number; constrained_kwh_day: number;
               unconstrained_kwh_day: number; cost_pct: number;
               port_limited: number }[];
    };
  };
  grid_coverage: {
    by_borough: { borough: string; cells: number; covered: number; pct: number }[];
    cells_covered: number; cells_total: number;
    substations: number; headroom_total_mw: number;
  };
  uncertainty: {
    n_bootstrap: number; n_seeds: number; bootstrap_bias_max_abs: number;
    captured_kwh_day: Interval;
    metric_intervals: Record<string, Interval>;
    plan_stability: {
      mean: number; median: number;
      sites_always_selected: number; sites_below_half: number;
      deciles: Record<string, number>;
    };
  };
  allocation: {
    k: number;
    candidate_pool: number;
    sites_selected: number;
    captured_kwh_day: number;
    total_unmet_kwh_day: number;
    coverage_gain_pct: number;
    total_ports: number;
    total_capex_usd: number;
    mean_wait_min: number;
    p95_wait_min: number;
    max_utilisation: number;
    sites_at_capacity: number;
    equity_floor: Record<string, number>;
    by_borough: Record<string, number>;
  };
  by_borough: {
    borough: string;
    hexes: number;
    demand: number;
    served: number;
    unmet: number;
    ports: number;
    coverage: number;
    kwh_per_port: number;
    planned: number;
  }[];
  network_mix: Record<string, number>;
  build_history: Record<string, number>;
}

export interface ExplainDriver {
  feature: string; label: string; shap: number; value: number;
}

export interface ExplainSite {
  rank: number; h3: string; margin: number; base: number;
  p_viable: number; equity_forced: boolean; drivers: ExplainDriver[];
}

export interface Explain {
  space: string;
  global: { feature: string; label: string; mean_abs_shap: number;
            mean_signed_shap: number }[];
  faithfulness: {
    spearman_vs_ensemble: number; pearson_vs_ensemble: number;
    mean_abs_prob_error: number; max_abs_prob_error: number;
  };
  equity_forced_sites: number;
  sites: ExplainSite[];
}

export interface Bundle {
  grid: Grid;
  stations: Stations;
  plan: PlanSite[];
  metrics: Metrics;
  explain: Explain | null;
}

export async function loadBundle(): Promise<Bundle> {
  const get = async <T,>(p: string): Promise<T> => {
    const r = await fetch(p);
    if (!r.ok) throw new Error(`${p} -> HTTP ${r.status}`);
    return r.json() as Promise<T>;
  };
  const [grid, stations, plan, metrics] = await Promise.all([
    get<Grid>("/data/grid.json"),
    get<Stations>("/data/stations.json"),
    get<PlanSite[]>("/data/plan.json"),
    get<Metrics>("/data/metrics.json"),
  ]);
  // Attribution is optional: the dashboard stays usable if Stage 7c has not
  // been run, rather than failing to load.
  const explain = await get<Explain>("/data/explain.json").catch(() => null);
  return { grid, stations, plan, metrics, explain };
}

export type LayerId = "demand" | "unmet" | "p_viable" | "aadt" | "dwell";

export const LAYERS: {
  id: LayerId;
  label: string;
  unit: string;
  note: string;
  field: keyof Grid;
}[] = [
  { id: "demand", label: "Demand", unit: "kWh/day", field: "demand",
    note: "Measured: terminating taxi VMT + fleet throughput x cell traversal." },
  { id: "unmet", label: "Unmet", unit: "kWh/day", field: "unmet",
    note: "Demand minus gravity-allocated capacity of the 2018 network." },
  { id: "p_viable", label: "Viability", unit: "P", field: "p_viable",
    note: "Bagged-PU siting probability, Elkan-Noto calibrated." },
  { id: "aadt", label: "Throughput", unit: "veh/day", field: "aadt",
    note: "Annual average daily traffic from 14,359 counted segments." },
  { id: "dwell", label: "Dwell", unit: "veh", field: "dwell",
    note: "Dropoff surplus: where vehicles rest rather than pass through." },
];
