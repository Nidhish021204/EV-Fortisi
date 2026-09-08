import { C, SERIES, fmt, ramp } from "../palette";
import type { Metrics, RankMetrics, Explain, ExplainSite } from "../data";

/**
 * Charts follow the data-viz procedure: form chosen by the data's job, single
 * series carries no legend (the title names it), values direct-labelled rather
 * than one number per point, grid and axes recessive, text in ink tokens rather
 * than the mark colour.
 */

/** Ranked comparison of the model against the baselines it has to beat. */
export function BaselineBars({ s }: { s: Metrics["suitability"] }) {
  const rows: { label: string; m: RankMetrics; isModel?: boolean }[] = [
    { label: "PU model", m: s.model_pu, isModel: true },
    { label: "Demand rank", m: s.baseline_demand },
    { label: "Near existing", m: s.baseline_near_existing },
    { label: "Traffic rank", m: s.baseline_traffic },
    { label: "Random", m: s.baseline_random },
  ];
  const max = Math.max(...rows.map((r) => r.m.pr_auc));
  const W = 300, rowH = 26, labelW = 96, valW = 42;
  const barW = W - labelW - valW;

  return (
    <figure className="chart">
      <figcaption>
        <span className="ch-title">Forward-test PR-AUC</span>
        <span className="mono ch-sub">
          {s.n_forward_positives} sites built 2019&ndash;21 / {fmt.int(s.n_forward_candidates)} candidates
        </span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${rows.length * rowH + 8}`} className="ch-svg"
           role="img" aria-label="PR-AUC by ranking method">
        {rows.map((r, i) => {
          const y = i * rowH + 4;
          const w = Math.max(1, (r.m.pr_auc / max) * barW);
          return (
            <g key={r.label}>
              <text x={0} y={y + 11} className="ch-lbl">{r.label}</text>
              <rect x={labelW} y={y + 2} width={barW} height={11} fill={C.surface2} />
              <rect x={labelW} y={y + 2} width={w} height={11}
                    fill={r.isModel ? SERIES[0] : C.ink3} rx={0} />
              <text x={labelW + barW + 6} y={y + 11} className="ch-val mono">
                {r.m.pr_auc.toFixed(3)}
              </text>
              {r.isModel && (
                <text x={labelW + w + 5} y={y + 11} className="ch-tag mono">MODEL</text>
              )}
            </g>
          );
        })}
      </svg>
      <p className="ch-note">
        Higher is better. Base rate {(s.model_pu.base_rate * 100).toFixed(2)}%, so
        random sits near that floor. The model leads the best baseline by{" "}
        {(((s.model_pu.pr_auc - s.baseline_demand.pr_auc) / s.baseline_demand.pr_auc) * 100).toFixed(0)}%.
      </p>
    </figure>
  );
}

/** Precision@k: the number a planner with a budget of k actually experiences. */
export function PrecisionAtK({ s }: { s: Metrics["suitability"] }) {
  const ks = [50, 100, 200, 500];
  const series = [
    { name: "PU model", m: s.model_pu, c: SERIES[0] },
    { name: "Demand rank", m: s.baseline_demand, c: SERIES[1] },
    { name: "Random", m: s.baseline_random, c: C.ink3 },
  ];
  const W = 300, H = 120, padL = 30, padB = 20, padT = 8;
  const max = Math.max(...series.flatMap((x) => ks.map((k) => x.m[`precision@${k}`])));
  const px = (i: number) => padL + (i / (ks.length - 1)) * (W - padL - 10);
  const py = (v: number) => padT + (1 - v / max) * (H - padT - padB);

  return (
    <figure className="chart">
      <figcaption>
        <span className="ch-title">Precision @ k</span>
        <span className="mono ch-sub">share of top-k that were really built</span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="ch-svg" role="img"
           aria-label="Precision at k by ranking method">
        {[0, max / 2, max].map((v, i) => (
          <g key={i}>
            <line x1={padL} x2={W - 10} y1={py(v)} y2={py(v)} stroke={C.hairline} strokeWidth={1} />
            <text x={0} y={py(v) + 3} className="ch-ax mono">{(v * 100).toFixed(0)}%</text>
          </g>
        ))}
        {series.map((sr) => (
          <polyline key={sr.name} fill="none" stroke={sr.c} strokeWidth={2}
                    points={ks.map((k, i) => `${px(i)},${py(sr.m[`precision@${k}`])}`).join(" ")} />
        ))}
        {series.map((sr) =>
          ks.map((k, i) => (
            <circle key={sr.name + k} cx={px(i)} cy={py(sr.m[`precision@${k}`])} r={2.5}
                    fill={sr.c} stroke={C.ground} strokeWidth={1} />
          )),
        )}
        {ks.map((k, i) => (
          <text key={k} x={px(i)} y={H - 6} className="ch-ax mono" textAnchor="middle">{k}</text>
        ))}
      </svg>
      <div className="ch-legend mono">
        {series.map((sr) => (
          <span key={sr.name}><i style={{ background: sr.c }} />{sr.name}</span>
        ))}
      </div>
    </figure>
  );
}

/** Build history: the temporal split that makes the forward test possible. */
export function BuildHistory({ m }: { m: Metrics }) {
  const years = Object.keys(m.build_history).map(Number).sort();
  const vals = years.map((y) => m.build_history[String(y)]);
  const max = Math.max(...vals, 1);
  const W = 300, H = 96, padB = 18, padT = 6;
  const bw = (W - 8) / years.length;
  const cutYear = 2018;

  return (
    <figure className="chart">
      <figcaption>
        <span className="ch-title">Network build history</span>
        <span className="mono ch-sub">stations opened / year, NYC window</span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="ch-svg" role="img"
           aria-label="Stations opened per year">
        {years.map((y, i) => {
          const h = (vals[i] / max) * (H - padT - padB);
          const train = y <= cutYear;
          return (
            <g key={y}>
              <rect x={4 + i * bw} y={H - padB - h} width={bw - 2} height={h}
                    fill={train ? C.ink3 : SERIES[0]} />
              <text x={4 + i * bw + (bw - 2) / 2} y={H - 6}
                    className="ch-ax mono" textAnchor="middle">
                {String(y).slice(2)}
              </text>
            </g>
          );
        })}
        <line x1={4 + (years.indexOf(cutYear) + 1) * bw - 1} x2={4 + (years.indexOf(cutYear) + 1) * bw - 1}
              y1={padT} y2={H - padB} stroke={C.hazard} strokeWidth={1} strokeDasharray="2 2" />
      </svg>
      <div className="ch-legend mono">
        <span><i style={{ background: C.ink3 }} />train &le;2018</span>
        <span><i style={{ background: SERIES[0] }} />forward test</span>
      </div>
    </figure>
  );
}

/** Borough table. Exact values matter and there are five rows, so not a chart. */
export function BoroughTable({ m, budget, planned }: {
  m: Metrics; budget: number; planned: Record<string, number>;
}) {
  const rows = [...m.by_borough].sort((a, b) => a.coverage - b.coverage);
  return (
    <figure className="chart">
      <figcaption>
        <span className="ch-title">Coverage by borough</span>
        <span className="mono ch-sub">at k={budget}</span>
      </figcaption>
      <table className="tbl mono">
        <thead>
          <tr>
            <th>Borough</th><th className="r">Demand</th><th className="r">Cover</th>
            <th className="r">kWh/port</th><th className="r">Plan</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const crit = r.coverage < 5;
            return (
              <tr key={r.borough}>
                <td>{r.borough}</td>
                <td className="r">{fmt.int(r.demand / 1000)}k</td>
                <td className="r" style={{ color: crit ? C.critical : C.ink2 }}>
                  {crit && <span aria-hidden>! </span>}{r.coverage.toFixed(1)}%
                </td>
                <td className="r">{Number.isFinite(r.kwh_per_port) ? fmt.int(r.kwh_per_port) : "—"}</td>
                <td className="r" style={{ color: C.ink }}>{planned[r.borough] ?? 0}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="ch-note">
        <span style={{ color: C.critical }}>!</span> marks coverage under 5% of
        measured demand. kWh/port is demand per installed port: higher means
        more strained.
      </p>
    </figure>
  );
}

/**
 * Cross-domain transfer. The matrix is a magnitude heatmap, so it takes the
 * sequential ramp; the verdict column is a pass/fail state, so it takes the
 * reserved status colours and carries a glyph rather than relying on hue.
 */
export function TransferMatrix({ t }: { t: Metrics["transfer"] }) {
  const doms = Object.keys(t.domains ?? {});
  if (!doms.length) return null;
  const spaces = t.matrices ?? { net: t.matrix };
  const hasBoth = !!(t.matrices?.net && t.matrices?.full);
  const primary = spaces.full ?? spaces.net;
  const vals = doms.flatMap((s) => doms.map((d) => primary?.[`${s}->${d}`]?.pr_auc ?? 0));
  const max = Math.max(...vals, 0.001);

  const sum = (sp: string) =>
    t.verdict_by_space?.[sp]?.["_summary"] as
      { n_transfers: number; n_beating_baseline: number } | undefined;
  const net = sum("net"), full = sum("full");

  return (
    <figure className="chart neg">
      <figcaption>
        <span className="ch-title">Transfer validation</span>
        <span className="mono ch-sub">
          PR-AUC · rows trained on, columns tested on
          {hasBoth && ` · network ${t.feature_counts?.net ?? ""} vs +built form ${t.feature_counts?.full ?? ""} features`}
        </span>
      </figcaption>

      <table className="tbl mono xfer">
        <thead>
          <tr><th />{doms.map((d) => <th key={d} className="r">{d.replace("DE_METRO", "DE·M")}</th>)}</tr>
        </thead>
        <tbody>
          {doms.map((src) => (
            <tr key={src}>
              <td>{src.replace("DE_METRO", "DE·M")}</td>
              {doms.map((tgt) => {
                const v = primary?.[`${src}->${tgt}`]?.pr_auc ?? 0;
                const n = spaces.net?.[`${src}->${tgt}`]?.pr_auc;
                const diag = src === tgt;
                return (
                  <td key={tgt} className="r xcell"
                      style={{ background: ramp(v / max),
                               color: v / max > 0.55 ? C.ground : C.ink,
                               outline: diag ? `1px solid ${C.ink3}` : undefined }}>
                    {v.toFixed(3)}
                    {hasBoth && n != null && (
                      <span className="delta">{v - n >= 0 ? "+" : "−"}{Math.abs(v - n).toFixed(2)}</span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="ch-note">
        Cells show the richer feature space; the small figure is the change from
        network-morphology alone. Outlined diagonal is the within-domain
        reference. Frontier sizes span two orders of magnitude — DE{" "}
        {(t.domains?.DE?.frontier_cells ?? 0).toLocaleString()} cells vs NYC{" "}
        {(t.domains?.NYC?.frontier_cells ?? 0).toLocaleString()} — so read each
        column against its own diagonal, never across rows.
      </p>

      {hasBoth && net && full && (
        <>
          <dl className="readout mono">
            <div className="rd">
              <dt>NETWORK SHAPE ONLY</dt>
              <dd style={{ color: C.ink3 }}>{net.n_beating_baseline} / {net.n_transfers} beat baseline</dd>
            </div>
            <div className="rd">
              <dt>+ BUILT FORM</dt>
              <dd style={{ color: C.good }}>{full.n_beating_baseline} / {full.n_transfers}</dd>
            </div>
          </dl>
          <p className="ch-note">
            Adding a globally-consistent built-form block (GHSL built-up surface
            and population, ~90 m, sampled identically in every country) lifts
            transfers beating their target baseline from{" "}
            {net.n_beating_baseline} to {full.n_beating_baseline} of{" "}
            {full.n_transfers}, and improves every within-domain diagonal. So
            Stage 7a was under-determined: it proved a network's own *shape*
            does not transfer, not that nothing does.
          </p>
          <p className="warn mono">
            <span aria-hidden>▲</span> EVERY TRANSFER INTO NYC STILL FAILS, AND
            DE→NYC GOT WORSE WITH BUILT FORM. NYC HAS THE STRONGEST TRIVIAL
            BASELINE OF ANY DOMAIN, SO NO FOREIGN MODEL BEATS SIMPLY RANKING BY
            DISTANCE TO THE NEAREST EXISTING CHARGER. DO NOT IMPORT A MODEL HERE.
          </p>
          <p className="ch-note">
            GHSL carries built <em>form</em> and population, not land-use
            composition — NYC's own strongest signals (office/retail area,
            garage area, land-use mix) have no global equivalent. This therefore
            bounds "built form transfers", which is weaker than "land use
            transfers".
          </p>
        </>
      )}
    </figure>
  );
}

export function Certification({ c, budget }: {
  c: Metrics["certification"]; budget: number;
}) {
  if (!c || !c.milp_objective_kwh_day) return null;
  const stale = budget !== c.k;
  const max = c.milp_objective_kwh_day;
  const W = 300, rowH = 24, labelW = 62, valW = 62;
  const barW = W - labelW - valW;
  const rows = [
    { label: "MILP", v: c.milp_objective_kwh_day, fill: SERIES[0] },
    { label: "Greedy", v: c.greedy_objective_kwh_day, fill: C.ink3 },
  ];

  return (
    <figure className="chart">
      <figcaption>
        <span className="ch-title">Optimality certificate</span>
        <span className="mono ch-sub">
          kWh/day served at k={c.k} · CBC {c.milp_status.toLowerCase()} in{" "}
          {c.solver.solve_seconds}s
        </span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${rows.length * rowH + 6}`} className="ch-svg"
           role="img" aria-label="Greedy versus MILP objective">
        {rows.map((r, i) => {
          const y = i * rowH + 3;
          return (
            <g key={r.label}>
              <text x={0} y={y + 11} className="ch-lbl">{r.label}</text>
              <rect x={labelW} y={y + 2} width={barW} height={11} fill={C.surface2} />
              <rect x={labelW} y={y + 2} width={Math.max(1, (r.v / max) * barW)}
                    height={11} fill={r.fill} />
              <text x={labelW + barW + 5} y={y + 11} className="ch-val mono">
                {fmt.int(r.v / 1000)}k
              </text>
            </g>
          );
        })}
      </svg>
      <dl className="readout mono">
        <div className="rd">
          <dt>GREEDY / OPTIMUM</dt>
          <dd style={{ color: C.good }}>{(c.greedy_ratio * 100).toFixed(2)}%</dd>
        </div>
        <div className="rd">
          <dt>WORST-CASE BOUND</dt>
          <dd style={{ color: C.ink3 }}>
            {(c.theoretical_worst_case * 100).toFixed(2)}%
          </dd>
        </div>
        <div className="rd">
          <dt>HEADROOM</dt>
          <dd>{fmt.int(c.headroom_kwh_day)} kWh/d · {c.headroom_pct.toFixed(2)}%</dd>
        </div>
      </dl>
      <p className="ch-note">
        The exact model is a capacitated facility-location MILP
        ({fmt.int(c.solver.n_binary)} binaries, {fmt.int(c.solver.n_continuous)}{" "}
        allocation variables, {fmt.int(c.solver.n_spacing_pairs)} spacing pairs) —
        not a coverage MCLP, which cannot express a hub's capacity limit.
        {c.optimality_proven
          ? " CBC proved optimality, so this ratio is exact."
          : " CBC hit its time limit, so the incumbent is a lower bound on the optimum and this ratio is an upper bound on the true one."}
      </p>
      <p className="ch-note">
        Greedy clears its own {(c.theoretical_worst_case * 100).toFixed(0)}%
        guarantee by{" "}
        {((c.greedy_ratio - c.theoretical_worst_case) * 100).toFixed(1)} points.
        The worst case is a theorem about adversarial instances; this instance
        is not adversarial, which is why the live budget slider can keep using
        greedy.
      </p>
      {stale && (
        <p className="warn mono">
          <span aria-hidden>▲</span> CERTIFICATE COMPUTED AT k={c.k}. SLIDER IS
          AT k={budget}, SO THE RATIO ABOVE DOES NOT APPLY TO THE PLAN CURRENTLY
          ON SCREEN.
        </p>
      )}
    </figure>
  );
}

/**
 * Per-site attribution. Answers "why this corner" for one recommended site.
 *
 * Sign is carried by position about the zero line first, by a signed numeral
 * second, and by hue only third. Hazard red is reserved for plan markers and
 * alerts (DESIGN.md), so the usual blue-red diverging pair is unavailable —
 * and since direction already encodes sign unambiguously, it is not needed.
 */
export function SiteDrivers({ site }: { site: ExplainSite | undefined }) {
  if (!site || !site.drivers?.length) return null;
  const max = Math.max(...site.drivers.map((d) => Math.abs(d.shap)), 1e-6);
  const W = 300, rowH = 22, labelW = 150, mid = labelW + (W - labelW - 34) / 2;
  const half = (W - labelW - 34) / 2;

  return (
    <figure className="chart">
      <figcaption>
        <span className="ch-title">Viability attribution</span>
        <span className="mono ch-sub">
          TreeSHAP, log-odds · base {site.base.toFixed(2)} → margin{" "}
          {site.margin.toFixed(2)} · P {site.p_viable.toFixed(3)}
        </span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${site.drivers.length * rowH + 12}`} className="ch-svg"
           role="img" aria-label="Feature contributions for this site">
        <line x1={mid} x2={mid} y1={0} y2={site.drivers.length * rowH + 2}
              stroke={C.hairline} strokeWidth={1} />
        {site.drivers.map((d, i) => {
          const y = i * rowH + 3;
          const w = (Math.abs(d.shap) / max) * half;
          const pos = d.shap >= 0;
          return (
            <g key={d.feature}>
              <text x={0} y={y + 11} className="ch-lbl">{d.label}</text>
              <rect x={pos ? mid : mid - w} y={y + 3} width={Math.max(1, w)}
                    height={10} fill={pos ? SERIES[0] : SERIES[1]} />
              <text x={W - 32} y={y + 11} className="ch-val mono"
                    style={{ fill: pos ? C.ink : C.ink2 }}>
                {pos ? "+" : "−"}{Math.abs(d.shap).toFixed(2)}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="ch-legend mono">
        <span><i style={{ background: SERIES[0] }} />raises viability</span>
        <span><i style={{ background: SERIES[1] }} />lowers</span>
      </div>
      <p className="ch-note">
        Contributions are log-odds and sum to the site's margin; the model's
        probability is sigmoid of that sum. Values shown are the cell's own
        feature readings.
      </p>
      {site.equity_forced ? (
        <p className="note-eq mono">
          SELECTED BY THE EQUITY FLOOR. RE-RUNNING THE OPTIMISER WITHOUT THE
          PER-BOROUGH MINIMUM DROPS THIS SITE, SO IT IS HERE ON ACCESSIBILITY
          GROUNDS RATHER THAN ON DEMAND AND VIABILITY ALONE.
        </p>
      ) : (
        <p className="ch-note">
          Selected on merit: it survives when the optimiser is re-run with the
          equity floor removed.
        </p>
      )}
      <p className="ch-note">
        This attributes the <em>viability score</em>, which is one input to
        selection. The optimiser maximises unmet demand × viability under a
        coverage radius, spacing rule, hub capacity and the borough floor, so a
        low-viability cell can still earn a place by capturing demand nothing
        else reaches.
      </p>
    </figure>
  );
}

/** Global attribution across all cells — what the model relies on overall. */
export function GlobalAttribution({ e }: { e: Explain | null }) {
  if (!e?.global?.length) return null;
  const rows = e.global.slice(0, 10);
  const max = Math.max(...rows.map((r) => r.mean_abs_shap));
  const W = 300, rowH = 20, labelW = 156;
  const barW = W - labelW - 38;

  return (
    <figure className="chart">
      <figcaption>
        <span className="ch-title">Global attribution</span>
        <span className="mono ch-sub">mean |SHAP| across {" "}
          all cells, log-odds</span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${rows.length * rowH + 6}`} className="ch-svg"
           role="img" aria-label="Mean absolute SHAP by feature">
        {rows.map((r, i) => {
          const y = i * rowH + 3;
          return (
            <g key={r.feature}>
              <text x={0} y={y + 10} className="ch-lbl">{r.label}</text>
              <rect x={labelW} y={y + 2} width={barW} height={9} fill={C.surface2} />
              <rect x={labelW} y={y + 2}
                    width={Math.max(1, (r.mean_abs_shap / max) * barW)}
                    height={9} fill={SERIES[0]} />
              <text x={W - 34} y={y + 10} className="ch-val mono">
                {r.mean_abs_shap.toFixed(2)}
              </text>
            </g>
          );
        })}
      </svg>
      {/* Named from the data. This read "Parking floor area leads" while the
          chart showed assessed value density ahead of it — a caption can go
          stale against its own figure, so it now reads the ranking. */}
      <p className="ch-note">
        {rows[0]?.label} leads, with {rows[1]?.label.toLowerCase()} behind it.
        Both arrived with the PLUTO land-use block in Stage 3, and together they
        say the physically right thing: charging hardware goes into parking
        structures on land worth building on, so a cell with neither is hard to
        build in however much demand passes through it.
      </p>
      <p className="ch-note">
        Attributions explain the mean log-odds, while the ranked score is the
        mean probability. Sigmoid is not affine, so these differ — measured at
        Spearman {e.faithfulness.spearman_vs_ensemble.toFixed(4)}, mean |Δp|{" "}
        {e.faithfulness.mean_abs_prob_error.toFixed(4)}, max |Δp|{" "}
        {e.faithfulness.max_abs_prob_error.toFixed(4)}. Ordering is preserved,
        which is what an explanation of a ranking needs.
      </p>
    </figure>
  );
}

/**
 * Grid capacity. Con Edison publishes, per substation bank, the rating at the
 * feeder head minus last year's peak. The interesting result here is negative,
 * and the coverage caveat is what bounds it — so both are given equal weight
 * rather than the headline being shown alone.
 */
export function GridPanel({ f, cov }: {
  f: Metrics["feeder_impact"]; cov: Metrics["grid_coverage"];
}) {
  if (!f || !cov?.by_borough?.length) return null;
  const curve = f.sweep?.curve ?? [];
  const maxCost = Math.max(...curve.map((c) => c.cost_pct), 0.5);
  const W = 300, H = 92, padL = 30, padB = 18, padT = 6;
  const px = (i: number) => padL + (i / Math.max(1, curve.length - 1)) * (W - padL - 12);
  const py = (v: number) => padT + (1 - v / maxCost) * (H - padT - padB);

  return (
    <figure className="chart">
      <figcaption>
        <span className="ch-title">Grid capacity</span>
        <span className="mono ch-sub">
          Con Edison per-feeder charging limit ·{" "}
          {cov.feeders != null ? `${fmt.int(cov.feeders)} feeders · ` : ""}
          {cov.feeder_capacity_total_mw != null
            ? `${fmt.int(cov.feeder_capacity_total_mw)} MW`
            : `${fmt.int(cov.headroom_total_mw)} MW`}
          {cov.feeder_capacity_total_mw != null
            ? ` vs ${fmt.int(cov.headroom_total_mw)} MW of bank headroom`
            : ""}
        </span>
      </figcaption>

      <dl className="readout mono">
        <div className="rd">
          <dt>COST AT k=120</dt>
          <dd style={{ color: C.good }}>{f.cost_pct.toFixed(2)}%</dd>
        </div>
        <div className="rd">
          <dt>SITES DISPLACED</dt><dd>{f.sites_displaced}</dd>
        </div>
        <div className="rd">
          <dt>PORT-LIMITED</dt><dd>{f.sites_grid_limited}</dd>
        </div>
        {f.sweep && (
          <div className="rd">
            <dt>POOL EXHAUSTS AT</dt>
            <dd>{f.sweep.sites_before_grid_exhausts} vs{" "}
              {f.sweep.sites_without_grid_limit}</dd>
          </div>
        )}
      </dl>

      {curve.length > 0 && (
        <>
          <svg viewBox={`0 0 ${W} ${H}`} className="ch-svg" role="img"
               aria-label="Cost of the grid constraint by budget">
            {[0, maxCost / 2, maxCost].map((v, i) => (
              <g key={i}>
                <line x1={padL} x2={W - 12} y1={py(v)} y2={py(v)}
                      stroke={C.hairline} strokeWidth={1} />
                <text x={0} y={py(v) + 3} className="ch-ax mono">
                  {v.toFixed(2)}%
                </text>
              </g>
            ))}
            <polyline fill="none" stroke={SERIES[0]} strokeWidth={2}
                      points={curve.map((c, i) => `${px(i)},${py(c.cost_pct)}`).join(" ")} />
            {curve.map((c, i) => (
              <circle key={c.k} cx={px(i)} cy={py(c.cost_pct)} r={2.5}
                      fill={SERIES[0]} stroke={C.ground} strokeWidth={1} />
            ))}
            {curve.map((c, i) => (
              <text key={"t" + c.k} x={px(i)} y={H - 5} className="ch-ax mono"
                    textAnchor="middle">{c.k}</text>
            ))}
          </svg>
          {/* Read from the data rather than asserted: the grid's cost moved
              once the constraint switched from bank headroom to Con Edison's
              per-feeder charging limit, so the conclusion has to follow the
              number instead of outliving it. */}
          <p className="ch-note">
            Energy given up to the grid limit, by programme size k. It
            plateaus near {curve[curve.length - 1].cost_pct.toFixed(2)}%
            {f.cost_pct < 1
              ? ": where Con Edison publishes feeder capacity, it is not what limits this programme — demand geography and the 800 m coverage radius bind first."
              : `: the feeder limit is a real constraint here, displacing ${f.sites_displaced} site${f.sites_displaced === 1 ? "" : "s"} at k=120 and shrinking ${f.sites_grid_limited} more.`}
          </p>
        </>
      )}

      <table className="tbl mono">
        <thead>
          <tr><th>Borough</th><th className="r">Cells</th>
            <th className="r">Feeder</th><th className="r">Network</th>
            <th className="r">No data</th></tr>
        </thead>
        <tbody>
          {[...cov.by_borough].sort((a, b) => a.pct - b.pct).map((r) => (
            <tr key={r.borough}>
              <td>{r.borough}</td>
              <td className="r dim">{fmt.int(r.cells)}</td>
              <td className="r" style={{ color: r.pct === 0 ? C.critical : C.ink2 }}>
                {fmt.int(r.feeder)}
              </td>
              <td className="r dim">{fmt.int(r.network_area)}</td>
              <td className="r dim">{fmt.int(r.no_data)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note-eq mono">
        LAYER 0 COVERS 3-PHASE RADIAL FEEDERS. MANHATTAN RUNS ON THE SECONDARY
        NETWORK, WHOSE PUBLISHED LAYERS CARRY <em>GENERATION</em> HOSTING
        CAPACITY, NOT CHARGING LOAD HEADROOM — SO THEY ARE NOT SUBSTITUTED IN.
        CELLS OVER 1 KM FROM A PUBLISHED FEEDER ARE LEFT UNCONSTRAINED RATHER
        THAN CHARGED TO A DISTANT FEEDER.
      </p>
      {/* Weighted by built floor area, because cell counts are diluted by
          river and park cells that have no grid because they have nothing
          on them. By area the covered share is markedly smaller. */}
      <p className="ch-note">
        The result holds only where the data reaches:{" "}
        {fmt.int(cov.cells_covered)} of {fmt.int(cov.cells_total)} cells
        {cov.built_share
          ? `, and only ${(100 * cov.built_share.feeder).toFixed(1)}% of the city's built floor area —
             ${(100 * cov.built_share.network_area).toFixed(1)}% sits on the secondary network,
             leaving ${(100 * cov.built_share.no_data).toFixed(1)}% genuinely unaccounted`
          : ""}
        . Absence of published capacity is not evidence of available capacity.
      </p>
    </figure>
  );
}

/**
 * Uncertainty. Two distinct questions, kept visually distinct: which sites are
 * robust (bag bootstrap), and how precise the headline metrics really are
 * (seed sweep). A single "confidence" number would conflate them.
 */
export function UncertaintyPanel({ u }: { u: Metrics["uncertainty"] }) {
  if (!u?.plan_stability) return null;
  const p = u.plan_stability;
  const rows: [string, string][] = [
    ["pr_auc", "PR-AUC"], ["precision@100", "P@100"],
    ["lift@100", "Lift@100"], ["ndcg", "NDCG"],
  ];

  return (
    <figure className="chart">
      <figcaption>
        <span className="ch-title">Uncertainty</span>
        <span className="mono ch-sub">
          {u.n_bootstrap} bag resamples · {u.n_seeds} seed refits
        </span>
      </figcaption>

      <table className="tbl mono">
        <thead>
          <tr><th>Metric</th><th className="r">Mean</th>
            <th className="r">95% interval</th><th className="r">dp</th></tr>
        </thead>
        <tbody>
          {rows.map(([k, label]) => {
            const m = u.metric_intervals?.[k];
            if (!m) return null;
            const dp = Math.max(1, m.supported_decimals);
            return (
              <tr key={k}>
                <td>{label}</td>
                <td className="r">{m.mean.toFixed(dp)}</td>
                <td className="r dim">
                  {m.lo95.toFixed(dp)}–{m.hi95.toFixed(dp)}
                </td>
                <td className="r">{m.supported_decimals}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="ch-note">
        Refitting under {u.n_seeds} seeds measures training-time
        nondeterminism, which a bootstrap over one fitted ensemble cannot see.
        The final column is how many decimals the spread actually supports —
        lift is quoted to{" "}
        {u.metric_intervals?.["lift@100"]?.supported_decimals ?? 1}, so
        "14.83×" was false precision.
      </p>

      <dl className="readout mono">
        <div className="rd">
          <dt>SITES ALWAYS CHOSEN</dt>
          <dd style={{ color: C.good }}>{p.sites_always_selected} / 120</dd>
        </div>
        <div className="rd">
          <dt>COIN-FLIP SITES</dt>
          <dd style={{ color: p.sites_below_half > 0 ? C.warning : C.ink }}>
            {p.sites_below_half} / 120
          </dd>
        </div>
        <div className="rd">
          <dt>MEDIAN STABILITY</dt><dd>{p.median.toFixed(2)}</dd>
        </div>
        <div className="rd">
          <dt>CAPTURED 95%</dt>
          <dd>{fmt.int(u.captured_kwh_day.lo95 / 1000)}–
            {fmt.int(u.captured_kwh_day.hi95 / 1000)}k kWh/d</dd>
        </div>
      </dl>
      <p className="ch-note">
        Stability is the share of {u.n_bootstrap} resampled ensembles that put a
        site in the plan. It separates the corners the model insists on from the
        ones that are a coin flip — a distinction the ranked list alone hides.
        Bootstrap bias against the point estimate: {u.bootstrap_bias_max_abs.toExponential(1)}.
      </p>
    </figure>
  );
}
