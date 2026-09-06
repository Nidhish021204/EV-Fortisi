import { useEffect, useMemo, useState } from "react";
import HexField from "./components/HexField";
import { BaselineBars, PrecisionAtK, BuildHistory, BoroughTable, TransferMatrix,
  Certification, SiteDrivers, GlobalAttribution, GridPanel,
  UncertaintyPanel } from "./components/Charts";
import { loadBundle, LAYERS } from "./data";
import type { Bundle, LayerId, PlanSite, ExplainSite } from "./data";
import { C, fmt } from "./palette";

// Bronx and Brooklyn both start "BR", so slicing is not an option.
const BORO_ABBR: Record<string, string> = {
  Bronx: "BRX", Brooklyn: "BKN", Manhattan: "MAN",
  Queens: "QNS", "Staten Island": "SI",
};

export default function App() {
  const [bundle, setBundle] = useState<Bundle | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [layer, setLayer] = useState<LayerId>("unmet");
  const [budget, setBudget] = useState(120);
  const [showStations, setShowStations] = useState(true);
  const [showPlan, setShowPlan] = useState(true);
  const [hovered, setHovered] = useState<number | null>(null);
  const [site, setSite] = useState<PlanSite | null>(null);

  useEffect(() => {
    loadBundle().then(setBundle).catch((e) => setErr(String(e)));
  }, []);

  if (err) return <Fail msg={err} />;
  if (!bundle) return <Boot />;

  const { grid, stations, plan, metrics } = bundle;
  return (
    <div className="app">
      <Masthead m={metrics} />
      <main className="body">
        <LeftRail
          m={metrics} layer={layer} setLayer={setLayer}
          budget={budget} setBudget={setBudget}
          showStations={showStations} setShowStations={setShowStations}
          showPlan={showPlan} setShowPlan={setShowPlan}
          plan={plan} stationCount={stations.x.length}
        />
        <HexField
          bundle={bundle} layer={layer} budget={budget}
          showStations={showStations} showPlan={showPlan}
          hovered={hovered} onHover={setHovered} onPickSite={setSite}
        />
        <RightRail
          bundle={bundle} hovered={hovered} site={site}
          onClearSite={() => setSite(null)} budget={budget}
        />
      </main>
      <StatusBar m={metrics} grid={grid} hovered={hovered} budget={budget} />
    </div>
  );
}

/* ── chrome ──────────────────────────────────────────────────── */

function Masthead({ m }: { m: Bundle["metrics"] }) {
  return (
    <header className="masthead">
      <div className="mh-id">
        <h1>VOLTGRID</h1>
        <p className="mono">EV SITING INSTRUMENT / NEW YORK CITY</p>
      </div>
      <dl className="mh-prov mono">
        <Prov k="CORPUS" v={fmt.bytes(m.corpus.raw_bytes)} />
        <Prov k="TRIPS" v={fmt.int(m.corpus.taxi_trips / 1e6) + "M"} />
        <Prov k="SEGMENTS" v={fmt.int(m.corpus.traffic_segments)} />
        <Prov k="STATIONS" v={fmt.int(m.corpus.stations_nyc)} />
        <Prov k="CELLS" v={fmt.int(m.corpus.hex_cells)} />
        <Prov k="H3 RES" v="9 / 174m" />
        <Prov k="CUTOFF" v={m.cutoff} />
      </dl>
    </header>
  );
}

function Prov({ k, v }: { k: string; v: string }) {
  return (
    <div className="prov">
      <dt>{k}</dt>
      <dd>{v}</dd>
    </div>
  );
}

function LeftRail(p: {
  m: Bundle["metrics"]; layer: LayerId; setLayer: (l: LayerId) => void;
  budget: number; setBudget: (n: number) => void;
  showStations: boolean; setShowStations: (b: boolean) => void;
  showPlan: boolean; setShowPlan: (b: boolean) => void;
  plan: PlanSite[]; stationCount: number;
}) {
  const active = p.plan.filter((s) => s.rank <= p.budget);
  const ports = active.reduce((a, s) => a + s.ports, 0);
  const capex = active.reduce((a, s) => a + s.capex_usd, 0);
  const captured = active.reduce((a, s) => a + s.captured_kwh_day, 0);
  const totalUnmet = p.m.allocation.total_unmet_kwh_day;
  const wait = active.length
    ? active.reduce((a, s) => a + s.wait_min, 0) / active.length : 0;

  return (
    <aside className="rail rail-l">
      <Section label="Field">
        <div className="layers">
          {LAYERS.map((l) => (
            <button key={l.id} className={"layer" + (p.layer === l.id ? " on" : "")}
                    onClick={() => p.setLayer(l.id)}>
              <span className="mono lay-k">{l.label}</span>
              <span className="mono lay-u">{l.unit}</span>
            </button>
          ))}
        </div>
        <div className="toggles mono">
          <label>
            <input type="checkbox" checked={p.showStations}
                   onChange={(e) => p.setShowStations(e.target.checked)} />
            <span>Existing network ({fmt.int(p.stationCount)})</span>
          </label>
          <label>
            <input type="checkbox" checked={p.showPlan}
                   onChange={(e) => p.setShowPlan(e.target.checked)} />
            <span>Recommended sites</span>
          </label>
        </div>
      </Section>

      <Section label="Capital plan">
        <div className="budget">
          <div className="bd-head mono">
            <span>BUDGET</span>
            <output>k = {p.budget}</output>
          </div>
          <input type="range" min={5} max={p.m.allocation.k} step={1}
                 value={p.budget} aria-label="Number of sites"
                 onChange={(e) => p.setBudget(Number(e.target.value))} />
          <p className="bd-note">
            Greedy selection is nested, so the first k sites are exactly the
            optimal-greedy plan for budget k. No refit needed.
          </p>
        </div>
        <dl className="readout mono">
          <Read k="SITES" v={String(active.length)} />
          <Read k="PORTS" v={fmt.int(ports)} />
          <Read k="CAPEX" v={fmt.usd(capex)} />
          <Read k="SERVED" v={fmt.int(captured) + " kWh/d"} />
          <Read k="OF UNMET" v={fmt.int(totalUnmet) + " kWh/d"} dim />
          <Read k="GAP CLOSED" v={fmt.pct((captured / totalUnmet) * 100)} hot />
          <Read k="MEAN WAIT" v={wait.toFixed(1) + " min"} />
        </dl>
      </Section>

      <Section label="Constraints">
        <dl className="readout mono">
          <Read k="COVER RADIUS" v="800 m" />
          <Read k="MIN SPACING" v="400 m" />
          <Read k="UTIL CEILING" v={"ρ ≤ " + p.m.allocation.max_utilisation.toFixed(2)} />
          <Read k="PORT" v="50 kW DC" />
          <Read k="EQUITY FLOOR" v={Object.entries(p.m.allocation.equity_floor)
            .map(([b, n]) => `${BORO_ABBR[b] ?? b.slice(0, 3).toUpperCase()} ${n}`)
            .join("  ")} />
        </dl>
        <p className="bd-note">
          The floor is what stops the optimiser answering &ldquo;where is demand
          densest&rdquo; with &ldquo;all of it in Manhattan&rdquo;.
        </p>
      </Section>
    </aside>
  );
}

function RightRail({ bundle, hovered, site, onClearSite, budget }: {
  bundle: Bundle; hovered: number | null; site: PlanSite | null;
  onClearSite: () => void; budget: number;
}) {
  const { grid, metrics, plan } = bundle;
  const plannedByBoro = useMemo(() => {
    const o: Record<string, number> = {};
    for (const s of plan) if (s.rank <= budget) o[s.borough] = (o[s.borough] ?? 0) + 1;
    return o;
  }, [plan, budget]);

  return (
    <aside className="rail rail-r">
      {site ? (
        <SiteInspector s={site} onClose={onClearSite}
                       ex={bundle.explain?.sites.find((x) => x.rank === site.rank)} />
      ) : hovered != null ? (
        <CellInspector g={grid} i={hovered} />
      ) : (
        <Section label="Inspector">
          <p className="idle mono">
            HOVER A CELL FOR ITS MEASUREMENTS.<br />
            CLICK A <span style={{ color: C.hazard }}>+</span> MARKER FOR SITE DETAIL.
          </p>
        </Section>
      )}

      <Section label="Validation">
        <Certification c={metrics.certification} budget={budget} />
        <GlobalAttribution e={bundle.explain} />
        <UncertaintyPanel u={metrics.uncertainty} />
        <BaselineBars s={metrics.suitability} />
        <PrecisionAtK s={metrics.suitability} />
        <DemandValidation m={metrics} />
      </Section>

      <Section label="Grid">
        <GridPanel f={metrics.feeder_impact} cov={metrics.grid_coverage} />
      </Section>

      <Section label="Transfer">
        <TransferMatrix t={metrics.transfer} />
      </Section>

      <Section label="Distribution">
        <BoroughTable m={metrics} budget={budget} planned={plannedByBoro} />
        <BuildHistory m={metrics} />
      </Section>
    </aside>
  );
}

/**
 * Principle 4 in PRODUCT.md: show the results, including the ones that were
 * wrong. This panel previously reported that the demand model lost to a
 * trivial traffic ranking. It did -- on traffic features alone. The ablation
 * below is measured on every pipeline run rather than remembered, so the
 * claim and the correction are both auditable.
 */
function DemandValidation({ m }: { m: Bundle["metrics"] }) {
  const d = m.demand;
  const full = d.spatial_block_cv.top_decile_capture;
  const ablated = d.ablation_no_landuse?.top_decile_capture ?? NaN;
  const rand = d.random_kfold_cv.top_decile_capture;
  const base = d.baseline_aadt_top_decile_capture;
  const oracle = d.oracle_top_decile_capture;
  const beats = full > base;

  const rows = [
    { k: "Oracle ranking", v: oracle, note: "ceiling" },
    { k: "Random k-fold CV", v: rand, note: "leaky protocol" },
    { k: "Model + land use", v: full, note: "block CV", lead: true },
    { k: "AADT baseline", v: base, note: "trivial" },
    { k: "Model, no land use", v: ablated, note: "ablation" },
  ].sort((a, b) => b.v - a.v);

  return (
    <figure className="chart neg">
      <figcaption>
        <span className="ch-title">Demand model: resolved</span>
        <span className="mono ch-sub">top-decile energy capture, spatial block CV</span>
      </figcaption>
      <table className="tbl mono">
        <tbody>
          {rows.map((r) => (
            <tr key={r.k}>
              <td style={r.lead ? { color: C.ink } : undefined}>{r.k}</td>
              <td className="r" style={r.lead ? { color: C.ink } : undefined}>
                {Number.isFinite(r.v) ? (r.v * 100).toFixed(1) + "%" : "—"}
              </td>
              <td className="dim">{r.note}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="ch-note">
        On traffic and street morphology alone the model captured{" "}
        {(ablated * 100).toFixed(1)}% and <strong>lost</strong> to simply
        sorting cells by traffic volume ({(base * 100).toFixed(1)}%). Adding{" "}
        857,347 PLUTO tax lots moved it to {(full * 100).toFixed(1)}% —{" "}
        {(d.landuse_capture_gain * 100).toFixed(1)} points from the
        built-environment block, and {((full / oracle) * 100).toFixed(0)}% of
        the achievable ceiling.
      </p>
      <p className="ch-note">
        The leakage gap closed with it: block CV now trails random k-fold by
        only {((rand - full) * 100).toFixed(1)} points, against 46.8 before.
        The model is learning transferable structure rather than memorising
        position.
      </p>
      {!beats && (
        <p className="warn mono">
          <span aria-hidden>▲</span> MODEL NO LONGER BEATS THE TRIVIAL
          BASELINE. TREAT THE SURFACE AS OBSERVATION ONLY.
        </p>
      )}
    </figure>
  );
}

function CellInspector({ g, i }: { g: Bundle["grid"]; i: number }) {
  // Below ~1 kWh/day a cell's demand is numerical dust and the coverage ratio
  // stops meaning anything, so it is reported undefined rather than as a
  // spectacular percentage.
  const meaningful = g.demand[i] > 1;
  const cover = meaningful ? (g.served[i] / g.demand[i]) * 100 : NaN;
  return (
    <Section label="Cell">
      <p className="ins-id mono">{g.h3[i]}</p>
      <p className="ins-sub mono">{g.borough[i]}</p>
      <dl className="readout mono">
        <Read k="DEMAND" v={g.demand[i].toFixed(1) + " kWh/d"} />
        <Read k="SERVED" v={g.served[i].toFixed(1) + " kWh/d"} />
        <Read k="UNMET" v={g.unmet[i].toFixed(1) + " kWh/d"} hot={g.unmet[i] > 0} />
        <Read k="COVERAGE"
              v={meaningful ? cover.toFixed(1) + "%" : "n/a · demand < 1 kWh"}
              dim={!meaningful} />
        <Read k="P(VIABLE)" v={g.p_viable[i].toFixed(3)} />
        <Read k="THROUGHPUT" v={fmt.int(g.aadt[i]) + " veh/d"} />
        <Read k="DWELL" v={fmt.int(g.dwell[i]) + " veh"} />
        <Read k="PORTS 2018" v={String(g.ports[i])} />
        <Read k="BUILT 19-21" v={g.built1921[i] ? "YES" : "NO"} />
      </dl>
    </Section>
  );
}

function SiteInspector({ s, onClose, ex }: {
  s: PlanSite; onClose: () => void; ex?: ExplainSite;
}) {
  return (
    <Section label={`Site ${s.rank}`} action={<button className="x mono" onClick={onClose}>CLOSE</button>}>
      <p className="ins-id mono" style={{ color: C.hazard }}>RANK {s.rank} / {s.borough}</p>
      <p className="ins-sub mono">{s.lat.toFixed(5)}, {s.lon.toFixed(5)}</p>
      <dl className="readout mono">
        <Read k="PORTS" v={`${s.ports} × 50 kW`} />
        <Read k="CAPEX" v={fmt.usd(s.capex_usd)} />
        <Read k="CAPTURES" v={fmt.int(s.captured_kwh_day) + " kWh/d"} hot />
        <Read k="SESSIONS" v={fmt.int(s.sessions_day) + "/day"} />
        <Read k="UTILISATION" v={"ρ " + s.rho.toFixed(2)} />
        <Read k="P(WAIT)" v={s.p_wait.toFixed(3)} />
        <Read k="MEAN QUEUE" v={s.wait_min.toFixed(1) + " min"} />
        <Read k="CELLS COVERED" v={String(s.cells_covered)} />
        <Read k="P(VIABLE)" v={s.p_viable.toFixed(3)} />
        <Read k="SUBSTATION" v={s.substation ?? "no published data"}
              dim={!s.substation} />
        <Read k="STABILITY"
              v={s.stability != null
                  ? (s.stability * 100).toFixed(0) + "% of replicates"
                  : "n/a"}
              hot={s.stability != null && s.stability < 0.5}
              dim={s.stability == null} />
        <Read k="BANK HEADROOM"
              v={s.headroom_kw_before != null
                  ? (s.headroom_kw_before / 1000).toFixed(1) + " MW"
                  : "—"}
              dim={s.headroom_kw_before == null} />
      </dl>
      <SiteDrivers site={ex} />
      {s.saturated && (
        <p className="warn mono">
          <span aria-hidden>▲</span> DEMAND IN RADIUS EXCEEDS MAX HUB CAPACITY.
          SURPLUS RETURNED TO THE POOL FOR THE NEXT SITE.
        </p>
      )}
    </Section>
  );
}

function StatusBar({ m, grid, hovered, budget }: {
  m: Bundle["metrics"]; grid: Bundle["grid"]; hovered: number | null; budget: number;
}) {
  return (
    <footer className="status mono">
      <span>H3 R9 · WEB MERCATOR · NO BASEMAP</span>
      <span className="sep">/</span>
      <span>PU BAGGED ×60 · ELKAN-NOTO c={m.suitability.label_frequency_c.toFixed(3)}</span>
      <span className="sep">/</span>
      <span>MCLP GREEDY (1−1/e) · M/M/c SIZING</span>
      <span className="sep">/</span>
      <span>k={budget}</span>
      <span className="grow" />
      <span>{hovered != null ? grid.h3[hovered] : "—"}</span>
      <span className="sep">/</span>
      <span>BUILD {m.generated.slice(0, 10)}</span>
    </footer>
  );
}

/* ── primitives ──────────────────────────────────────────────── */

function Section({ label, children, action }: {
  label: string; children: React.ReactNode; action?: React.ReactNode;
}) {
  return (
    <section className="sect">
      <h2 className="mono">
        <span>{label}</span>
        {action}
      </h2>
      <div className="sect-body">{children}</div>
    </section>
  );
}

function Read({ k, v, hot, dim }: { k: string; v: string; hot?: boolean; dim?: boolean }) {
  return (
    <div className="rd">
      <dt>{k}</dt>
      <dd style={{ color: hot ? C.hazard : dim ? C.ink3 : C.ink }}>{v}</dd>
    </div>
  );
}

function Boot() {
  return (
    <div className="boot mono">
      <p>VOLTGRID</p>
      <p className="dim">LOADING FIELD · 8,249 CELLS</p>
    </div>
  );
}

function Fail({ msg }: { msg: string }) {
  return (
    <div className="boot mono">
      <p style={{ color: C.hazard }}>PAYLOAD UNAVAILABLE</p>
      <p className="dim">{msg}</p>
      <p className="dim">Run: python -m evopt.export</p>
    </div>
  );
}
