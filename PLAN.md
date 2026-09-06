# PLAN.md — EV Charging Infrastructure Siting

Seven stages, data ingestion through dashboard deployment. Each stage states its objective, its inputs and outputs, and the exit criterion that has to hold before the next stage is worth starting.

Status markers: **[done]** built and verified in this repo · **[partial]** working but with named gaps · **[next]** not started.

**A note on precision.** Every metric in this document is quoted to the number of decimals the Stage-7g seed sweep shows it supports (PR-AUC 3, lift 1). Figures re-derive on each pipeline run and move within the intervals reported in 7g.

Run the whole thing with `python run_pipeline.py`, or one stage with `python run_pipeline.py --only suitability`.

---

## Stage 1 — Data ingestion and provenance **[done]**

**Objective.** Get 11.5 GB of heterogeneous municipal records into typed, spatially-indexed columnar storage without loading any of it into memory at once, and record where every number came from.

| Input | Role | Volume |
|---|---|---|
| `ev_stations_v1.csv` | label source + temporal validator | 50,289 ELEC rows, 863 in the NYC window |
| `yellow_tripdata_*.csv` ×4 | primary demand signal | ~47.4 M trips, 7.4 GB |
| `Automated_Traffic_Volume_Counts.csv` | throughput + congestion | 14,359 segments, 3.3 GB |
| `charging_data.csv` (DE) | transfer prior, growth panel | 36,770 stations, 2014–2023 |
| `Electric_Vehicle_Charging_Stations.csv` (CT) | out-of-domain probe | 385 stations |
| `pluto.parquet` (NYC Open Data, added Stage 3) | built environment | 858,284 tax lots |

**Key decisions.**
- DuckDB streams CSV to aggregate in SQL rather than pandas: the whole corpus reduces in ~45 s on a laptop, no cluster.
- Traffic geometry is published in EPSG:2263 (NY State Plane, US feet) and is reprojected to WGS84 before any spatial work. Verified against a known Queens segment (94 Ave / 207 St → −73.754, 40.715).
- Connecticut is deliberately **excluded from training**: its records are a subset of what AFDC already carries, so including both would double-count and leak.

**Exit criterion.** Every source parses with a real CSV reader, coordinates fall inside the expected envelope, and `Open Date` coverage is high enough to support a temporal split. *Met: 862/863 NYC stations carry an open date.*

---

## Stage 2 — Spatial substrate and feature engineering **[done]**

**Objective.** Put every source on one comparable spatial unit and build features that describe a location's *context*, not just its own cell.

- **H3 resolution 9** (~0.105 km², 174 m edge) → 8,741 sitable cells. Uniform 6-neighbour adjacency, near-constant area, lossless roll-up to res 8.
- **Land mask from observation.** A cell with no taxi activity, no traffic counts and no tax lots is water or otherwise unreachable. No boundary shapefile needed; PLUTO's lot coverage later sharpened this, taking the universe from 8,249 to 8,741 cells.
- **Borough assignment** from the nearest labelled traffic segment within 2 km, which also drops the New Jersey collar the bounding box admits.
- **Spatial lag features** over k=1 and k=2 rings, for ten base fields. A site's value is largely determined by what surrounds it.
- **Leave-one-cell-out supply features.** Supply is computed twice: including all stations, and excluding stations sited in the cell being scored. Only the `_ex` variants reach the classifier.

**Why that last one matters.** A cell holding a station has `dist_station_m ≈ 0` by construction, so a classifier can recover its own label from its footprint. That drove Elkan-Noto *c* to 0.998 and turned the model into an incumbency detector. Removing it **improved** every forward-test metric.

**Exit criterion.** No feature available at scoring time encodes the label. *Met.*

---

## Stage 3 — Demand estimation **[done]**

**Objective.** Produce a demand surface in physically interpretable units (kWh/day) rather than an arbitrary score.

```
kWh_day = 0.29 kWh/mi × terminating taxi VMT          (measured)
        + AADT × cell traversal length × kWh/mi × EV share   (fleet)
        + dwell-weighted resting fleet
```

Model: LightGBM, **Tweedie objective** (`p = 1.3`) — correct for a non-negative, zero-inflated, heavy-tailed target where L2-on-log1p biases the conditional mean.

### The failure, and what fixed it

This stage originally **failed its exit criterion**. On traffic-volume features alone the model captured 22.4% of true energy in its top decile against 49.5% for simply sorting cells by traffic — it lost to the trivial baseline, and naive random k-fold hid that by reporting 66.8%.

The diagnosis was that traffic volume says how many vehicles *pass* a cell but nothing about what the cell *is*. Two feature blocks were added:

- **Built environment** — 858,284 NYC MapPLUTO tax lots (Socrata `64uk-42ks`, cached once to `data/raw/landuse/`). Land-use composition across 11 categories, Shannon mixed-use entropy, floor area split by residential / office / retail / factory / **garage**, residential units, built FAR, assessed value density.
- **Street-network morphology** — intersection density, distinct streets, arterial vs local share and directional diversity, parsed from the `street` / `fromSt` / `toSt` columns already in the traffic file.

| Ranking method | Top-decile energy capture |
|---|---|
| Oracle (perfect ranking) | 90.0% |
| Random k-fold CV *(leaky)* | 83.2% |
| **Model + land use, block CV** | **78.8%** |
| AADT-only baseline | 49.5% |
| Model without land use, block CV | 22.4% |

The built-environment block contributes **+56.4 points** of capture, measured as an ablation on every pipeline run rather than asserted. Block-CV R² rose from −0.018 to +0.169, R²(log) from 0.245 to 0.741, Spearman ρ from 0.576 to 0.834.

**The more important number.** The gap between block CV and random k-fold collapsed from 46.8 points to **4.4**. Spatial cross-validation punishes a model that has memorised position; a model that survives it has learned structure that transfers. That is what makes the Germany and Connecticut transfer tests in Stage 7 worth running at all.

Top features are almost entirely built-environment k-ring lags — `k2_jobs_area`, `k2_far_built`, `k2_garage_area`, `k2_lu_entropy` — confirming that a cell's demand is driven by what surrounds it, not what sits inside its own 174 m hexagon.

**Leakage guard held throughout.** No taxi-derived column, and no k-ring lag of one, may enter this model: the target *is* taxi VMT, so their inclusion would be circular and would destroy the transfer story the whole stage exists to support.

**Known gap.** Census ACS (household income, and specifically vehicle-ownership rate from table B25044) now requires an API key and is **not** included. PLUTO substitutes for its substance at finer resolution — `unitsres` for population, `assesstot` for economic intensity, office+retail floor area for employment — but vehicle ownership per household has no PLUTO equivalent and remains the one genuinely missing signal.

**Exit criterion.** Block-CV performance beats the AADT baseline on top-decile capture. **Met — 78.8% vs 49.5%, reaching 88% of the oracle ceiling.**

---

## Stage 4 — Suitability and validation **[done]**

**Objective.** Learn where a station is *viable*, without assuming that unbuilt means unsuitable.

- **Bagged PU** (Mordelet–Vert), 60 bags: all positives plus a resampled unlabeled pool per bag, out-of-bag scores averaged.
- **Elkan–Noto** calibration on held-out positives recovers *c* = P(labeled | positive) = **0.971**, so the output is a calibrated probability, not a presence score.
- **Temporal forward test** — the headline. Fit on the network as of 2018-12-31 (219 positive cells), score every unbuilt cell, evaluate against the 190 cells that actually received stations in 2019–2021.

| Method | PR-AUC | P@100 | Lift |
|---|---|---|---|
| **PU model** | **0.173** | **33.0%** | **14.8×** |
| Distance to existing | 0.099 | 19.0% | 8.5× |
| Demand ranking | 0.097 | 23.0% | 10.3× |
| Traffic ranking | 0.058 | 9.0% | 4.0× |
| Random | 0.022 | 1.0% | 0.5× |

The built-environment block lifted this stage too, from PR-AUC 0.130 to 0.173 (+33%). `garage_area` and `parking_share` rank second and fourth by importance: whether there is physically anywhere to put the hardware turns out to be one of the strongest predictors of where a charger actually got built.

**These figures move in the third decimal between runs.** Bagged PU resamples the unlabeled pool and LightGBM subsamples rows, so PR-AUC varies by roughly ±0.003 across identical pipeline runs. Quoting three decimals is therefore precision the estimate does not have; characterising that spread properly is Stage 7f.

PR-AUC is the headline, not ROC-AUC: at a 2.37% base rate ROC-AUC is dominated by the negative mass and flatters everything.

**Exit criterion.** Beat every baseline on the forward test. *Met — +80% PR-AUC over the strongest baseline.*

---

## Stage 5 — Gap analysis and allocation **[done]**

**Objective.** Turn a scored surface into a costed, constraint-satisfying site list. Ranking is not planning: the top-k cells overlap and cover the same demand repeatedly.

**Gap.** Supply decays with distance through a Huff/gravity kernel, with separate bandwidths per charger class (L2 400 m, DCFC 3 km). Two properties are enforced:
- **Conservation** — weights normalised per station, so no charger's energy is counted twice. Without it citywide coverage inflates by an order of magnitude.
- **Demand routing** — weights are kernel × demand, so energy flows where demand is. Distance alone credited chargers with serving empty cells (coverage ratios of 17,000%). Capacity that reaches no demand is reported as **stranded** (922 kWh/day) rather than silently counted.

**Allocation.** Capacitated Maximal Covering Location Problem, lazy-forward greedy with a (1 − 1/e) guarantee:

```
max Σ wᵢyᵢ   s.t.  coverage radius 800 m · min spacing 400 m
                  capture ≤ site capacity · ρ ≤ 0.70 · per-borough floor
```

- **Capacity limits capture.** A hub is 4–20 ports, so it absorbs at most `MAX_PORTS × 50 kW × 14 h × 0.70`. Without this the optimiser handed one Manhattan cell 37 MWh/day — a station needing 76 ports and an unstable queue (ρ = 4.4, "wait" of 6×10¹⁰ minutes).
- **Congestion is modelled.** Each site is an M/M/c queue; ports are sized so ρ ≤ 0.70, which keeps the queue stable by construction and lets the plan *state* an expected wait.
- **Equity floor** guarantees Bronx 18, Brooklyn 22, Queens 22, Staten Island 12.

**Result at k = 120:** 790 ports, $35.5 M capex, **38.5% of unmet demand served**, mean queue 5.0 min (p95 13.8 min).

**Exit criterion.** Every constraint provably holds in the emitted plan. *Met — max ρ = 0.70 exactly, floors satisfied.*

---

## Stage 6 — Dashboard **[done]**

**Objective.** Make the plan defensible to someone who will be questioned about it in public.

- Vite + React + TypeScript. The H3 field is a custom canvas renderer — **no tile server, no basemap, and no network at runtime**; boundaries ship pre-projected and quantised to integer metres, so the client needs neither an H3 library nor a projection step. Fonts are self-hosted (Stage 7h), so the built page makes **zero third-party requests** and runs on an isolated machine.
- Rank-scaled sequential ramp: demand spans four orders of magnitude, so a linear ramp would paint the city one colour.
- **The budget slider needs no refit.** Greedy selection is nested, so the first *k* sites *are* the greedy optimum for budget *k*.
- Design contract in `DESIGN.md`; palette validated against the dark surface (CVD separation, normal-vision floor, contrast all pass).
- The **adverse finding from Stage 3 is displayed in the product**, not buried in an appendix.

**Exit criterion.** Every headline number traces to a record count, and no percentage appears without its denominator. *Met.*

---

## Stage 7 — Hardening, transfer, and deployment **[done]**

### 7a · Cross-domain transfer validation **[done — later qualified by 7f]**

**Objective.** Is siting suitability city-invariant? Fit in one domain, test in another.

**The constraint that shaped the experiment.** Taxi, traffic counts and PLUTO are NYC-only. What Germany, NYC and Connecticut all have is a georeferenced network with ports, power, operator and a commissioning date. So the transferable feature space is **network morphology + temporal state** (23 features: k-th nearest distances, station/port/fast counts in five radius bands, local power, operator entropy, build velocity, network age), and the question is narrower than it first appears:

> Given the shape and age of an existing charging network, where does the next station go?

That is agglomeration dynamics. It is **not** demand-driven siting, because no demand signal exists outside NYC.

Every domain is treated identically — universe is the *expansion frontier* (H3 res-9 cells within ~1.1 km of a station open at the cutoff, holding no station themselves), label is "receives a station in the forward window." Candidates hold no station at the cutoff, so leave-one-cell-out is automatic. Connecticut is drawn from the **national AFDC file** (533 stations with open dates), not the 385-row state extract, which lacks them.

| Domain | Frontier cells | Positives | Base rate | Cutoff → window |
|---|---|---|---|---|
| Germany | 241,490 | 6,785 | 2.81% | 2019 → 2020–22 |
| NYC | 3,104 | 130 | 4.19% | 2018 → 2019–21 |
| Connecticut | 7,090 | 75 | 1.06% | 2018 → 2019–21 |

**Transfer matrix (PR-AUC), rows trained on, columns tested on:**

| | DE | NYC | CT |
|---|---|---|---|
| **DE** | *0.123* | 0.110 | 0.019 |
| **NYC** | 0.055 | *0.330* | 0.040 |
| **CT** | 0.049 | 0.090 | *0.101* |

Diagonal is the within-domain reference (out-of-bag). Frontier sizes differ by two orders of magnitude, so columns are only comparable against their own diagonal — never across rows.

**The decision test.** Retention against a locally-fit model flatters transfer: a domain with a weak local model makes any import look competitive. The question that matters is whether shipping a foreign model beats what the target could already do for free — ranking by distance to the nearest existing station.

| Transfer | Model | Target baseline | Ratio | |
|---|---|---|---|---|
| NYC → CT | 0.040 | 0.021 | **1.90×** | pass |
| NYC → DE | 0.055 | 0.053 | 1.04× | pass, marginal |
| CT → DE | 0.049 | 0.053 | 0.92× | **fail** |
| DE → CT | 0.019 | 0.021 | 0.92× | **fail** |
| DE → NYC | 0.110 | 0.132 | 0.84× | **fail** |
| CT → NYC | 0.090 | 0.132 | 0.68× | **fail** |

**Result: 2 of 6 transfers beat their target's trivial baseline, and only one does so meaningfully.**

> **Superseded in part.** This ran on network morphology alone. Stage 7f adds a globally-consistent built-form block and lifts the pass rate from 5/12 to 9/12 across an expanded domain set, so the correct claim is narrower than it reads here: a network's own *shape* does not transfer, but built form largely does. The one conclusion that survives unchanged is that **nothing transfers into NYC**.

**What this overturns.** The Stage-2 architecture proposed Germany as a pretraining prior on the reasoning that a converged, centrally-registered network at ~5× US port density encodes what a finished network looks like. **That proposal is wrong.** DE → NYC scores 0.110 against a 0.132 baseline: a model fit on 13,005 German stations is *worse than ranking NYC cells by distance to the nearest charger*. Pretraining on Germany and deploying in the US is not supported by the evidence and should not be done.

**What does transfer.** NYC → CT at 1.90× is the one genuine success: same country, same AFDC data-generating process, same charging market and incentive regime, differing mainly in density. Transfer appears to be bounded by *market and regulatory regime*, not by geometry.

**Why Germany is hard to predict at all.** DE → DE is only 0.123 against NYC's 0.330 within-domain. German 2020–22 build-out was driven by national subsidy programmes that pushed chargers into *new* areas rather than densifying existing clusters — exactly the dynamic a network-morphology model cannot see. Policy shocks are not in the feature space.

**Honest scope.** This bounds agglomeration dynamics only. It does not rule out that a demand-and-land-use model would transfer better; it says the network's own shape does not carry across markets. Testing the richer feature space would require PLUTO-equivalent land use for Germany and Connecticut, which is Stage 7d.

### 7b · Certified optimum **[done]**

**Objective.** Stage 5 shipped a (1 - 1/e) ~ 63.2% guarantee. That is a worst-case theorem about adversarial instances, not a measurement of this one. A planner told "at least 63% of optimal" is entitled to ask what the number actually is.

**Why not an MCLP.** The textbook Maximal Covering Location Problem cannot express this plan. Capture here is capacity-limited -- a hub absorbs at most `SITE_CAPACITY_KWH` however much demand sits in its radius -- and MCLP's binary "cell i is covered" has nowhere to put that. The exact model is therefore a **capacitated facility-location problem with explicit allocation**:

```
max  sum_ij z_ij                        energy actually served
s.t. sum_j z_ij <= w_i       for i       cannot serve absent demand
     z_ij       <= w_i x_j   for i,j     only an open site serves
     sum_i z_ij <= CAP x_j   for j       hub capacity
     sum_j x_j  <= k                     budget
     x_j + x_l  <= 1         d(j,l)<s    anti-clustering
     sum over borough b of x_j >= floor_b   equity floor
```

Same candidate pool, same weights, same constraints as greedy, so the two are directly comparable. Solved with CBC via PuLP: 1,500 binaries, 13,564 allocation variables, 2,204 spacing pairs.

| | kWh/day served |
|---|---|
| Certified optimum (MILP) | **381,043** |
| Greedy | 363,135 |
| Headroom | 17,908 (4.70%) |

**CBC proved optimality in 46.9 s**, so the ratio is exact rather than a bound:

> **Greedy achieves 95.30% of the certified optimum**, clearing its own 63.21% worst-case guarantee by 32.1 points.

**What this licenses.** The dashboard's live budget slider recomputes plans with greedy on every drag, which was previously defensible only by a 63% floor. At 96% measured, greedy is the right runtime choice and the MILP is the audit path: run it once for the plan that actually goes into a capital programme. The worst case is a theorem about adversarial instances, and a real city is not adversarial.

**Caveat.** The certificate is computed for one budget (k=120). It says nothing about other k, and the dashboard flags that explicitly when the slider is moved off it.

### 7c · Per-site attribution **[done]**

**Objective.** Stage 7's exit criterion is that a planner can answer "why this corner" for every entry. A ranked score cannot; an attribution can.

**No new dependency.** LightGBM's `predict(..., pred_contrib=True)` *is* TreeSHAP, exact and already in the project. Installing the `shap` package would add a dependency to compute the same numbers more slowly.

**Two things this had to get right.**

*Space.* For a binary objective the contributions are raw margin (log-odds), not probability. They sum to the margin and sigmoid of that sum recovers the model's probability. Labelling them "probability points" would be wrong, so everything is reported as log-odds.

*The ensemble is nonlinear in its parts.* The PU score is `mean_k sigmoid(margin_k)` over 60 bags, but averaging contributions yields `mean_k margin_k`, and sigmoid is not affine — so the attributions explain the **mean log-odds**, a slightly different quantity from the score that was ranked. That gap is measured rather than waved away:

| | |
|---|---|
| Spearman vs ranked score | **0.9985** |
| Pearson | 0.9982 |
| mean abs probability error | 0.0106 |
| max abs probability error | 0.0891 |

Ordering is preserved, which is what an explanation of a *ranking* needs. The ensemble is rebuilt with the same seed as `suitability.run()` and verified against the stored scores to a max absolute error of 2.9e-05 — reproduction is checked, not assumed.

**Global drivers (mean |SHAP|, log-odds).**

| Feature | mean \|SHAP\| |
|---|---|
| Parking floor area | 0.656 |
| Assessed value density | 0.603 |
| Parking share of floor area | 0.435 |
| Unmet demand | 0.391 |
| Taxi dropoffs | 0.349 |
| Nearby L2 supply | 0.322 |

**Parking floor area leads**, which is the physically right answer: charging hardware goes into parking structures, so a cell with none is hard to build in however much demand passes through. That column came in with the PLUTO block in Stage 3 and was the reason for choosing PLUTO over census tracts.

**The labelling error this exposed, and the counterfactual that fixed it.**

The panel was first titled "Why this site" — which is wrong. SHAP explains the **viability model**, not the **selection decision**. The optimiser maximises unmet demand × viability under a coverage radius, spacing rule, hub capacity *and* a per-borough equity floor, so a low-viability cell can be in the plan for reasons the attribution does not show. 47 of 120 sites score below 0.5 viability, every one of them in an equity-floor borough.

Attributing that per site needs a counterfactual, not a guess. Greedy is deterministic, so it is re-run with `min_per_borough` removed and the two plans diffed: anything present under the floor and absent without it is there *because* of the floor.

> **26 of 120 sites exist because of the equity floor.**

The heuristic reading — "47 low-viability sites must be equity-driven" — would have overstated it by nearly 2×. The other 21 low-viability sites survive on merit, because they capture unmet demand nothing else reaches.

The panel is now titled **Viability attribution**, and each site carries the counterfactual verdict: selected on merit, or selected by the floor. A planner asked "why this corner" in a hearing can now separate "the model rates it highly" from "our accessibility mandate required it" — two very different answers to give in public.

### 7d - Vehicle ownership **[done]**

**The block that was not a block.** PLAN.md recorded this as blocked on a Census API key. The API is indeed key-gated, but the **ACS bulk summary file is not** - `acsdt5y2022-b25044.dat`, 63.5 MB, no key, same estimates.

**Tract to cell, conserving households.** A census tract dwarfs an H3 res-9 cell, so tract values are *allocated*, not broadcast. PLUTO carries `bct2020` on 99.9% of lots, so each tract's households split across its cells in proportion to the residential units standing there. Households follow homes.

```
households allocated  3,280,860
source total          3,280,860
error                 0.0000%
```

Sanity against reality: 54.7% zero-vehicle households (NYC's actual rate is ~54%), 2.06M vehicles, 0.834 vehicles per household. **659,357 cars with no home charging.**

**The derived feature, and what the model did with it.** The hypothesis was that raw car counts mislead - high ownership usually means private off-street parking, while Manhattan has few cars whose owners have nowhere to plug in. So a composite shipped alongside its own components:

```
home_charge_cov = min(1, garage_spaces / units_res)
no_home_charge  = renter_veh_hh x (1 - home_charge_cov)
```

TreeSHAP adjudicated, and **the hypothesis was half wrong**:

| Feature | Global rank | Source |
|---|---|---|
| `home_charge_cov` | **#6** | PLUTO garage / units |
| `no_home_charge_density` | #17 | composite |
| `no_home_charge` (count) | **not in top 25** | composite |
| every raw ACS vehicle feature | **not in top 25** | ACS |

The model kept the *denominator* and discarded the composite built on top of it. Garage floor area per residential unit carries the signal; the ACS vehicle layer adds little once the built environment is known. Shipping the composite next to its components is what made that visible instead of assumed.

**Effect on the models.** Demand block-CV top-decile capture 78.5% (essentially unchanged - the built environment already carried this signal). Suitability improved: P@100 **34.0%**, lift **15.25x**, NDCG **0.702**.

---

### 7e - Feeder capacity **[done]**

**The constraint that was in the architecture but never in the code.** `allocate.py` had no grid term, so the tool recommended sites the network might not support. Con Edison publishes the exact quantity - rating at the feeder head minus last year's peak - via the hosting-capacity ArcGIS service (`CECONY_EV_Storage_Prod` layer 0, 59,600 feeder polylines in the NYC envelope).

**The double-count this design exists to avoid.** `TOTAL_XFMR_CAPACITY` and `SUBST_PEAK` are **substation-bank** values that repeat on every feeder hanging off the bank - five separate feeders all quote WASHINGTON STREET's 264 MW / 202 MW. Summing per feeder would license capacity that does not exist:

```
headroom, deduped to substation      1,686 MW
headroom, naive per feeder          43,110 MW
naive overstates by                     25.6x
```

Same failure mode as the Stage-5 gravity kernel, where a charger's energy was counted once per cell it could reach. The constraint is therefore enforced per **bank**, not per feeder.

**The grid shrinks hubs, it does not ban them.** A site gets as many ports as its bank can still carry - what actually happens at interconnection review. A bank with under one port of headroom left excludes its cells outright.

**Result: the grid is not the binding constraint.**

| k | constrained | unconstrained | cost |
|---|---|---|---|
| 120 | 395,496 | 395,656 | 0.04% |
| 200 | 416,841 | 418,067 | 0.29% |
| 300 | 420,181 | 422,258 | 0.49% |
| 400 | 420,273 | 422,433 | 0.51% |
| 500 | 420,273 | 422,433 | 0.51% |
| 600 | 420,273 | 422,433 | 0.51% |

At k=120 the limit costs **0.04%** of served energy and displaces 1 sites; no site is port-limited. The cost plateaus near 0.51%, and the grid exhausts the candidate pool at **340 sites versus 355** without it. Even the tightest bank (JAMAICA, 3 MW headroom) runs at 6.7% of its remaining capacity under this plan. Demand geography and the 800 m coverage radius bind long before Con Edison does.

**The caveat that bounds that claim - and a bug it caught.** The first implementation assigned every cell its nearest feeder unconditionally. That put 295 **Manhattan** cells on Brooklyn's BROWNSVILLE bank across the East River, with a p90 assignment distance of 4,009 m. Layer 0 covers 3-phase *radial* feeders; Manhattan runs on Con Edison's secondary **network** system, published separately. Charging a Manhattan site against a Brooklyn bank is worse than not constraining it at all.

Assignment is now capped at 1 km, and cells beyond it are marked "no published data" and left unconstrained - absence of published capacity is not evidence of available capacity. Median distance fell 418 m -> **117 m**.

| Borough | Cells | With published grid data |
|---|---|---|
| Manhattan | 960 | **0.0%** |
| Brooklyn | 1,851 | 58.0% |
| Queens | 3,266 | 66.2% |
| Bronx | 1,272 | 68.8% |
| Staten Island | 1,393 | 97.0% |

So the negative result holds **only where the data reaches**: 5,460 of 8,742 cells, and 38 of the 120 plan sites. Manhattan's network capacity is genuinely unknown from this source. A related bug was fixed alongside: missing headroom was being filled with 0, which reads as "no capacity" and would have silently excluded every uncovered cell from the plan.

**Certification re-run.** The MILP gained 20 substation constraints, charging every opened site the full MAX_PORTS draw - deliberately conservative, so it can only shrink the feasible region and the certified ratio stays honest. CBC proved optimality in 91.4 s: greedy achieves **95.30%** of the certified optimum.

### 7f - Land-use transfer **[done]**

**What 7a could not settle.** Stage 7a found that 4 of 6 transfers failed to beat the target's own trivial baseline, but it ran on **network morphology alone**, because taxi, traffic and PLUTO stop at the NYC line. So its conclusion was under-determined: did *nothing* transfer, or had it only shown that a network's own shape does not?

**A genuinely common feature space.** Overpass is reachable but rate-limited to two concurrent queries, hopeless against Germany's 241,490 frontier cells. The JRC Global Human Settlement Layer is open, global, and produced by one method everywhere -- `GHS_BUILT_S` and `GHS_POP`, WGS84 at 3 arc-seconds (~90 m), so a Queens cell and a Berlin cell are measured identically. Sampled at two scales (a ~450 m window for the cell, ~1.3 km for its surroundings, the raster-native analogue of the k-ring lags that dominate the NYC model), plus cell-versus-context contrast ratios. 23 features -> 29.

**A latitude bug this caught.** Pixel area was first computed as `(RES x 111,320)^2`, ignoring that a 3 arc-second pixel keeps its height but narrows as cos(latitude). Germany sits at ~52N (cos 0.62) and New York at ~40.7N (cos 0.76) -- a latitude-blind denominator would have baked a ~19% systematic bias into precisely the quantity this experiment compares across those countries. With the correction, implied NYC population lands at 8.78M against the true ~8.3M, and peak density at 82,013/km2 against Manhattan's real ~70,000.

**Built form helps everywhere, internally.**

| Domain | network only | + built form | delta |
|---|---|---|---|
| DE | 0.123 | 0.169 | +0.045 |
| DE-metro | 0.423 | 0.515 | +0.092 |
| NYC | 0.330 | 0.414 | +0.084 |
| CT | 0.101 | 0.154 | +0.053 |

**And it substantially rescues transfer.**

| Transfer | network only | + built form |
|---|---|---|
| DE -> DE-metro | 0.240 pass | 0.296 pass |
| DE -> NYC | 0.111 **fail** | 0.083 **fail** |
| DE -> CT | 0.019 **fail** | 0.036 pass |
| DE-metro -> DE | 0.078 pass | 0.133 pass |
| DE-metro -> NYC | 0.076 **fail** | 0.051 **fail** |
| DE-metro -> CT | 0.023 pass | 0.023 pass |
| NYC -> DE | 0.055 pass | 0.077 pass |
| NYC -> DE-metro | 0.097 **fail** | 0.111 pass |
| NYC -> CT | 0.040 pass | 0.051 pass |
| CT -> DE | 0.049 **fail** | 0.071 pass |
| CT -> DE-metro | 0.103 **fail** | 0.133 pass |
| CT -> NYC | 0.090 **fail** | 0.092 **fail** |

> **5 of 12 transfers beat their target baseline on network shape alone; 9 of 12 once built form is added.**

So 7a was under-determined, and its headline needs qualifying: a network's own *shape* does not transfer, but built form and population substantially do.

**The exception, which is the important part.** **Every transfer into NYC still fails**, and DE -> NYC got *worse* with built form (0.111 -> 0.083). NYC carries the strongest trivial baseline of any domain (0.132 versus 0.053 for Germany and 0.021 for Connecticut), reflecting an unusually clustered network: ranking by distance to the nearest existing charger is already very good there, and no foreign model beats it. **Do not import a model into New York.**

**Full transfer matrix (PR-AUC, + built form), rows trained on, columns tested on**

| | DE | DE-metro | NYC | CT |
|---|---|---|---|---|
| **DE** | 0.169 | 0.296 | 0.083 | 0.036 |
| **DE-metro** | 0.133 | 0.515 | 0.052 | 0.023 |
| **NYC** | 0.077 | 0.111 | 0.414 | 0.051 |
| **CT** | 0.071 | 0.133 | 0.092 | 0.154 |

**My 7a hypothesis, tested and confirmed.** 7a speculated that Germany's weak within-domain score came from nationwide, subsidy-driven expansion into rural areas that network morphology cannot see. Restricting to the seven densest cities settles it: DE-metro -> DE-metro reaches **0.423** against nationwide DE -> DE's **0.123**. Base rate differs (7.85% vs 2.81%), and PR-AUC scales with base rate, so the fair comparison is lift over each domain's own baseline: **4.0x for metros against 2.3x nationwide**. The hypothesis holds even after normalising -- dense German cities are genuinely more predictable than the country as a whole.

**Scope, stated rather than buried.** GHSL carries built *form* and *population*, not land-use *composition*. NYC's own strongest signals -- office/retail floor area, garage area, land-use entropy -- have no global equivalent. This result therefore bounds **"built form and population transfer"**, which is strictly weaker than "land use transfers". Closing that last gap would need a PLUTO-grade parcel dataset for Germany and Connecticut, which does not exist in open form.

**Cost, and a correction.** The first dual-space run took **3.1 hours** and I reported that as the stage cost. It is not: that run paid cold-cache penalties, reading four freshly-written 12000x12000 GHSL tiles per product with nothing in the OS page cache. Warm, the same stage completes in **8.6 minutes**, and the full 17-stage pipeline in **23 minutes**. The honest figures are 8.6 min steady-state and ~3 h on a cold first run, which is a one-time cost per machine.

### 7g - Uncertainty **[done]**

**Two questions, two methods.** Conflating them would answer neither, so they are measured separately.

**(a) Which sites are robust?** `bagged_pu` is an ensemble of 60 bags, so resampling those bags with replacement yields many equally valid viability surfaces. Each produces its own greedy plan, and a site's **selection frequency** across 200 replicates is its stability. Per-bag predictions are cached once, so a replicate costs only a greedy run.

| | |
|---|---|
| Sites selected in **every** replicate (>=0.99) | **77 of 120** |
| Sites that are a coin flip (<0.50) | **3 of 120** |
| Median stability | 1.00 |
| Captured, 95% interval | 359,736 - 364,921 kWh/day |

A ranked list hides this distinction entirely: rank 40 and rank 41 look identical on paper, but one may be chosen every time and the other in a third of replicates. Each site now carries its frequency in the inspector.

**(b) How precise are the headline numbers?** PR-AUC drifts between identical pipeline runs because LightGBM's threaded histogram build is not bit-deterministic. A bag bootstrap is blind to that -- it conditions on one fitted ensemble -- so the only honest measurement is refitting. 15 seeds:

| Metric | mean +/- sd, 95% interval | Decimals supported |
|---|---|---|
| PR-AUC | 0.1706 +/- 0.0028  [0.1657, 0.1749] | 3 |
| P@100 | 0.3313 +/- 0.0083  [0.3200, 0.3465] | 3 |
| lift@100 | 14.86 +/- 0.37  [14.35, 15.54] | **1** |
| NDCG | 0.6978 +/- 0.0068  [0.6843, 0.7076] | 3 |

The earlier informal estimate of "+/-0.003" on PR-AUC was almost exactly right (sd 0.0028). The correction this forces is on **lift**, which supports only **1 decimal place** -- so the honest figure is 14.9x, and every earlier "14.86x" was false precision. The dashboard now prints each metric to the precision its own spread supports, with the supported-decimal count shown beside it.

**A flaw in my own check.** The first bias guard compared `P.mean(axis=0)` against `full` -- but `full` *is* that mean, so it could only ever return zero and tested nothing. It duly returned `0.00e+00`, which reads like a pass and was worthless. It now compares the mean across the 200 *resampled* ensembles against the point estimate, which is a real question: does resampling skew the surface it is meant to characterise? Measured residual: 8.78e-03.

### 7h - Deployment **[done]**

**Self-hosted fonts, and a claim made true.** Earlier in the project I had to *weaken* a claim: PLAN.md, DESIGN.md and PRODUCT.md all said the page made "no network request at runtime", which was false because `index.html` pulled a Google Fonts stylesheet. Rewording it was the honest short-term fix; removing the dependency is the real one.

Subsets were chosen from what the interface actually renders rather than by reflex. Scanning the source for non-ASCII codepoints found **rho (U+03C1) and Delta (U+0394) in monospace telemetry readouts**, so JetBrains Mono needs the *greek* subset as well as latin and latin-ext; Archivo carries ASCII headings only. Five woff2 files, **114.7 KB**, with `unicode-range` preserved so a browser still downloads only what a page needs (in practice three of the five). Both families are SIL OFL 1.1, which permits redistribution.

The original, stronger claim is now restored in all three documents, and **CI fails the build if any external URL appears in `dist/`** so it cannot silently regress.

**The static build genuinely stands alone.** Served from a bare `python -m http.server` with no dev server and no bundler in the loop:

```
origin              http://localhost:5391
panels rendered     8 / 8
fonts               Archivo + JetBrains Mono, greek subset included
external requests   []          <- zero
resources           10, all same-origin
```

**Freshness and drift, `src/evopt/drift.py`.** Two questions a deployed pipeline must answer with nobody watching.

*Is anything stale?* Sources change on their own schedules -- AFDC republishes, PLUTO cuts a release, Con Edison refreshes hosting capacity twice a year. A manifest fingerprints 20 raw files across 8 groups (size plus head/tail bytes, so a 3.3 GB CSV checks in milliseconds instead of minutes) and maps each group to the stages that depend on it. The output is "re-run these six stages", not "re-run all eighteen".

*Has the model drifted, or is this noise?* This is where Stage 7g earns its keep a second time. A naive threshold like "alert if PR-AUC falls 0.01" would fire on nothing, because the metric moves +/-0.003 between identical runs from non-deterministic threaded training. The seed sweep already measured the interval the metric occupies **under no change at all**, so that interval becomes the control limit: inside is noise, outside is worth a human.

| Metric | current | 95% control limits | verdict |
|---|---|---|---|
| `pr_auc` | 0.1732 | 0.1657 - 0.1749 | ok |
| `precision@100` | 0.3300 | 0.3200 - 0.3465 | ok |
| `lift@100` | 14.8014 | 14.3528 - 15.5414 | ok |
| `ndcg` | 0.7037 | 0.6843 - 0.7077 | ok |

The temporal split gets the same treatment. The forward window is fixed at 2019-2021 while AFDC keeps publishing, so the check reports how much build-out now falls past the window: **356 stations (0.7%)**, against a deliberately conservative 5% trigger for a human to re-cut the split. Latest AFDC open date is 2022-10-04. Current verdict: **no action required**.

**CI, `.github/workflows/ci.yml`.** The 11.5 GB corpus is gitignored and not reproducible on a runner, so CI verifies everything that does not need it: the dashboard typechecks and builds, the external-URL assertion passes, every `evopt` module imports cleanly, and `scripts/check_reports.py` asserts that the figures PLAN.md prints still match `reports/` at the precision 7g supports. **This is written but unverified here** -- the project has no GitHub remote, so the workflow has never actually executed. `scripts/check_reports.py` does run locally and passes (3 checks).

**Pipeline is now 18 stages**, ending with `drift`, so every full run leaves behind a fresh manifest and control-limit verdict.

### Operational follow-ups **[not blocking]**

Nothing outstanding in Stage 7. The remaining work is operational rather than methodological: run `ci.yml` against a real remote, and schedule `python run_pipeline.py --only drift` so the manifest and control-limit check run without being asked.

**Exit criterion — met.** Every site carries a viability attribution, a counterfactual verdict on whether the equity floor placed it, its substation and grid headroom, and a stability score over 200 resampled ensembles. "Why this corner" and "how sure are you" both have answers, the plan is buildable on the real distribution grid, and the instrument ships as a self-contained static build that makes no third-party request.

---

## Repository layout

```
EV-Project-Optimization/
├── PLAN.md · PRODUCT.md · DESIGN.md · run_pipeline.py
├── data/
│   ├── raw/{stations,mobility,traffic}/     11.5 GB, gitignored
│   ├── raw/landuse/pluto.parquet            858k tax lots, fetched once
│   ├── interim/                             typed parquet
│   └── processed/                           H3 features, surface, plan
├── src/evopt/
│   ├── config.py · geo.py · features.py · surface.py · export.py
│   ├── ingest/{stations,mobility,traffic,landuse}.py
│   ├── ingest/{stations,mobility,traffic,landuse,acs,ghsl,feeders}.py
│   ├── models/{demand,suitability,allocate,certify,explain,uncertainty,transfer}.py
│   └── drift.py                      freshness manifest + control limits
├── models/            trained boosters
├── reports/           metrics JSON, including the adverse result
└── dashboard/         Vite + React + TS
```

## Known limitations

- Taxi trips are a biased sample of the vehicle fleet, concentrated in Manhattan. The fleet term anchors on traffic counts to compensate, but outer-borough demand is the weaker estimate.
- The 2015–2016 mobility window predates both the pandemic and the TLC electrification mandate. It measures *structure* — where trips terminate and vehicles dwell — which is more stable than volume, but it is not current.
- Cost is a flat $45k/port. Real siting cost is dominated by electrical service upgrades and varies by an order of magnitude.
- No parcel-level feasibility. A recommended cell is 0.105 km²; whether a specific lot inside it is available, zoned, and connectable is out of scope — though PLUTO now makes lot-level screening a tractable next step rather than a research problem.
- Vehicle ownership rate is absent (ACS key required). In a city where car ownership varies from 22% in Manhattan to 83% on Staten Island, this is a real omission from the demand estimate.
- PLUTO is a current snapshot while the mobility window is 2015–2016. Building stock changes slowly enough that this is a minor mismatch, but it is a mismatch.
- **The model is NYC-specific.** Stage 7a shows that network-morphology suitability does not transfer across markets: only NYC → Connecticut clears its baseline. Do not deploy this model in another country without refitting.
