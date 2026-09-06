# DESIGN.md

## Scene sentence (this is what forced the theme)

A transportation planner and a utility engineer stand at a wall-mounted 32-inch display in a windowless DOT operations room at 7pm, two days before a capital-plan hearing, arguing over which 120 sites make the cut.

Dim room, long session, a dense continuous data field covering most of the screen. A light substrate would bloom on a wall display and flatten the field's low end. **Dark is forced by the scene, not chosen by category.**

## Colour strategy

**Restrained chrome, reserved data ramp.** Chrome is tinted neutral throughout. Saturated colour appears only where it encodes a measured value or flags an action.

The governing decision: **the measured field runs cool, interventions run hot.**

- The H3 demand surface uses a single-hue **blue sequential ramp**, dark to light. On this substrate the lightest step is the highest magnitude and near-zero recedes into the ground.
- **Hazard red is reserved exclusively** for recommended sites, constraint violations, and destructive actions. Nothing measured is ever red.

This is why the ramp is not the semantically obvious amber/heat: with hazard red as the mandated accent, an orange demand ramp would make "high demand" and "recommended action" indistinguishable at a glance. Cool field against hot intervention keeps the two channels unconfusable. Blue-for-data is a category reflex and it is being used anyway, on the grounds that the alternative breaks the more important separation.

### Tokens

```
--ground        #0d0d0c   substrate (never #000)
--surface-1     #161614   panel / chart surface
--surface-2     #1e1e1b   raised rows, hover
--hairline      #2a2a26   1px structural dividers
--ink           #eaeae6   primary (white phosphor)
--ink-2         #a3a29a   secondary
--ink-3         #6b6a63   muted, units, axis
--hazard        #e61919   RESERVED: plan sites, alerts, violations
--good          #0ca30c   status only
--warning       #fab219   status only
--critical      #d03b3b   status only
```

Sequential ramp, demand field (blue 700 -> 100, low to high):
`#0d366b #104281 #184f95 #1c5cab #256abf #2a78d6 #3987e5 #5598e7 #6da7ec #86b6ef #9ec5f4 #b7d3f6 #cde2fb`

Categorical, supporting charts (fixed order, never cycled):
`#3987e5 #d95926 #199e70 #c98500 #d55181`

Validated with the dataviz validator against surface `#0d0d0c`: lightness band PASS, chroma floor PASS, CVD separation PASS (worst adjacent dE 8.4 protan), normal-vision floor PASS (19.3), contrast PASS (all >= 3:1).

## Typography

Two families, strictly separated by job.

- **Macro / structural:** heavy neo-grotesque, uppercase, tracking `-0.03em`, leading `0.9`. Section headers and the few numbers that carry the argument. Fluid via `clamp()`.
- **Telemetry / data:** monospace, uppercase, 10-13px, tracking `0.08em`. Every unit, coordinate, label, table cell, axis tick. Tabular figures so columns of numbers align on the decimal.

No third family. No serif disruption here: this is an instrument, and the textural-serif move from the brutalist reference belongs to editorial work.

## Layout

Grid-deterministic. `display: grid; gap: 1px` over a `--hairline` ground so dividers are structural rather than drawn. **No `border-radius` anywhere.** No card grids: the field is one continuous object and panels dock to its edges.

Bimodal density is the point. The map is vast and quiet; the telemetry rails beside it are tight, packed monospace.

## Motion

Reserved and functional. Ease-out-quart, 180-260ms. Transitions only communicate state change: a recomputed plan, a hovered cell, a panel opening. Nothing loops, nothing decorates. No scanline animation, no fake tickers.

## Bans specific to this product

- No photographic or tiled basemap, and no tile server.
- **No third-party request of any kind.** Fonts are self-hosted; geometry, data and rendering are all local. Verified in CI, which fails the build if an external URL appears in `dist/`.
- No decorative scanlines or CRT glow that encode nothing.
- No rounded corners, no drop shadows, no glassmorphism.
- No number without a unit.
- No percentage without its denominator visible.
