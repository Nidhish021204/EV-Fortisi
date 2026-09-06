# PRODUCT.md

**register:** product

## Product Purpose

A planning instrument for siting electric-vehicle charging infrastructure across New York City. It reads 11.5 GB of municipal mobility and traffic records, estimates where charging demand actually falls, measures where the existing network fails to serve it, and solves for the set of new sites that closes the largest share of that gap under a capital budget, a queueing-delay ceiling, and a per-borough equity floor.

It is not a map viewer. It is the artifact a planner brings to a capital-allocation meeting and has to defend line by line.

## Users

**Primary — the transportation planner.** Owns a capital program with a fixed budget. Needs a ranked, costed site list, and needs to know why each site is on it. Will be asked "why this corner and not that one" in a public hearing and cannot answer "the model said so."

**Secondary — the utility grid planner.** Cares about feeder capacity and simultaneous load. Reads the port counts and the kilowatt totals before anything else.

**Tertiary — the modeler.** Has to defend the methodology. Needs validation metrics visible in the product, including the ones that are unflattering, because a reviewer will find them anyway.

## Tone

Instrument, not dashboard. The interface reports measurements and states its own uncertainty. It never congratulates the user, never uses an exclamation mark, and never rounds a number to make it look better. Where the model underperforms a trivial baseline, the interface says so in the same typeface it uses for its successes.

Numbers carry units. Every derived quantity can be traced to the record count it came from.

## Anti-references

- **Consumer EV apps** (PlugShare, ChargePoint). Rounded cards, friendly green, pin-drop maps, gamified availability. Wrong audience, wrong stakes.
- **Generic BI dashboards** (Power BI / Tableau default themes). Grey chrome, drop shadows, a grid of identical KPI tiles, chart junk.
- **The hero-metric template.** One enormous number over a gradient, three supporting stats beneath.
- **Crypto-terminal cosplay.** Matrix-green monospace everywhere, fake tickers, decorative scanlines that encode nothing. The technical look has to be earned by actual technical density.
- **Photographic basemaps.** Satellite tiles and street imagery are noise here; they compete with the data field and imply a precision the model does not claim. The field is rendered from local geometry, not fetched from a tile server, and the page makes no third-party request at all.

## Strategic principles

1. **The data field is the interface.** The map is not a backdrop with widgets on top. The hex surface is the primary object and everything else annotates it.
2. **Colour belongs to data, not to chrome.** Chrome is tinted neutral. Saturated colour appears only where it encodes a measured value or flags an alert.
3. **Every claim shows its denominator.** "40.3% of unmet demand" is meaningless without the 958,274 kWh/day it is a share of. Both appear.
4. **Show the results, including the ones that were wrong.** The demand model once lost to a trivial AADT baseline under spatial cross-validation. The product shows that failure, the ablation that diagnosed it, and the fix — not just the final number.
5. **No interaction without a decision behind it.** Controls exist to change the plan (budget, radius, equity floor), not to let the user fiddle with visual settings.
