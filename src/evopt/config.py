"""Central configuration. All paths resolve from the repo root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
DASH_DATA = ROOT / "dashboard" / "public" / "data"

# ── Source files (see README dataset placement spec) ────────────────
AFDC_NATIONAL = RAW / "stations" / "afdc_national" / "ev_stations_v1.csv"
STATIONS_CT = RAW / "stations" / "connecticut" / "Electric_Vehicle_Charging_Stations.csv"
STATIONS_DE = RAW / "stations" / "germany" / "charging_data.csv"
TAXI_GLOB = str(RAW / "mobility" / "nyc_taxi" / "yellow_tripdata_*.csv")
TRAFFIC = RAW / "traffic" / "nyc_volume" / "Automated_Traffic_Volume_Counts.csv"

# Current-era for-hire window (Stage 8b). Q1, because the taxi baseline is
# 2015-01 + 2016-01..03 -- matching the season keeps the zone-share comparison
# from picking up seasonal redistribution and calling it growth. These are the
# same months the aggregate window covers, so the count-vs-VMT cross-check
# compares the same trips rather than two different periods.
TLC_CURRENT_MONTHS = ["2026-01", "2026-02", "2026-03"]
TLC_INDUSTRIES = ["fhvhv", "yellow", "green"]

# ── Spatial substrate ───────────────────────────────────────────────
H3_RES = 9                 # ~0.105 km2, 174 m edge
H3_RES_COARSE = 8          # roll-up for context features

# NYC analysis window (5 boroughs + a small collar so edge hexes have
# neighbours; NJ cells are masked out later by the borough polygon test).
NYC_BBOX = dict(lat_min=40.48, lat_max=40.93, lon_min=-74.28, lon_max=-73.68)

# ── Physical constants for the demand model ─────────────────────────
KWH_PER_MILE = 0.29        # fleet-average EV consumption
MEAN_SESSION_KWH = 22.0    # calibrated against the German power panel
L2_KERNEL_M = 400.0        # destination charging: you walk from it
DCFC_KERNEL_M = 3000.0     # en-route charging: you drive to it

# ── Temporal split for forward validation ───────────────────────────
TRAIN_CUTOFF = "2018-12-31"   # fit on the network as of this date
TEST_WINDOW = ("2019-01-01", "2021-12-31")

for _d in (INTERIM, PROCESSED, MODELS, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)
