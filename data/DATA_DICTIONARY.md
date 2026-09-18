# Data Dictionary

Field-level reference for both data sources. Written in Week 1; update it as
new sources or fields arrive.

> **Unit trap, read this first.** OpenSky reports distances and altitudes in
> **metres** and speeds in **metres per second**. ICAO separation minima are
> expressed in **nautical miles** and **feet**. Every labeling and feature
> function must convert explicitly. Getting this wrong produces a plausible
> looking model trained on meaningless labels.
>
> `1 nm = 1852 m`  ·  `1 ft = 0.3048 m`  ·  `1 kt = 0.514444 m/s`

---

## 1. OpenSky Network — ADS-B state vectors

**Primary modeling input.**

### Provenance

| | |
|---|---|
| Endpoint | `https://opensky-network.org/api/states/all` |
| Auth | OAuth2 client credentials (basic auth removed March 2026) |
| Bounding box | `lamin=32.5, lomin=-120.5, lamax=35.5, lomax=-116.0` (Southern California) |
| Retrieved by | [`src/data/fetch_opensky.py`](../src/data/fetch_opensky.py) |
| Granularity | One row per aircraft per snapshot |
| Sample file | `data/raw/states_socal_1787874146.csv` — 302 aircraft, 2026-08-27 16:42:26 local |

**Rate and history limits.** Authenticated access returns state vectors up to
1 hour old at 5-second resolution; anonymous access returns only the most
recent vectors at 10-second resolution. Neither provides the multi-month
archive Phase 1 needs — that requires OpenSky's **Trino** interface, which is a
separate approved access path.

### Fields

Order matches the API's returned array. `category` (index 17) only appears when
the request sets `extended=1`, so responses are commonly 17 fields wide, not 18.

| # | Field | Type | Units | Notes |
|---|---|---|---|---|
| 0 | `icao24` | string | — | 24-bit transponder address, lowercase hex. The stable aircraft identifier — use this for pairing, not callsign |
| 1 | `callsign` | string | — | Flight identifier, may be blank. Whitespace-padded by the API; stripped on ingest |
| 2 | `origin_country` | string | — | Inferred from the ICAO24 registration range |
| 3 | `time_position` | int | Unix seconds | Last **position** update. Null if no position report in the past 15 s |
| 4 | `last_contact` | int | Unix seconds | Last update of **any** kind |
| 5 | `longitude` | float | decimal degrees (WGS-84) | Nullable |
| 6 | `latitude` | float | decimal degrees (WGS-84) | Nullable |
| 7 | `baro_altitude` | float | **metres** | Barometric. Nullable — routinely null for ground traffic |
| 8 | `on_ground` | bool | — | True if the report came from a surface position |
| 9 | `velocity` | float | **m/s** | Ground speed, not airspeed. Nullable |
| 10 | `true_track` | float | degrees | Clockwise from true north (north = 0°). Nullable |
| 11 | `vertical_rate` | float | **m/s** | Positive = climbing, negative = descending. Nullable |
| 12 | `sensors` | int[] | — | Receiver IDs. Dropped on ingest: nested, and no signal for collision risk |
| 13 | `geo_altitude` | float | **metres** | Geometric (GNSS) altitude. Nullable |
| 14 | `squawk` | string | — | Transponder code. Nullable. 7700/7600/7500 are emergency codes |
| 15 | `spi` | bool | — | Special purpose indicator |
| 16 | `position_source` | int | enum | See below |
| 17 | `category` | int | enum | Aircraft size/type class, 0–20. Only present with `extended=1` |

**`position_source`**

| Value | Meaning |
|---|---|
| 0 | ADS-B — direct transponder broadcast, highest accuracy |
| 1 | ASTERIX |
| 2 | MLAT — multilateration, position inferred from receiver timing; less precise |
| 3 | FLARM — light aircraft/glider system |

**`category`** — 0 = no information, then increasing size classes (Light,
Small, Large, Heavy), plus Rotorcraft, Glider, UAV and surface vehicles.

### Which fields feed Week 2

| Feature | Fields used |
|---|---|
| Haversine distance | `latitude`, `longitude` |
| Vertical separation | `baro_altitude` (fall back to `geo_altitude`) |
| Closing speed | `velocity`, `true_track`, positions |
| Bearing difference | `true_track` |
| Time to CPA | positions + velocity vectors |

`icao24` identifies each aircraft in a pair; `time_position` and `last_contact`
gate whether a report is fresh enough to trust.

### Known quality issues

| Issue | Observed | Consequence |
|---|---|---|
| **Stale positions** | `time_position` can lag `last_contact` | The dangerous one. Pairing a stale position with a current one fabricates a conflict and injects false positives into an already tiny positive class |
| Null altitudes | 41 of 302 in the sample, all ground traffic | Vertical separation undefined; naive code treats null as 0 ft separation |
| Ground traffic | 41 of 302 airborne=False | Taxiing aircraft are not midair risks and pollute the negative class |
| Mixed `position_source` | ADS-B and MLAT interleaved | MLAT positions are less precise; consider filtering |
| Implausible kinematics | Not yet quantified | Bad decodes produce impossible closing speeds |

Filtering decisions made against this list are **research decisions**, not
plumbing — log each one to MLflow alongside the separation thresholds.

---

## 2. NTSB CAROL — aviation accident records

**Domain context only.** Not a modeling input and not a label source.

### Provenance

| | |
|---|---|
| Source | CAROL query tool, `https://data.ntsb.gov/carol-main-public/` |
| Filters | Country = USA; event date 2019-01-01 → present; no state filter |
| Records | 9,268 (JSON) — below the 10,000 export cap, so **not truncated** |
| Actual range | 2019-01-02 → 2026-08-25 |
| Files | `data/raw/ntsb_carol_2019_present_us.json` (78.6 MB, 28 fields)<br>`data/raw/ntsb_carol_2019_present_us.csv` (7.5 MB, 117 columns) |

**Why 2019.** The FAA's ADS-B Out mandate took effect 1 January 2020, so a 2019
cutoff roughly aligns the accident record with the era of usable ADS-B
coverage. The 10,000-row export cap also made a wider range impractical.

**JSON vs CSV.** They are different exports, not two formats of one thing. The
JSON carries `PrelimNarrative` (populated for only 820 of 9,268 records) and a
nested `Vehicles` structure. The CSV is wider and, despite being labelled a
"summary", carries **more** usable text: `ProbableCause` and `Findings` are
populated at 88%. For reading about events, prefer the CSV.

### Key CSV fields

Grouped rather than exhaustively listed; 117 columns, many of them injury
counts broken out by occupant role.

| Group | Fields | Fill |
|---|---|---|
| Identity | `NtsbNo`, `Mkey`, `EventID`, `ReportNo` | 100% |
| When / where | `EventDate`, `City`, `State`, `Country`, `Latitude`, `Longitude` | 99–100% |
| Event class | `EventType` (Accident / Incident / Occurrence), `InvestigationClass` | 100% |
| Aircraft | `N#`, `Make`, `Model`, `SerialNumber`, `AirCraftCategory`, `NumberOfEngines`, `EngineType`, `AmateurBuilt` | 97–100% |
| Operation | `FAR` (regulation part), `PurposeOfFlight`, `Scheduled`, `Operator`, `BroadPhaseofFlight` | 95–100% |
| Conditions | `WeatherCondition` (VMC/IMC), `AirportID`, `AirportName` | 69–98% |
| Outcome | `HighestInjuryLevel`, `AirCraftDamage`, `FatalInjuryCount` and ~20 related counts | 25–98% |
| **Analysis text** | `ProbableCause`, `Findings` | 88% |
| Status | `ReportStatus`, `ReportType`, `MostRecentReportType`, `OriginalPublishedDate` | 88–100% |

### Composition

| | |
|---|---|
| Event types | 9,143 Accident · 116 Incident · 9 Occurrence |
| Top states | TX 825 · CA 772 · FL 737 · AK 629 · AZ 349 |
| Midair references | **22 records** mention "midair" in `ProbableCause` |

Those 22 are the records most relevant to this project — worth reading before
writing the Week 2 labeling rule, to check how real loss-of-separation events
are actually described.

### Known quality issues

| Issue | Detail |
|---|---|
| **One malformed CSV row** | `ERA26LA025` (Boca Raton, 2025-10-28) has 118 fields against a 117-field header — an unescaped comma in NTSB's export. `pd.read_csv` fails outright; use `on_bad_lines='skip'`, which loads 9,267 of 9,268 rows. The JSON contains the full set |
| **`'unk'` string sentinel** | The `Aviation*Crew*` / `Aviation Passengers*` injury columns are 100% "filled" but largely contain the literal string `unk`. They are object-dtype, not numeric — do not aggregate without cleaning |
| Duplicate column names | The CSV repeats several column names; pandas suffixes them `.1` |
| Open investigations | Records with `ReportStatus = Ongoing` have no `ProbableCause` yet, which accounts for most of the missing 12% |
| Free-text fields | `BroadPhaseofFlight` concatenates repeated phases (`"Landing; Landing; Landing"`); parse, don't compare literally |

**Do not clean these files in place.** `data/raw/` mirrors what the source
produced. Cleaning belongs in a pipeline that writes to `data/processed/`.

---

## 3. Not yet acquired

| Source | Needed for | Status |
|---|---|---|
| **OpenSky Trino historical archive** | All Phase 1 training data — Weeks 2 onward | Application pending. Requires separate approval; not granted on registration. **Critical path** |

The REST snapshots collected so far prove the pipeline works but cannot supply
the multi-month trajectory sequences the models require.
