import pandas as pd, xarray as xr, pathlib

RAW   = pathlib.Path("data/raw")
PROC  = pathlib.Path("data/processed"); PROC.mkdir(parents=True, exist_ok=True)

# ── 3-a  HADCRUT5 global annual mean (works with 4-column file) ────────────
had = (pd.read_csv(
          RAW / "HadCRUT5_global_annual.csv",   # your file
          usecols=["Time", "Anomaly (deg C)"],  # only the two columns we need
          header=0)                             # first row *is* the header
         .rename(columns={"Time": "year",
                          "Anomaly (deg C)": "gmst_obs"})
         .assign(year=lambda d: d.year.astype(int))
         .query("year >= 1850"))



# ── 3-b  Zanna global OHC (works with your file) ────────────────────────────
import pandas as pd, xarray as xr, pathlib

RAW = pathlib.Path("data/raw")   # adjust if your folders differ

ds = xr.open_dataset(RAW / "OHC_GF_1870_2018.nc")

# 1. Grab the 0–2000 m series
ohc_series = ds["OHC_2000m"].values          # J (10²² J) anomalies
years      = ds["time"].values.astype(int)   # years as ints, e.g. 1870

# 2. Build a tidy DataFrame
ohc = (pd.DataFrame({"year": years, "OHC": ohc_series})
         .query("year >= 1850"))             # keep common window


# ── 3-c  Total ERF  ──────────────────────────────────────────────────────
years, vals = [], []
with open(RAW / "Fi_net_Miller_et_al14_upd.txt") as f:
    for line in f:
        if line.lstrip().startswith("#"):
            continue                    # skip comments
        tokens = line.strip().split()
        if len(tokens) < 2:
            continue
        try:
            yr  = int(tokens[0])
            val = float(tokens[1])
        except ValueError:
            continue                    # skip footer or stray text
        years.append(yr); vals.append(val)

forc = (pd.DataFrame({"year": years, "erf_wm2": vals})
          .query("year >= 1850")
          .sort_values("year"))


# ── 3-d  Merge & save ──────────────────────────────────────────────────────
# Debug
print("DEBUG had  rows:", len(had))
print("DEBUG ohc  rows:", len(ohc))
print("DEBUG forc rows:", len(forc))


annual = (had.merge(ohc,  on="year", how="left")
              .merge(forc, on="year", how="left")
              .sort_values("year"))

# save to disk  ↓↓↓
annual.to_csv(PROC / "annual_climate_inputs.csv", index=False)

print("Wrote", PROC / "annual_climate_inputs.csv")
