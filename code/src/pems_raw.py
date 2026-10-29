"""Raw PeMS Clearinghouse 'Station 5-Minute' files (District 4), parsed for the corridor stations.

File layout (no header): timestamp, station, district, freeway, direction, lane type, station length, samples,
% observed, total flow (veh / 5 min), avg occupancy (fraction), avg speed (mph), then 5 fields per lane.
'% observed' = share of the station's lane points that were actually measured; 0 means PeMS imputed the row.
Outputs per-station frames on a 5-min index: speed (km/h), flow (veh/h), occupancy, pct_observed; cached as pkl.
"""
from __future__ import annotations
import os, re, sys, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from data import MI_TO_KM

RAW_DIR = "/Users/carmenliu/Desktop/CGT/CGT-DB/Traffic/PeMS_raw/d04/station_5min/"
CACHE_DIR = "/Users/carmenliu/Desktop/CGT/CGT-DB/Traffic/PeMS_raw/d04/cache/"
COLS = ["ts", "station", "district", "fwy", "dir", "ltype", "slen", "samples", "pct_obs", "flow", "occ", "speed"]
FIELDS = ("speed", "flow", "occ", "pct_obs")


def files(year: int, month: int) -> list[str]:
    return sorted(glob.glob(os.path.join(RAW_DIR, f"d04_text_station_5min_{year}_{month:02d}_*.txt.gz")))


def _read_day(path: str, ids: set[int]) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, usecols=range(12), names=COLS)
    df = df[df.station.isin(ids)]
    df["ts"] = pd.to_datetime(df.ts, format="%m/%d/%Y %H:%M:%S")
    return df


def load_month(year: int, month: int, ids, tag: str, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Dict field -> DataFrame (time x station). Speeds converted to km/h, flows to veh/h."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{tag}_{year}_{month:02d}.pkl")
    if os.path.exists(cache) and not refresh:
        return pd.read_pickle(cache)
    ids = set(int(i) for i in ids)
    parts = [_read_day(f, ids) for f in files(year, month)]
    df = pd.concat(parts, ignore_index=True)
    out = {}
    for f in FIELDS:
        w = df.pivot_table(index="ts", columns="station", values=f, aggfunc="first")
        out[f] = w
    out["speed"] = out["speed"] * MI_TO_KM
    out["flow"] = out["flow"] * 12.0
    out["meta"] = df.groupby("station").agg(fwy=("fwy", "first"), dir=("dir", "first"), ltype=("ltype", "first"))
    out["days"] = sorted({re.search(r"(\d{4}_\d{2}_\d{2})", f).group(1) for f in files(year, month)})
    pd.to_pickle(out, cache)
    return out


def observed_mask(d: dict, min_pct: float = 50.0) -> pd.DataFrame:
    """True where the 5-min row was measured (not imputed by PeMS)."""
    return d["pct_obs"] >= min_pct


if __name__ == "__main__":
    from data import load_meta, route_sensors
    from flows import boundary_sensors
    m = load_meta()
    ids = [int(s) for r in "AB" for s in route_sensors(m, r).sensor_id] + boundary_sensors()
    d = load_month(2026, 4, ids, "corridor", refresh="--refresh" in sys.argv)
    print("days:", len(d["days"]), "| index:", d["speed"].index.min(), "->", d["speed"].index.max(), "| stations:", d["speed"].shape[1])
    obs = observed_mask(d)
    print("share of rows observed (median over stations):", round(float(obs.mean().median()), 3))
