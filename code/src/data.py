"""PEMS-BAY loading and route geometry for the SR-85 / SR-87 + US-101 northbound pair.

Route A: SR-85 N from the SR-87 interchange (Santa Teresa, PM 5.18) to US-101 (Moffett, PM 23.89).
Route B: SR-87 N (PM 0.45 to 9.04), then US-101 N from the SR-87 junction (PM 389.23) to Moffett (PM 397.85).
Both routes connect south San Jose to the US-101 / SR-85 interchange in Mountain View.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

DATA_DIR = "/Users/carmenliu/Desktop/CGT/CGT-DB/Traffic/PEMS-BAY/"
MI_TO_KM = 1.609344
HOLIDAYS = pd.to_datetime(["2017-01-02", "2017-01-16", "2017-02-20", "2017-05-29",
                          "2026-01-01", "2026-01-19", "2026-02-16", "2026-05-25"])

# (freeway, direction, Abs_PM start, Abs_PM end) per route, in travel order
ROUTES = {
    "A": [(85, "N", 5.0, 23.95)],
    "B": [(87, "N", 0.0, 9.1), (101, "N", 389.2, 397.9)],
}


def load_meta() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR + "PEMS-BAY-META.csv")


def load_speeds() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR + "PEMS-BAY.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(int)
    return df * MI_TO_KM  # km/h


def route_sensors(meta: pd.DataFrame, route: str) -> pd.DataFrame:
    """Sensors along a route with cumulative distance x_km from the route start."""
    parts, offset = [], 0.0
    for fwy, d, pm0, pm1 in ROUTES[route]:
        g = meta[(meta.Fwy == fwy) & (meta.Dir == d) & (meta.Abs_PM >= pm0) & (meta.Abs_PM <= pm1)]
        g = g[g.Type == "ML"].sort_values("Abs_PM").copy()
        g["x_km"] = offset + (g.Abs_PM - pm0) * MI_TO_KM
        g["seg"] = f"{fwy}{d}"
        parts.append(g)
        offset += (pm1 - pm0) * MI_TO_KM
    out = pd.concat(parts, ignore_index=True)
    out["is_ramp"] = out.Name.str.contains(r"rm-n", case=False, regex=True)
    return out


def route_length_km(route: str) -> float:
    return sum((pm1 - pm0) * MI_TO_KM for _, _, pm0, pm1 in ROUTES[route])


def weekdays(index: pd.DatetimeIndex) -> np.ndarray:
    days = pd.DatetimeIndex(sorted(set(index.normalize())))
    return np.array([d for d in days if d.weekday() < 5 and d not in HOLIDAYS])


def day_window(speeds: pd.DataFrame, day, t0="05:00", t1="11:00") -> pd.DataFrame:
    start = pd.Timestamp(f"{pd.Timestamp(day).date()} {t0}")
    end = pd.Timestamp(f"{pd.Timestamp(day).date()} {t1}")
    return speeds.loc[(speeds.index >= start) & (speeds.index < end)]


def instantaneous_tt_min(sp: pd.DataFrame, sens: pd.DataFrame, length_km: float) -> np.ndarray:
    """Route travel time (min) from spot speeds, each sensor covering half the gap to its neighbours."""
    x = sens.x_km.values
    edges = np.concatenate([[0.0], (x[1:] + x[:-1]) / 2, [length_km]])
    w = np.diff(edges)
    v = np.clip(sp[sens.sensor_id.values].values, 5.0, None)
    return (w[None, :] / v).sum(1) * 60.0
