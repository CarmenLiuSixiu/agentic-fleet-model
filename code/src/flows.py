"""Mainline 5-min flows for the route detectors from LargeST (PeMS, CC BY-NC 4.0), Jan-Jun 2017.

LargeST ca_his_raw_2017.h5 is a pandas fixed-format HDF5 table (time x 8600 sensors, veh per 5 min).
We read only the columns we need with h5py, convert to veh/h, and cache them next to the source file.
Also picks boundary sensors that the routes do not contain: US-101 N just upstream of the SR-87 junction
(the major on-ramp of route B), so its inflow is measured rather than inferred.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from data import load_meta, route_sensors, load_speeds

LST_DIR = "/Users/carmenliu/Desktop/CGT/CGT-DB/Traffic/LargeST/"
H5 = LST_DIR + "ca_his_raw_2017.h5"
CACHE = LST_DIR + "route_flows_2017H1.pkl"


def load_lst_meta() -> pd.DataFrame:
    return pd.read_csv(LST_DIR + "ca_meta.csv")


def boundary_sensors(n: int = 3) -> list[int]:
    """US-101 N mainline sensors just south (upstream) of the SR-87 junction, nearest first."""
    pb = load_meta()
    g = pb[(pb.Fwy == 101) & (pb.Dir == "N") & (pb.Type == "ML") & (pb.Abs_PM < 389.2) & (pb.Abs_PM > 385.0)]
    lst = set(load_lst_meta().ID)
    g = g[g.sensor_id.isin(lst)].sort_values("Abs_PM", ascending=False)
    return [int(s) for s in g.sensor_id.values[:n]]


def _read_columns(ids: list[int]) -> pd.DataFrame:
    import h5py
    with h5py.File(H5, "r") as f:
        key = list(f.keys())[0]
        g = f[key]
        items = g["block0_items"][:].astype(np.int64)
        pos = {int(s): i for i, s in enumerate(items)}
        cols = sorted(pos[s] for s in ids if s in pos)
        vals = g["block0_values"][:, cols]                     # h5py needs increasing indices
        idx = pd.to_datetime(g["axis1"][:])
    names = [int(items[c]) for c in cols]
    return pd.DataFrame(vals, index=idx, columns=names)


def route_flows(refresh: bool = False) -> pd.DataFrame:
    """Flows in veh/h for all route detectors plus boundary sensors, indexed like PEMS-BAY (Jan-Jun 2017)."""
    if os.path.exists(CACHE) and not refresh:
        return pd.read_pickle(CACHE)
    pb = load_meta()
    ids = [int(s) for r in "AB" for s in route_sensors(pb, r).sensor_id] + boundary_sensors()
    df = _read_columns(ids)
    df = df.loc["2017-01-01":"2017-06-30 23:55"] * 12.0
    df.columns = df.columns.astype(str)
    df.to_pickle(CACHE)
    return df


def check(df: pd.DataFrame) -> None:
    sp = load_speeds()
    common = df.index.intersection(sp.index)
    print("flow rows", len(df), "aligned with PEMS-BAY rows", len(common), "of", len(sp))
    print("missing share per sensor (median, max):", float(df.isna().mean().median()), float(df.isna().mean().max()))
    print("zero share per sensor (median, max):", float((df == 0).mean().median()), float((df == 0).mean().max()))


if __name__ == "__main__":
    d = route_flows(refresh="--refresh" in sys.argv)
    check(d)
    print(d.describe().T[["mean", "50%", "max"]].round(0).head(12))
