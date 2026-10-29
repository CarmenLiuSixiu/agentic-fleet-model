"""2017 (PEMS-BAY) vs April 2026 (raw PeMS) on the same corridor: data quality and route travel times.

Travel times are computed two ways for 2026: from all stations as published (imputed rows included, as in
PEMS-BAY) and from measured rows only (imputed rows dropped and filled by linear interpolation along the route).
Writes results/compare_2026.json and figs/fig1b_2017_vs_2026.png.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import load_meta, load_speeds, route_sensors, route_length_km, weekdays, day_window, instantaneous_tt_min
from flows import boundary_sensors
from pems_raw import load_month, observed_mask

ROOT = os.path.join(os.path.dirname(__file__), "..")
FALLBACK = [0, 0]   # slots that fell back to published values, slots total


def tt_days(sp, sens, L, days, mask=None):
    out = []
    for d in days:
        w = day_window(sp, d)
        if len(w) != 72:
            continue
        w = w[sens.sensor_id.values].copy()
        if mask is not None:
            m = day_window(mask, d)[sens.sensor_id.values].values
            pub = w.values.copy()
            v = pub.copy(); v[~m] = np.nan
            x = sens.x_km.values
            for t in range(72):
                ok = np.isfinite(v[t])
                if ok.sum() >= 2:
                    v[t] = np.interp(x, x[ok], v[t, ok])
                else:                                   # too few measured stations: fall back to published values
                    v[t] = pub[t]; FALLBACK[0] += 1
                FALLBACK[1] += 1
            w = pd.DataFrame(v, index=w.index, columns=w.columns)
        if w.isna().any().any():
            w = w.ffill().bfill()
        out.append(instantaneous_tt_min(w, sens, L))
    return np.array(out)


def summarize(tt):
    diff = tt["A"] - tt["B"]
    times = pd.date_range("2000-01-01 05:00", periods=72, freq="5min").strftime("%H:%M")
    pk = 12 + 12 * 1  # slots 06:00-10:00 = 12..60
    flips = np.mean([np.any(np.diff(np.sign(d[12:60])) != 0) for d in diff])
    res = {"n_days": int(len(diff)), "B_faster_share": float((diff > 0).mean()), "flip_days_06_10": float(flips)}
    for r in "AB":
        med = np.median(tt[r], 0)
        res[f"peak_median_tt_{r}"] = [round(float(med.max()), 1), times[int(med.argmax())]]
        res[f"freeflow_tt_{r}"] = round(float(np.percentile(tt[r][:, :6], 10)), 1)
    return res


def main():
    meta = load_meta()
    sens = {r: route_sensors(meta, r) for r in "AB"}
    L = {r: route_length_km(r) for r in "AB"}
    # 2017
    sp17 = load_speeds()
    d17 = weekdays(sp17.index)
    tt17 = {r: tt_days(sp17, sens[r], L[r], d17) for r in "AB"}
    # 2026-04
    ids = [int(s) for r in "AB" for s in sens[r].sensor_id] + boundary_sensors()
    raw = load_month(2026, 4, ids, "corridor")
    sp26 = raw["speed"]; obs = observed_mask(raw)
    d26 = weekdays(sp26.index)
    sens26 = {r: s[s.sensor_id.isin(sp26.columns)].reset_index(drop=True) for r, s in sens.items()}
    tt26_all = {r: tt_days(sp26, sens26[r], L[r], d26) for r in "AB"}
    tt26_obs = {r: tt_days(sp26, sens26[r], L[r], d26, mask=obs) for r in "AB"}
    am = (sp26.index.hour >= 5) & (sp26.index.hour < 11)
    q = {}
    for r in "AB":
        s = sens[r].sensor_id.values
        present = [i for i in s if i in sp26.columns]
        share = obs.loc[am, present].mean()
        q[r] = {"stations_2017": int(len(s)), "present_2026": len(present),
                "measured_ge_50pct_of_AM_rows": int((share >= 0.5).sum()), "never_measured": int((share == 0).sum())}
    res = {"2017": summarize(tt17), "2026_04_as_published": summarize(tt26_all), "2026_04_measured_only": summarize(tt26_obs),
           "station_quality_2026_04": q, "measured_only_fallback_slots": FALLBACK}
    json.dump(res, open(os.path.join(ROOT, "results", "compare_2026.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))

    x = np.arange(72); times = pd.date_range("2000-01-01 05:00", periods=72, freq="5min").strftime("%H:%M")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True, layout="constrained")
    for k, (name, tt) in enumerate([("2017 Jan-Jun (PEMS-BAY)", tt17), ("2026 Apr (raw PeMS, measured rows only)", tt26_obs)]):
        for r, c in [("A", "tab:blue"), ("B", "tab:orange")]:
            med = np.median(tt[r], 0)
            ax[k].plot(x, med, c=c, label={"A": "A: SR-85 N", "B": "B: SR-87 N + US-101 N"}[r])
            ax[k].fill_between(x, np.percentile(tt[r], 25, 0), np.percentile(tt[r], 75, 0), color=c, alpha=0.2)
        ax[k].set_title(f"{name}, {len(tt['A'])} weekdays"); ax[k].set_xticks(range(0, 73, 12))
        ax[k].set_xticklabels(times[::12].tolist() + ["11:00"]); ax[k].grid(alpha=0.3)
    ax[0].set_ylabel("route travel time (min), median and IQR"); ax[0].legend()
    fig.savefig(os.path.join(ROOT, "figs", "fig1b_2017_vs_2026.png"), dpi=150)


if __name__ == "__main__":
    main()
