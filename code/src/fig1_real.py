"""Figure 1: observed weekday morning conditions on the two routes (PEMS-BAY, Jan-Jun 2017)."""
import sys, json
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import load_meta, load_speeds, route_sensors, route_length_km, weekdays, day_window, instantaneous_tt_min

OUT = "/Users/carmenliu/Desktop/Study2026/AgenticMobility_Yu/code/"
meta, sp = load_meta(), load_speeds()
days = weekdays(sp.index)
sens = {r: route_sensors(meta, r) for r in "AB"}
L = {r: route_length_km(r) for r in "AB"}

tt = {r: [] for r in "AB"}
stack = {r: [] for r in "AB"}
for d in days:
    w = day_window(sp, d)
    if len(w) != 72:
        continue
    for r in "AB":
        tt[r].append(instantaneous_tt_min(w, sens[r], L[r]))
        stack[r].append(w[sens[r].sensor_id.values].values)
tt = {r: np.array(v) for r, v in tt.items()}
mean_sp = {r: np.mean(stack[r], 0) for r in "AB"}
times = pd.date_range("2017-01-03 05:00", periods=72, freq="5min").strftime("%H:%M")
diff = tt["A"] - tt["B"]  # >0 means route B (87 + 101) faster

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.15]}, layout="constrained")
for k, r in enumerate("AB"):
    im = ax[k].imshow(mean_sp[r].T, aspect="auto", origin="lower", cmap="RdYlGn", vmin=20, vmax=110,
                      extent=[0, 72, 0, len(sens[r])])
    ax[k].set_title({"A": "Route A: SR-85 N", "B": "Route B: SR-87 N + US-101 N"}[r] + "  (mean weekday)")
    ax[k].set_xticks(range(0, 73, 12)); ax[k].set_xticklabels(times[::12].tolist() + ["11:00"])
    ax[k].set_ylabel("detector index (upstream to downstream)")
fig.colorbar(im, ax=ax[1], label="speed (km/h)", fraction=0.05, pad=0.02)
x = np.arange(72)
for r, c in [("A", "tab:blue"), ("B", "tab:orange")]:
    med, lo, hi = np.median(tt[r], 0), np.percentile(tt[r], 25, 0), np.percentile(tt[r], 75, 0)
    ax[2].plot(x, med, c=c, label={"A": "A: SR-85", "B": "B: SR-87 + US-101"}[r]); ax[2].fill_between(x, lo, hi, color=c, alpha=0.2)
ax[2].set_xticks(range(0, 73, 12)); ax[2].set_xticklabels(times[::12].tolist() + ["11:00"])
ax[2].set_ylabel("route travel time (min), median and IQR over weekdays"); ax[2].legend(loc="upper left")
ax2 = ax[2].twinx(); ax2.plot(x, (diff > 0).mean(0), c="gray", ls="--", lw=1); ax2.set_ylim(0, 1)
ax2.set_ylabel("share of days route B is faster (dashed)")
ax[2].set_title(f"Route travel times, {tt['A'].shape[0]} weekdays")
fig.savefig(OUT + "figs/fig1_real_routes.png", dpi=160)

summary = {
    "n_days": int(tt["A"].shape[0]),
    "free_flow_tt_min": {r: float(np.percentile(tt[r][:, :6], 10)) for r in "AB"},
    "peak_median_tt_min": {r: float(np.median(tt[r], 0).max()) for r in "AB"},
    "peak_time": {r: times[int(np.median(tt[r], 0).argmax())] for r in "AB"},
    "share_slots_B_faster": float((diff > 0).mean()),
    "median_abs_diff_min_0700_0900": float(np.median(np.abs(diff[:, 24:48]))),
    "share_days_sign_flips_0600_1000": float(np.mean([(np.sign(d[12:60]) != np.sign(d[12])).any() for d in diff])),
}
json.dump(summary, open(OUT + "results/fig1_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
