"""Figure 2: observed vs world-model speeds (test period, May-Jun 2017) and hidden-detector error."""
import sys, os, json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
tag = sys.argv[1] if len(sys.argv) > 1 else ""
a = np.load(os.path.join(ROOT, "results", f"calibration_arrays{tag}.npz"))
res = json.load(open(os.path.join(ROOT, "results", f"calibration{tag}.json")))["result"]
V, Yte, hidden, route = a["V"], a["Yte"], a["hidden"], a["route"]

fig, ax = plt.subplots(1, 5, figsize=(19, 4.0), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.1]})
ticks = range(0, 73, 12); labels = ["05:00", "06:00", "07:00", "08:00", "09:00", "10:00", "11:00"]
for k, (r, name) in enumerate([(1, "Route A: SR-85 N"), (2, "Route B: SR-87 N + US-101 N")]):
    m = route == r
    for j, (M, kind) in enumerate([(Yte, "observed"), (V, "world model")]):
        axx = ax[2 * k + j]
        im = axx.imshow(M[:, m].T, aspect="auto", origin="lower", cmap="RdYlGn", vmin=20, vmax=110)
        hid = np.where(hidden[m])[0]
        axx.scatter(np.full(len(hid), 71.5), hid, marker="<", s=12, c="k", clip_on=False)
        axx.set_title(f"{name}\n{kind}, mean weekday May-Jun", fontsize=9)
        axx.set_xticks(list(ticks)); axx.set_xticklabels(labels, fontsize=7)
        if j == 0:
            axx.set_ylabel("detector (upstream to downstream); arrows = hidden", fontsize=8)
bars = [("world model\nvisible", res["wm_mae_visible_test"]), ("world model\nhidden", res["wm_mae_hidden_test"]),
        ("interpolation\nhidden", res["interp_mae_hidden_test"])]
ax[4].bar(range(3), [b[1] for b in bars], color=["tab:blue", "tab:blue", "tab:gray"], alpha=0.85)
ax[4].set_xticks(range(3)); ax[4].set_xticklabels([b[0] for b in bars], fontsize=8)
for i, b in enumerate(bars):
    ax[4].text(i, b[1] + 0.2, f"{b[1]:.1f}", ha="center", fontsize=9)
ax[4].set_ylabel("speed MAE, km/h (test period)"); ax[4].set_title("Detector-only calibration, 30% detectors hidden", fontsize=9)
fig.colorbar(im, ax=ax[:4], fraction=0.015, label="speed (km/h)")
fig.savefig(os.path.join(ROOT, "figs", f"fig2_calibration{tag}.png"), dpi=150, bbox_inches="tight")
print("saved")
