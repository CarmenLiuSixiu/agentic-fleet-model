"""Figure 3: operator ladder, vehicle-agency ladder, and agency x autonomy (paired by day vs A0)."""
import os, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
df = pd.read_csv(os.path.join(ROOT, "results", "closed_loop_runs.csv"))
base = df[(df.agent == "A0")].set_index("day")


def pct(sub, col="tts_vh"):
    s = sub.set_index("day")[col]
    return (s - base[col]) / base[col] * 100


def boot(x, n=4000):
    rng = np.random.default_rng(0); x = np.asarray(x)
    b = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), np.percentile(b, 5), np.percentile(b, 95)


arms = [("A0", 0, 0.0, "A0 no control"), ("A1", 0, 0.0, "A1 ALINEA"), ("A2", 0, 0.0, "A2 world-model MPC"),
        ("A2", 1, 0.0, "A2 + L1 self-interested"), ("A2", 2, 0.0, "A2 + L2 obedient"), ("A2", 3, 0.0, "A2 + L3 negotiating"),
        ("A2", 4, 0.0, "A2 + L4 negotiating + Caltrain offer"),
        ("A2", 0, 0.3, "A2 + autonomy only"), ("A2", 3, 0.3, "A2 + L3 + autonomy")]
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.4, 1, 1]})
y = np.arange(len(arms))[::-1]
for k, (inc, c) in enumerate([(False, "tab:green"), (True, "tab:red")]):
    for i, (a, lv, kp, name) in enumerate(arms):
        sub = df[(df.agent == a) & (df.level == lv) & (df.kappa == kp)]
        d = pct(sub)[sub.set_index("day").incident == inc]
        m, lo, hi = boot(d.values)
        ax[0].errorbar(m, y[i] + (0.15 if inc else -0.15), xerr=[[m - lo], [hi - m]], fmt="o", c=c,
                       label=("incident days" if inc else "normal days") if i == 0 else None)
ax[0].axvline(0, c="k", lw=0.8); ax[0].set_yticks(y); ax[0].set_yticklabels([a[3] for a in arms])
ax[0].set_xlabel("total time spent vs no control (%), mean and 90% bootstrap CI over days"); ax[0].legend(loc="upper left")
ax[0].set_title("Operator and vehicle agency ladders (agentic share 30%)")

lv_names = ["L0", "L1", "L2", "L3", "L4"]
w = [df[(df.agent == "A2") & (df.level == lv) & (df.kappa == 0)].worst_ramp_wait_min.mean() for lv in range(5)]
sp = [df[(df.agent == "A2") & (df.level == lv) & (df.kappa == 0)].spend_usd.mean() for lv in range(5)]
ax[1].bar(lv_names, w, color="tab:purple", alpha=0.8); ax[1].set_ylabel("worst controlled ramp: mean wait (min)")
ax1 = ax[1].twinx(); ax1.plot(lv_names, sp, "ko--"); ax1.set_ylabel("incentive spend per day ($, dashed)")
ax[1].set_title("Who bears the cost (operator A2)")

summ = json.load(open(os.path.join(ROOT, "results", "closed_loop_summary.json")))["x3"]
keys = [("agency_only", "agency\n(L3)"), ("autonomy_only", "autonomy\n(kappa=0.3)"), ("both", "both"), ("interaction", "interaction")]
for i, (k, lab) in enumerate(keys):
    m, lo, hi = summ[k]
    ax[2].bar(i, m, color="tab:blue" if k != "interaction" else "tab:orange", alpha=0.85)
    ax[2].errorbar(i, m, yerr=[[m - lo], [hi - m]], c="k", capsize=4)
ax[2].axhline(0, c="k", lw=0.8); ax[2].set_xticks(range(4)); ax[2].set_xticklabels([k[1] for k in keys])
ax[2].set_ylabel("TTS change vs A2 + L0 (%)"); ax[2].set_title("Agency x autonomy, paired by day")
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figs", "fig3_closed_loop.png"), dpi=150)
print("saved")
