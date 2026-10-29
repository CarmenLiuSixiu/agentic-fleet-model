"""Closed-loop experiments X1 (operator ladder), X2 (vehicle agency ladder), X3 (agency x autonomy).

Truth = calibrated CTM plus day-specific demand noise, capacity drop after breakdown, incidents on
half of the days, a true passenger value-of-time distribution, and day-specific human route bias.
World model used by the MPC operator = calibrated CTM with mean demand, no capacity drop, no incident
knowledge, and a misspecified value-of-time belief. The operator observes the true current state.
Common random numbers: every arm sees the same 10 days.
"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, torch
from multiprocessing import Pool
import ctm, agentic
from network import build_default_net

ROOT = os.path.join(os.path.dirname(__file__), "..")
DAYS = [(s, s > 5) for s in range(1, 11)]   # seeds 6-10 are incident days
P_AG = 0.3

ARMS = (
    [("X1", a, 0, 0.0) for a in ("A0", "A1", "A2")]
    + [("X2", "A2", lv, 0.0) for lv in (1, 2, 3)]
    + [("X3", "A2", lv, 0.3) for lv in (0, 3)]
    + [("X4", "A2", 4, 0.0)]
)


def controlled_ramps(net, p, per_route=3):
    """Regular on-ramps whose cell is most congested in the world model's no-control mean day (07:00-09:30)."""
    with torch.no_grad():
        st = ctm.init_state(net, 1)
        meters = net.ramp_cap.expand(1, -1)
        rp = lambda a, b: torch.sigmoid(p["b0"] + p["b1"] * (a - b)).reshape(1, 1).expand(1, 2)
        kc = ctm.Q_LANE / 105.0 * net.lanes * net.L
        occ, n = torch.zeros(net.M), 0
        for t in range(ctm.N_STEPS):
            st, _ = ctm.step(st, p, net, t, meters, rp)
            if (7 * 60 - ctm.T0_MIN) * 2 <= t < (9 * 60 + 30 - ctm.T0_MIN) * 2:
                occ += st["n"][0].sum(0) / kc; n += 1
        occ /= n
    idx = []
    for r in (1, 2):
        cand = [i for i in range(net.R) if int(net.ramp_route[i]) == r and not bool(net.ramp_major[i])]
        cand.sort(key=lambda i: -float(occ[net.ramp_cell[i]]))
        idx += cand[:per_route]
    return torch.tensor(sorted(idx))


def _job(args):
    params_path, exp, agent, level, kappa, seed, incident, truth_model = args
    torch.set_num_threads(1)
    net, _, _ = build_default_net()
    p = torch.load(params_path)
    ctrl = controlled_ramps(net, p)
    r = agentic.run_day(net, p, p, agent, level, P_AG, kappa, seed, incident, ctrl, truth_model=truth_model)
    r["exp"] = exp; r["truth"] = truth_model
    return r


def boot_ci(x, n=4000, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    bs = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


def summarize(df):
    out = {}
    base = df[(df.agent == "A0") & (df.level == 0) & (df.kappa == 0)].set_index("day")
    def rel(sub, col="tts_vh"):
        s = sub.set_index("day")[col]
        return (s - base[col]) / base[col] * 100
    rows = []
    for (agent, level, kappa), sub in df.groupby(["agent", "level", "kappa"]):
        d = rel(sub)
        for label, mask in [("all", slice(None)), ("normal", ~sub.set_index("day").incident), ("incident", sub.set_index("day").incident)]:
            vals = d[mask] if label != "all" else d
            m, lo, hi = boot_ci(vals.values)
            rows.append({"agent": agent, "level": level, "kappa": kappa, "days": label, "tts_change_pct": m,
                         "ci90_lo": lo, "ci90_hi": hi})
    out["tts_vs_A0"] = rows
    agg = df.groupby(["agent", "level", "kappa"]).agg(
        tts_vh=("tts_vh", "mean"), through_avg_min=("through_avg_min", "mean"),
        worst_ramp_wait_min=("worst_ramp_wait_min", "mean"), max_ramp_queue=("max_ramp_queue", "mean"),
        spill_vh=("spill_vh", "mean"), spend_usd=("spend_usd", "mean"), vkt_agentic=("vkt_agentic", "mean"),
        decision_sec=("decision_sec", "mean")).reset_index()
    out["means"] = agg.to_dict(orient="records")
    # X3 interaction: (k.3,L3) - (k.3,L0) - (k0,L3) + (k0,L0), paired by day, % of A2-L0-k0 TTS
    def tts(level, kappa):
        return df[(df.agent == "A2") & (df.level == level) & (df.kappa == kappa)].set_index("day").tts_vh
    b = tts(0, 0.0)
    eff_agency = (tts(3, 0.0) - b) / b * 100
    eff_autonomy = (tts(0, 0.3) - b) / b * 100
    eff_both = (tts(3, 0.3) - b) / b * 100
    inter = eff_both - eff_agency - eff_autonomy
    out["x3"] = {k: boot_ci(v.values) for k, v in
                 [("agency_only", eff_agency), ("autonomy_only", eff_autonomy), ("both", eff_both), ("interaction", inter)]}
    return out


def main(params_path, workers=4, truth_model="ctm", tag=""):
    jobs = [(params_path, exp, a, lv, k, s, inc, truth_model) for (exp, a, lv, k), (s, inc) in itertools.product(ARMS, DAYS)]
    with Pool(workers) as pool:
        res = pool.map(_job, jobs)
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(ROOT, "results", f"closed_loop_runs{tag}.csv"), index=False)
    summ = summarize(df)
    json.dump(summ, open(os.path.join(ROOT, "results", f"closed_loop_summary{tag}.json"), "w"), indent=2)
    print(json.dumps(summ, indent=1)[:4000])


if __name__ == "__main__":
    a = sys.argv
    main(a[1] if len(a) > 1 else os.path.join(ROOT, "results", "wm_params.pt"), workers=8,
         truth_model=a[2] if len(a) > 2 else "ctm", tag=a[3] if len(a) > 3 else "")
