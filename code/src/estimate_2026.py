"""Short-term forecasting on raw PeMS April-June 2026 test days (the operator's actual use of the world model).

At each decision time (06:30-09:30 every 30 min) on the 17 test days of calibrate_daily_2026 (every third usable
weekday), moving-horizon estimation fits the state from the last 30 min of measured detector speeds (imputed rows are
missing, not zeros), then the model is rolled forward 60 min. Scored at +15/+30/+60 min on measured rows of valid
detectors (all valid detectors are observed in operation; nothing is hidden here).
Baselines: persistence (last measured 5-min value of each detector in the window) and the mean profile of the
training days. Route travel times use all valid detectors, linearly interpolated along the route.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, torch
torch.set_num_threads(4)
import ctm
from estimate import rollout, prior_state, expand, W_MIN, SLOT0_STEP, HORIZONS
from calibrate_daily_2026 import load_days, route_tt
from calibrate_flow import interp_nan

ROOT = os.path.join(os.path.dirname(__file__), "..")


def mhe_masked(p, net, t_dec, obsV, mask, cell_of, iters=60, lr=0.08, smooth=2.0):
    """As estimate.mhe, but obsV may contain NaN (unmeasured); mask (B, W/5, n) selects measured entries."""
    B = obsV.shape[0]
    W = W_MIN * 2
    st0 = expand(prior_state(p, net, t_dec - W), B)
    Y = torch.nan_to_num(obsV)
    x = {"logm": torch.zeros(B, net.M), "s0": torch.zeros(B), "sA": torch.zeros(B), "sB": torch.zeros(B),
         "bias": torch.zeros(B), "logq": torch.zeros(B)}
    x = {k: v.requires_grad_(True) for k, v in x.items()}
    opt = torch.optim.Adam(x.values(), lr=lr)
    def state():
        return {"n": st0["n"] * torch.exp(x["logm"])[:, None, :], "qo": st0["qo"] * torch.exp(x["logq"])[:, None],
                "qr": st0["qr"] * torch.exp(x["logq"])[:, None, None], "qh": st0["qh"]}
    for _ in range(iters):
        opt.zero_grad()
        _, V, _ = rollout(p, net, state(), t_dec - W, W, (x["s0"], x["sA"], x["sB"]), x["bias"], smooth=smooth)
        r = ((V[:, :, cell_of] - Y) / 10.0) ** 2
        loss = r[mask].mean() + 0.05 * (x["logm"] ** 2).mean() + 0.5 * (x["logm"].diff(dim=1) ** 2).mean() \
            + 0.5 * sum((x[k] ** 2).mean() for k in ("s0", "sA", "sB", "bias", "logq"))
        loss.backward()
        opt.step()
        with torch.no_grad():
            x["logm"].clamp_(-2.0, 2.0)
            for k in ("s0", "sA", "sB", "bias", "logq"):
                x[k].clamp_(-1.0, 1.0)
    with torch.no_grad():
        st, _, _ = rollout(p, net, state(), t_dec - W, W, (x["s0"], x["sA"], x["sB"]), x["bias"])
    return st, {k: v.detach() for k, v in x.items()}


def main(params="results/wm_params_2026i.pt", tag="_2026i", iters=60):
    torch.manual_seed(0)
    p = {k: v.float() for k, v in torch.load(os.path.join(ROOT, params)).items()}
    net, sens, days, V, Q, M, B = load_days()
    ids = sens.sensor_id.values
    cell_of = torch.tensor([net.sensor_cell[int(s)] for s in ids])
    import split
    tr_i, tu_i, te_i = split.indices(days)
    valid = M[tr_i].mean((0, 1)) >= 0.5
    with np.errstate(all="ignore"):
        hist = np.nanmean(V[tr_i], 0)
    Y = V[te_i]                                        # (D, 72, n) measured speeds, NaN elsewhere
    rows, t0 = [], time.time()
    for hhmm in ("06:30", "07:00", "07:30", "08:00", "08:30", "09:00", "09:30"):
        h, m = map(int, hhmm.split(":"))
        slot = (h * 60 + m - 300) // 5
        t_dec = SLOT0_STEP + slot * ctm.STEPS_PER_5MIN
        w = W_MIN // 5
        win = Y[:, slot - w:slot].copy()
        win[:, :, ~valid] = np.nan
        mask = torch.tensor(np.isfinite(win))
        st, x = mhe_masked(p, net, t_dec, torch.tensor(win, dtype=torch.float32), mask, cell_of, iters=iters)
        with torch.no_grad():
            _, Vf, _ = rollout(p, net, st, t_dec, 60 * 2, (x["s0"], x["sA"], x["sB"]), x["bias"])
        Vf = Vf[:, :, cell_of].numpy()
        last = pd.DataFrame(win.reshape(-1, win.shape[2])).groupby(np.repeat(np.arange(win.shape[0]), w)).last().values
        for hz in HORIZONS:
            k = hz // 5 - 1
            truth = Y[:, slot + k]
            preds = {"world_model": Vf[:, k], "persistence": last, "historical": np.broadcast_to(hist[slot + k], truth.shape)}
            for d in range(truth.shape[0]):
                tt_true = route_tt(interp_nan(truth[d][None], sens, valid), sens)[0]
                if not np.isfinite(tt_true).all():
                    continue
                for name, pr in preds.items():
                    pr_d = interp_nan(np.asarray(pr[d], dtype=float)[None], sens, valid)
                    tt_p = route_tt(pr_d, sens)[0]
                    ok = valid & np.isfinite(truth[d]) & np.isfinite(pr[d])
                    gap = float(abs(tt_true[0] - tt_true[1]))
                    wrong = tt_p.argmin() != tt_true.argmin()
                    rows.append({"time": hhmm, "horizon_min": hz, "method": name, "day": str(days[te_i[d]].date()),
                                 "speed_mae": float(np.abs(pr[d] - truth[d])[ok].mean()) if ok.any() else np.nan,
                                 "tt_mae_min": float(np.abs(tt_p - tt_true).mean()),
                                 "faster_route_correct": float(not wrong),
                                 "true_gap_min": gap, "choice_loss_min": gap if wrong else 0.0,
                                 "ttA_true": float(tt_true[0]), "ttB_true": float(tt_true[1]),
                                 "ttA_pred": float(tt_p[0]), "ttB_pred": float(tt_p[1])})
        print(hhmm, f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    import split
    tag = tag + split.tag()
    df.to_csv(os.path.join(ROOT, "results", f"nowcast{tag}.csv"), index=False)
    summ = df.groupby(["horizon_min", "method"])[["speed_mae", "tt_mae_min", "faster_route_correct", "choice_loss_min"]].mean().round(3)
    dec = df[df.true_gap_min > 2.0]       # decision-relevant cases: routes differ by more than 2 min
    summ_dec = dec.groupby(["horizon_min", "method"])[["faster_route_correct", "choice_loss_min"]].mean().round(3)
    summ_dec["n_cases"] = dec.groupby(["horizon_min", "method"]).size()
    print("decision-relevant cases (|true gap| > 2 min):"); print(summ_dec.to_string())
    # paired by day and time: share of (day, time) cases where the world model beats each baseline on route TT
    piv = df.pivot_table(index=["horizon_min", "day", "time"], columns="method", values="tt_mae_min")
    wins = {int(hz): {b: float((g["world_model"] < g[b]).mean()) for b in ("persistence", "historical")}
            for hz, g in piv.groupby(level=0)}
    print(summ.to_string()); print("world model better on route TT (share of day-time cases):", wins)
    json.dump({"params": params, "n_test_days": len(te_i), "summary": summ.reset_index().to_dict(orient="records"),
               "wm_win_share_tt": wins, "decision_relevant": summ_dec.reset_index().to_dict(orient="records")},
              open(os.path.join(ROOT, "results", f"nowcast{tag}.json"), "w"), indent=2)


if __name__ == "__main__":
    a = sys.argv
    main(a[1] if len(a) > 1 else "results/wm_params_2026i.pt", a[2] if len(a) > 2 else "_2026i",
         int(a[3]) if len(a) > 3 else 60)
