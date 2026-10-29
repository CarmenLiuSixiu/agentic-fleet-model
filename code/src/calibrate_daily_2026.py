"""Day-by-day calibration on raw PeMS April-June 2026 (measured rows only).

Shared across days (road): per-cell capacity and free-flow multipliers, route free-flow speeds, exit fractions,
demand shapes (origin log_d0, per-ramp log_r) and the human route-choice sensitivity. Warm start from the April
mean-day calibration (results/wm_params_2026i.pt).
Per day: 15-min log-multipliers on origin demand and on route-A and route-B ramp demands (3 x N_BLOCKS, smoothed),
a route-choice bias, and the measured US-101 inflow at the SR-87 junction as boundary.
Usable weekdays: >= 30% measured mainline rows 05:00-11:00. Every third usable weekday is a test day. On test days
the road parameters are frozen and only that day's multipliers are fitted, on visible detectors; hidden detectors
(30%, same draw as all other calibrations) are never used.
Baselines on test days, hidden detectors: linear interpolation from that day's visible detectors, and the mean
profile of training days (no same-day information).
"""
import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, torch
torch.set_num_threads(2)
import ctm
from data import weekdays, day_window, instantaneous_tt_min, route_length_km
from network import build_default_net
from flows import boundary_sensors
from pems_raw import load_month, observed_mask
from calibrate_flow import interp_nan, SPILL_Q, SPILL_MAJOR, V_SCALE, Q_SCALE

ROOT = os.path.join(os.path.dirname(__file__), "..")
torch.manual_seed(0)
BND = 402364


def load_days(months=None, cell_km=1.0):
    import split
    months = months or split.months()
    net, sA, sB = build_default_net(cell_km=cell_km)
    sens = pd.concat([sA.assign(route=1), sB.assign(route=2)], ignore_index=True)
    ids = sens.sensor_id.values.astype(int)
    sp, fl, ob = [], [], []
    for m in months:
        d = load_month(2026, m, list(ids) + boundary_sensors(), "corridor")
        sp.append(d["speed"]); fl.append(d["flow"]); ob.append(observed_mask(d))
    sp, fl, ob = (pd.concat(x).sort_index() for x in (sp, fl, ob))
    days, V, Q, M, B = [], [], [], [], []
    for d in weekdays(sp.index):
        w = day_window(sp, d)
        if len(w) != 72:
            continue
        m = day_window(ob.reindex(columns=ids), d).fillna(False).values
        if m.mean() < 0.3:
            continue
        v = day_window(sp.reindex(columns=ids), d).values; q = day_window(fl.reindex(columns=ids), d).values
        b = day_window(fl, d)[BND].where(day_window(ob, d)[BND]).interpolate(limit_direction="both").values
        if not np.isfinite(b).all():
            continue
        days.append(d); V.append(np.where(m, v, np.nan)); Q.append(np.where(m, q, np.nan)); M.append(m); B.append(b)
    return net, sens, days, np.array(V), np.array(Q), np.array(M), np.array(B)


def boundary_blocks(B):
    """(D, 72) 5-min boundary flows -> (D, N_BLOCKS) 15-min blocks, before 05:00 = first block."""
    blk = B.reshape(B.shape[0], 24, 3).mean(2)
    pre = ctm.N_BLOCKS - 24
    return torch.tensor(np.concatenate([np.repeat(blk[:, :1], pre, 1), blk], 1), dtype=torch.float32)


N_INC = 4                     # incident-layer time windows: 05:00-06:30, 06:30-08:00, 08:00-09:30, 09:30-11:00


def simulate(shared, day, net, bnd, smooth):
    """Batched over days; returns speeds and cell outflows (D, 72, M) and the ramp-queue penalty."""
    D = bnd.shape[0]
    st = ctm.init_state(net, D)
    meters = net.ramp_cap.expand(D, -1)
    vf = ctm.vf_cells(shared, net)
    rp = lambda a, b: torch.sigmoid(shared["b0"] + day["bias"] + shared["b1"] * (a - b))[:, None].expand(D, 2)
    majB = int(torch.where(net.ramp_major & (net.ramp_route == 2))[0])
    storage = torch.where(net.ramp_major, torch.tensor(SPILL_MAJOR), torch.tensor(SPILL_Q))
    Vs, Qs, av, aq, qpen = [], [], 0.0, 0.0, 0.0
    for t in range(ctm.N_STEPS):
        blk = min(ctm.N_BLOCKS - 1, t // (2 * ctm.BLOCK_MIN))
        d0 = torch.exp(shared["log_d0"][blk] + day["z0"][:, blk])
        zr = torch.stack([day["zA"][:, blk], day["zB"][:, blk]], 1)[:, net.ramp_route - 1]       # (D, R)
        dr = torch.exp(shared["log_r"][:, blk][None, :] + zr)
        dr = torch.cat([dr[:, :majB], bnd[:, blk:blk + 1], dr[:, majB + 1:]], 1)
        capf = None
        if "zc" in day:              # day x cell x window capacity reductions (non-recurrent bottlenecks)
            w = min(N_INC - 1, max(0, (t - 60) // 180))
            capf = torch.exp(day["zc"][:, :, w])
        st, info = ctm.step(st, shared, net, t, meters, rp, demand=(d0, dr), cap_factor=capf)
        q = st["qr"].sum(1)
        qpen = qpen + ((q - storage).clamp_min(0.0) / storage).pow(2).mean() / ctm.N_STEPS
        if t >= 60:
            av = av + ctm.fd_speed(st["n"].sum(1), info["cap"], vf, net, smooth=smooth)
            aq = aq + info["out_tot"] / ctm.DT_H
            if (t - 60) % ctm.STEPS_PER_5MIN == ctm.STEPS_PER_5MIN - 1:
                Vs.append(av / ctm.STEPS_PER_5MIN); Qs.append(aq / ctm.STEPS_PER_5MIN); av, aq = 0.0, 0.0
    return torch.stack(Vs, 1), torch.stack(Qs, 1), qpen


def new_day(D, M=None):
    d = {k: torch.zeros(D, ctm.N_BLOCKS, requires_grad=True) for k in ("z0", "zA", "zB")} | \
        {"bias": torch.zeros(D, requires_grad=True)}
    if M is not None:
        d["zc"] = torch.full((D, M, N_INC), -1e-3, requires_grad=True)
    return d


def det_scale(shared, n):
    """Detector observation operator: detector speed = cell speed x (1 + 0.15 tanh(det_logit)), absorbing local
    geometry (curves, grades, lane drops) within a cell. Identity when the parameter is absent."""
    if "det_logit" not in shared:
        return torch.ones(n)
    return 1.0 + 0.15 * torch.tanh(shared["det_logit"])


TAU_FLOOR = 2.0     # lowest smoothing temperature of the speed read-out during fitting (set from the command line)
HUBER = 0.0         # > 0: Huber loss on speed with this threshold in km/h (missed jams then cannot dominate the fit)


def fit(shared, day, net, V, Q, mV, mQ, bnd, cell_of, iters, lr, anneal, train_shared, w_q=0.3, tag=""):
    params = list(day.values()) + (list(shared.values()) if train_shared else [])
    opt = torch.optim.Adam(params, lr=lr)
    Yv = torch.tensor(np.nan_to_num(V), dtype=torch.float32); Yq = torch.tensor(np.nan_to_num(Q), dtype=torch.float32)
    mV = torch.tensor(mV); mQ = torch.tensor(mQ)
    log, t0 = [], time.time()
    for it in range(iters):
        opt.zero_grad()
        tau = max(TAU_FLOOR, anneal * (1 - it / (0.6 * iters)))
        Vm, Qm, qpen = simulate(shared, day, net, bnd, tau)
        Vm, Qm = Vm[:, :, cell_of] * det_scale(shared, len(cell_of)), Qm[:, :, cell_of]
        if HUBER > 0:
            fv = torch.nn.functional.huber_loss(Vm[mV] / V_SCALE, Yv[mV] / V_SCALE, delta=HUBER / V_SCALE)
        else:
            fv = (((Vm - Yv) / V_SCALE)[mV] ** 2).mean()
        fq = (((Qm - Yq) / Q_SCALE)[mQ] ** 2).mean()
        reg = sum(5.0 * (day[k].diff(dim=1) ** 2).mean() + 0.05 * (day[k] ** 2).mean() for k in ("z0", "zA", "zB")) \
            + 0.05 * (day["bias"] ** 2).mean()
        if "zc" in day:              # sparse: most cells/windows keep full capacity
            reg = reg + 2.0 * day["zc"].abs().mean()
        if train_shared:
            reg = reg + 0.5 * (shared["cap_logit"] ** 2).mean() + 0.2 * (shared["vf_logit"] ** 2).mean() \
                + 5.0 * (shared["log_r"].diff(dim=1) ** 2).mean() + 5.0 * (shared["log_d0"].diff() ** 2).mean()
            if "det_logit" in shared:
                reg = reg + 0.5 * (shared["det_logit"] ** 2).mean()
        (fv + w_q * fq + reg + qpen).backward()
        opt.step()
        with torch.no_grad():
            for k in ("z0", "zA", "zB"):
                day[k].clamp_(-1.0, 1.0)
            day["bias"].clamp_(-2.0, 2.0)
            if "zc" in day:
                day["zc"].clamp_(-1.2, 0.0)
            if train_shared:
                shared["vf"].clamp_(70.0, 125.0); shared["b1"].clamp_(0.0, 1.0)
                shared["log_r"].clamp_(math.log(5.0), math.log(4000.0))
        if it % 10 == 0 or it == iters - 1:
            log.append({"iter": it, "tau": round(tau, 1), "mae_v": (Vm - Yv)[mV].abs().mean().item(),
                        "mae_q": (Qm - Yq)[mQ].abs().mean().item(), "sec": round(time.time() - t0, 1)})
            print(tag, log[-1], flush=True)
    return log


def route_tt(Vd, sens):
    out = []
    for r, n in ((1, "A"), (2, "B")):
        m = (sens.route == r).values; s = sens[m]
        out.append(instantaneous_tt_min(pd.DataFrame(Vd[:, m], columns=s.sensor_id.values), s, route_length_km(n)))
    return np.stack(out, 1)


def main(iters=200, test_iters=120, tag="_daily2026", cell_km=1.0, drop=False, det=False, inc=False):
    net, sens, days, V, Q, M, B = load_days(cell_km=cell_km)
    ids = sens.sensor_id.values
    cell_of = torch.tensor([net.sensor_cell[int(s)] for s in ids])
    import split
    tr_i, tu_i, te_i = split.indices(days)
    tag = tag + split.tag()
    valid = M[tr_i][:, :, :].mean((0, 1)) >= 0.5
    hidden = np.zeros(len(ids), bool)
    hidden[np.random.default_rng(0).choice(len(ids), int(round(0.3 * len(ids))), replace=False)] = True
    vis = ~hidden & valid
    mV = np.isfinite(V) & vis[None, None, :]; mQ = np.isfinite(Q) & vis[None, None, :]
    print(f"usable weekdays {len(days)}: train {len(tr_i)}, test {len(te_i)}; valid detectors {valid.sum()}, "
          f"hidden&valid {(hidden & valid).sum()}", flush=True)
    p0 = torch.load(os.path.join(ROOT, "results", "wm_params_2026i.pt"))
    p0.pop("bnd_major_B", None)
    if cell_km != 1.0:
        from network import remap_params
        p0 = remap_params(p0, build_default_net()[0], net)
    if drop:
        p0["drop_logit"] = torch.zeros(2)                       # 10% initial drop on each route (0.2 * sigmoid(0))
    if det:
        p0["det_logit"] = torch.zeros(len(ids))
    print(f"cells {net.M} (sub {net.sub}), ramps {net.R}, capacity drop {drop}, detector operator {det}, incident layer {inc}", flush=True)
    shared = {k: v.clone().float().requires_grad_(True) for k, v in p0.items()}
    bnd = boundary_blocks(B)
    day_tr = new_day(len(tr_i), net.M if inc else None)
    log_tr = fit(shared, day_tr, net, V[tr_i], Q[tr_i], mV[tr_i], mQ[tr_i], bnd[tr_i], cell_of, iters, 0.05, 15.0, True, tag="train")
    shared = {k: v.detach() for k, v in shared.items()}
    torch.save(shared, os.path.join(ROOT, "results", f"wm_params{tag}.pt"))
    torch.save({"days": [str(days[i].date()) for i in tr_i], "bnd": bnd[tr_i],
                **{k: v.detach() for k, v in day_tr.items()}},
               os.path.join(ROOT, "results", f"day_params{tag}_train.pt"))   # real days for closed-loop truth
    if tu_i:                                   # tuning days: same treatment as test days (road frozen, visible detectors)
        day_tu = new_day(len(tu_i), net.M if inc else None)
        fit(shared, day_tu, net, V[tu_i], Q[tu_i], mV[tu_i], mQ[tu_i], bnd[tu_i], cell_of, test_iters, 0.08, 8.0, False, tag="tune")
        torch.save({"days": [str(days[i].date()) for i in tu_i], "bnd": bnd[tu_i], **{k: v.detach() for k, v in day_tu.items()}},
                   os.path.join(ROOT, "results", f"day_params{tag}_tune.pt"))
    day_te = new_day(len(te_i), net.M if inc else None)
    log_te = fit(shared, day_te, net, V[te_i], Q[te_i], mV[te_i], mQ[te_i], bnd[te_i], cell_of, test_iters, 0.08, 8.0, False, tag="test")
    with torch.no_grad():
        Vte, Qte, _ = simulate(shared, {k: v.detach() for k, v in day_te.items()}, net, bnd[te_i],
                               TAU_FLOOR if TAU_FLOOR > 2.0 else 0.0)      # same read-out as fitted
        Vte, Qte = (Vte[:, :, cell_of] * det_scale(shared, len(cell_of))).numpy(), Qte[:, :, cell_of].numpy()
    Yte = V[te_i]
    with np.errstate(all="ignore"):
        hist = np.nanmean(V[tr_i], 0)
    interp = np.stack([interp_nan(Yte[k], sens, vis) for k in range(len(te_i))])
    hv = hidden & valid
    per_day = {"wm": [], "interp": [], "hist": [], "tt_wm": [], "tt_interp": [], "tt_hist": []}
    for k in range(len(te_i)):
        y = Yte[k]
        for name, pr in (("wm", Vte[k]), ("interp", interp[k]), ("hist", hist)):
            per_day[name].append(float(np.nanmean(np.abs(pr - y)[:, hv])))
        truth_full = interp_nan(y, sens, valid)
        tt_true = route_tt(truth_full, sens)
        for name, pr in (("tt_wm", Vte[k]), ("tt_interp", interp_nan(interp[k], sens, valid)), ("tt_hist", interp_nan(hist, sens, valid))):
            per_day[name].append(float(np.nanmean(np.abs(route_tt(pr, sens) - tt_true))))     # slots with too few measured stations skipped
    res = {"n_days": len(days), "train_days": len(tr_i), "test_days": [str(days[i].date()) for i in te_i],
           "n_valid": int(valid.sum()), "n_hidden_valid": int(hv.sum()),
           "speed_mae_hidden_test": {k: float(np.mean(per_day[k])) for k in ("wm", "interp", "hist")},
           "route_tt_mae_test_min": {k: float(np.mean(per_day[k])) for k in ("tt_wm", "tt_interp", "tt_hist")},
           "days_wm_beats_interp_speed": int(np.sum(np.array(per_day["wm"]) < np.array(per_day["interp"]))),
           "days_wm_beats_hist_speed": int(np.sum(np.array(per_day["wm"]) < np.array(per_day["hist"]))),
           "days_wm_beats_interp_tt": int(np.sum(np.array(per_day["tt_wm"]) < np.array(per_day["tt_interp"]))),
           "days_wm_beats_hist_tt": int(np.sum(np.array(per_day["tt_wm"]) < np.array(per_day["tt_hist"]))),
           "visible_speed_mae_train_last": log_tr[-1]["mae_v"], "visible_speed_mae_test_last": log_te[-1]["mae_v"],
           "vf": shared["vf"].tolist(), "b1": float(shared["b1"]), "cell_km": cell_km, "cells": net.M,
           "capacity_drop": [float(x) for x in 0.2 * torch.sigmoid(shared["drop_logit"])] if drop else None,
           "detector_scale_range": [float(det_scale(shared, len(ids)).min()), float(det_scale(shared, len(ids)).max())] if det else None,
           "max_vf_cell": float(ctm.vf_cells(shared, net).max()), "tau_floor": TAU_FLOOR, "huber_kmh": HUBER,
           "incident_layer": None if not inc else {
               "share_cell_windows_reduced_train": float((day_tr["zc"].detach() < -0.05).float().mean()),
               "share_cell_windows_reduced_test": float((day_te["zc"].detach() < -0.05).float().mean())}}
    print(json.dumps(res, indent=2))
    torch.save({"days": [str(days[i].date()) for i in te_i], "bnd": bnd[te_i],
                **{k: v.detach() for k, v in day_te.items()}},
               os.path.join(ROOT, "results", f"day_params{tag}_test.pt"))
    json.dump({"result": res, "per_day": per_day, "log_train": log_tr, "log_test": log_te},
              open(os.path.join(ROOT, "results", f"calibration{tag}.json"), "w"), indent=2)
    np.savez(os.path.join(ROOT, "results", f"calibration{tag}_arrays.npz"), Vte=Vte, Yte=Yte, interp=interp, hist=hist,
             hidden=hidden, valid=valid)


if __name__ == "__main__":
    a = sys.argv
    if len(a) > 8:
        TAU_FLOOR = float(a[8])
    if len(a) > 9:
        HUBER = float(a[9])
    main(int(a[1]) if len(a) > 1 else 200, int(a[2]) if len(a) > 2 else 120, a[3] if len(a) > 3 else "_daily2026",
         float(a[4]) if len(a) > 4 else 1.0, (len(a) > 5 and a[5] == "drop"), (len(a) > 6 and a[6] == "det"),
         (len(a) > 7 and a[7] == "inc"))
