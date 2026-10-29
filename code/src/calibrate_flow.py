"""Calibration v4: detector speeds AND mainline flows (LargeST), gradient descent through the CTM.

Changes from calibrate.py (speeds only):
- loss adds 5-min flows at visible detectors (veh/h), scaled so 10 km/h and 500 veh/h weigh the same;
- each ramp gets its own 15-min demand profile (was: one shape per route), now identifiable from flows;
- the US-101 N inflow at the SR-87 junction (route B major ramp) is a measured boundary condition;
- per-cell capacity and free-flow speed are initialised from each detector's empirical fundamental diagram.
Train: mean weekday Jan-Apr 2017; test: mean weekday May-Jun 2017. Same 30% hidden detectors as calibrate.py
(hidden for both speed and flow). Baselines at hidden detectors: linear interpolation along the route.
"""
import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, torch
torch.set_num_threads(2)
import ctm
from data import load_speeds, weekdays, day_window, instantaneous_tt_min, route_length_km
from network import build_default_net
from flows import route_flows, boundary_sensors
from calibrate import interp_baseline

ROOT = os.path.join(os.path.dirname(__file__), "..")
torch.manual_seed(0)
V_SCALE, Q_SCALE = 10.0, 500.0
SPILL_Q = 60.0       # regular on-ramp storage (veh); unmetered queues beyond it are implausible
SPILL_MAJOR = 600.0  # freeway-to-freeway connectors: a point queue standing for upstream freeway congestion


def mean_profiles(df, ids, days):
    arr = [day_window(df, d)[ids].values for d in days]
    arr = [a for a in arr if a.shape[0] == 72 and np.isfinite(a).all()]
    return np.mean(arr, 0), len(arr)


def targets():
    net, sA, sB = build_default_net()
    sp = load_speeds()
    fl = route_flows()
    fl.columns = fl.columns.astype(int)
    days = weekdays(sp.index)
    sens = pd.concat([sA.assign(route=1), sB.assign(route=2)], ignore_index=True)
    ids = sens.sensor_id.values
    has_q = np.isin(ids, fl.columns)
    flq = fl.reindex(columns=ids)                                  # NaN column for the missing sensor
    tr = [d for d in days if d.month <= 4]
    te = [d for d in days if d.month >= 5]
    out = {"net": net, "sens": sens, "has_q": has_q}
    out["Vtr"], out["ntr"] = mean_profiles(sp, ids, tr)
    out["Vte"], out["nte"] = mean_profiles(sp, ids, te)
    out["Qtr"] = np.nanmean([day_window(flq, d).values for d in tr if len(day_window(flq, d)) == 72], 0)
    out["Qte"] = np.nanmean([day_window(flq, d).values for d in te if len(day_window(flq, d)) == 72], 0)
    bnd = boundary_sensors()
    out["bnd_id"] = bnd[0]
    out["Btr"] = np.nanmean([day_window(fl, d)[bnd[0]].values for d in tr if len(day_window(fl, d)) == 72], 0)
    rng = np.random.default_rng(0)                                 # identical draw to calibrate.py
    hidden = np.zeros(len(ids), bool)
    hidden[rng.choice(len(ids), int(round(0.3 * len(ids))), replace=False)] = True
    out["hidden"] = hidden
    # empirical fundamental diagram per detector (train weekdays, 05:00-11:00)
    V = np.concatenate([day_window(sp, d)[ids].values for d in tr])
    Q = np.concatenate([day_window(flq, d).values for d in tr])
    lanes = sens.Lanes.values.astype(float)
    qpl = Q / lanes
    cap = np.nanquantile(qpl, 0.98, axis=0)
    vf = np.array([np.nanmedian(V[(qpl[:, j] < 900) & np.isfinite(qpl[:, j]), j]) if has_q[j] else np.nan
                   for j in range(len(ids))])
    out["emp_cap_lane"], out["emp_vf"] = cap, vf
    return out


def masked_mean_profile(df, mask, ids, days):
    """Mean 05:00-11:00 profile over days using only measured rows; NaN where never measured. (72, n)."""
    arr = []
    for d in days:
        w = day_window(df, d).reindex(columns=ids)
        m = day_window(mask, d).reindex(columns=ids).fillna(False).astype(bool)
        if len(w) != 72:
            continue
        arr.append(np.where(m.values, w.values, np.nan))
    with np.errstate(all="ignore"):
        return np.nanmean(np.array(arr, dtype=float), 0), len(arr)


def cell_of_x(net, route, x):
    idx = net.a_idx if route == "A" else net.b_idx
    Lc = float(net.L[idx[0]])
    return int(idx[min(len(idx) - 1, int(x // Lc))])


def ramp_targets(net, tr, te, year=2026, month=4):
    """Measured on-ramp inflow per model ramp and off-ramp flow per exit cell, mean 05:00-11:00 profiles (veh/h).
    A model ramp (cell) is a target only if every OR station in that cell is measured in >= 50% of training AM rows;
    same rule for exit cells and FR/FF stations. 30% of the targets are hidden for evaluation."""
    from pems_raw import load_month, observed_mask
    from network import RAMPS_2026
    rm = pd.read_csv(RAMPS_2026)
    raw = load_month(year, month, rm.ID.tolist(), "corridor_ramps")
    fl, obs = raw["flow"], observed_mask(raw)
    rm["cell"] = [cell_of_x(net, r, x) for r, x in zip(rm.route, rm.x_km)]
    am_tr = np.concatenate([day_window(obs.reindex(columns=rm.ID), d).fillna(False).values for d in tr])
    rm["ok"] = am_tr.astype(float).mean(0) >= 0.5
    prof_tr, _ = masked_mean_profile(fl, obs, rm.ID.values, tr)
    prof_te, _ = masked_mean_profile(fl, obs, rm.ID.values, te)
    ramp_cells = net.ramp_cell.tolist()
    R = net.R
    Rtr = np.full((72, R), np.nan); Rte = np.full((72, R), np.nan)
    for i, c in enumerate(ramp_cells):
        g = np.where((rm.cell.values == c) & (rm.Type.values == "OR"))[0]
        if len(g) and rm.ok.values[g].all() and not (bool(net.ramp_major[i]) and int(net.ramp_route[i]) == 2):
            Rtr[:, i] = prof_tr[:, g].sum(1); Rte[:, i] = prof_te[:, g].sum(1)
    Xtr = np.full((72, net.M), np.nan); Xte = np.full((72, net.M), np.nan)
    for c in sorted(set(rm.cell[rm.Type.isin(["FR", "FF"])])):
        g = np.where((rm.cell.values == c) & np.isin(rm.Type.values, ["FR", "FF"]))[0]
        if rm.ok.values[g].all():
            Xtr[:, c] = prof_tr[:, g].sum(1); Xte[:, c] = prof_te[:, g].sum(1)
    has_r = np.isfinite(Rtr).any(0); has_x = np.isfinite(Xtr).any(0)
    rng = np.random.default_rng(1)
    hid_r = np.zeros(R, bool); cand = np.where(has_r)[0]
    hid_r[rng.choice(cand, int(round(0.3 * len(cand))), replace=False)] = True
    hid_x = np.zeros(net.M, bool); cand = np.where(has_x)[0]
    hid_x[rng.choice(cand, int(round(0.3 * len(cand))), replace=False)] = True
    return {"Rtr": Rtr, "Rte": Rte, "Xtr": Xtr, "Xte": Xte, "has_r": has_r, "has_x": has_x,
            "hid_r": hid_r, "hid_x": hid_x}


def targets_2026(year=2026, month=4, split_day=21, ramps=False, split="interleave"):
    """Raw PeMS month: train = weekdays up to split_day, test = later weekdays. Only measured rows are used;
    a detector is valid if it is measured in >= 50% of training 05:00-11:00 rows."""
    from pems_raw import load_month, observed_mask
    net, sA, sB = build_default_net(ramps="2026" if ramps else "2017")
    sens = pd.concat([sA.assign(route=1), sB.assign(route=2)], ignore_index=True)
    ids = sens.sensor_id.values.astype(int)
    bnd_all = boundary_sensors()
    raw = load_month(year, month, list(ids) + bnd_all, "corridor")
    sp, fl, obs = raw["speed"], raw["flow"], observed_mask(raw)
    days = weekdays(sp.index)
    # drop days with (almost) no measured mainline rows 05:00-11:00 (detector/communication outages)
    ml_ok = obs.reindex(columns=ids).fillna(False)
    share = {d: float(day_window(ml_ok, d).values.mean()) for d in days}
    dropped = [str(d.date()) for d in days if share[d] < 0.3]
    days = [d for d in days if share[d] >= 0.3]
    if split == "interleave":          # every third usable weekday is a test day
        te = days[2::3]; tr = [d for d in days if d not in te]
    else:
        tr = [d for d in days if d.day <= split_day]; te = [d for d in days if d.day > split_day]
    out = {"net": net, "sens": sens, "days_dropped": dropped}
    out["Vtr"], out["ntr"] = masked_mean_profile(sp, obs, ids, tr)
    out["Vte"], out["nte"] = masked_mean_profile(sp, obs, ids, te)
    out["Qtr"], _ = masked_mean_profile(fl, obs, ids, tr)
    out["Qte"], _ = masked_mean_profile(fl, obs, ids, te)
    am_tr = np.concatenate([day_window(obs.reindex(columns=ids), d).fillna(False).values for d in tr])
    valid = am_tr.astype(float).mean(0) >= 0.5
    out["valid"], out["has_q"] = valid, valid.copy()
    bshare = {b: float(np.concatenate([day_window(obs.reindex(columns=[b]), d).fillna(False).values for d in tr]).mean()) for b in bnd_all}
    bnd = next(b for b in bnd_all if bshare[b] >= 0.5)
    out["bnd_id"], out["bnd_measured_share"] = bnd, bshare
    out["Btr"], _ = masked_mean_profile(fl, obs, [bnd], tr)
    out["Btr"] = pd.Series(out["Btr"][:, 0]).interpolate(limit_direction="both").values
    rng = np.random.default_rng(0)                                 # identical draw to calibrate.py
    hidden = np.zeros(len(ids), bool)
    hidden[rng.choice(len(ids), int(round(0.3 * len(ids))), replace=False)] = True
    out["hidden"] = hidden
    V = np.concatenate([np.where(day_window(obs.reindex(columns=ids), d).fillna(False).values,
                                 day_window(sp.reindex(columns=ids), d).values, np.nan) for d in tr])
    Q = np.concatenate([np.where(day_window(obs.reindex(columns=ids), d).fillna(False).values,
                                 day_window(fl.reindex(columns=ids), d).values, np.nan) for d in tr])
    qpl = Q / sens.Lanes.values.astype(float)
    with np.errstate(all="ignore"):
        out["emp_cap_lane"] = np.nanquantile(qpl, 0.98, axis=0)
        out["emp_vf"] = np.array([np.nanmedian(V[(qpl[:, j] < 900) & np.isfinite(qpl[:, j]), j]) if valid[j] else np.nan
                                  for j in range(len(ids))])
    out["days_train"] = [str(d.date()) for d in tr]; out["days_test"] = [str(d.date()) for d in te]
    if ramps:
        out.update(ramp_targets(net, tr, te, year, month))
    return out


def interp_nan(Y, sens, usable):
    """Linear interpolation along each route from usable, finite entries; (72, n)."""
    out = np.full_like(Y, np.nan, dtype=float)
    for r in (1, 2):
        m = (sens.route == r).values
        x = sens.x_km.values[m]; idx = np.where(m)[0]
        for t in range(Y.shape[0]):
            y = Y[t, idx]; ok = usable[m] & np.isfinite(y)
            if ok.sum() >= 2:
                out[t, idx] = np.interp(x, x[ok], y[ok])
    return out


def init_params(net, T):
    p = ctm.default_params(net)
    cell_of = np.array([net.sensor_cell[int(s)] for s in T["sens"].sensor_id])
    vis = ~T["hidden"] & T["has_q"]
    cap_m = np.ones(net.M); vf_c = np.full(net.M, np.nan)
    for c in range(net.M):
        js = np.where((cell_of == c) & vis)[0]
        if len(js):
            cap_m[c] = np.clip(np.nanmin(T["emp_cap_lane"][js]) / ctm.Q_LANE, 0.62, 1.38)
            vf_c[c] = np.nanmedian(T["emp_vf"][js])
    p["cap_logit"] = torch.tensor(np.arctanh((cap_m - 1.0) / 0.4), dtype=torch.float32)
    for r in (1, 2):
        m = (net.route == r).numpy()
        base = float(np.nanmedian(vf_c[m]))
        p["vf"][r - 1] = base
        mult = np.where(np.isfinite(vf_c[m]), vf_c[m] / base, 1.0)
        p["vf_logit"][torch.tensor(m)] = torch.tensor(np.arctanh(np.clip((mult - 1) / 0.2, -0.95, 0.95)), dtype=torch.float32)
    # per-ramp demand profile, starting from the old route-level shape
    p["log_r"] = (p["log_a"][:, None] + p["log_g"][net.ramp_route - 1]).clone()
    del p["log_a"], p["log_g"]
    return p


def boundary_profile(T, net):
    """Measured US-101 N inflow (veh/h) per 15-min block for the route-B major ramp; before 05:00 use 05:00."""
    q5 = T["Btr"]                                                  # 72 slots from 05:00
    blocks = q5.reshape(-1, 3).mean(1)                             # 24 blocks 05:00-11:00
    pre = (ctm.N_BLOCKS - len(blocks))
    return torch.tensor(np.concatenate([np.full(pre, blocks[0]), blocks]), dtype=torch.float32)


def simulate(p, net, bnd, smooth=0.0):
    """Returns cell speeds and cell outflows (veh/h) per 5-min slot 05:00-11:00, each (72, M)."""
    st = ctm.init_state(net, 1)
    meters = net.ramp_cap.expand(1, -1)
    route_prob = lambda a, b: torch.sigmoid(p["b0"] + p["b1"] * (a - b)).reshape(1, 1).expand(1, 2)
    majB = int(torch.where(net.ramp_major & (net.ramp_route == 2))[0])
    vf_cell = ctm.vf_cells(p, net)
    Vs, Qs, Rs, Xs, av, aq, ar, ax, qpen, qmax = [], [], [], [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    storage = torch.where(net.ramp_major, torch.tensor(SPILL_MAJOR), torch.tensor(SPILL_Q))
    for t in range(ctm.N_STEPS):
        blk = min(ctm.N_BLOCKS - 1, t // (2 * ctm.BLOCK_MIN))
        d0 = torch.exp(p["log_d0"][blk]).reshape(1)
        dr = torch.exp(p["log_r"][:, blk])
        dr = torch.cat([dr[:majB], bnd[blk].reshape(1), dr[majB + 1:]])[None, :]
        st, info = ctm.step(st, p, net, t, meters, route_prob, demand=(d0, dr))
        q = st["qr"][0].sum(0)
        qpen = qpen + ((q - storage).clamp_min(0.0) / storage).pow(2).mean() / ctm.N_STEPS
        qmax = max(qmax, float((q / storage).max()))
        if t >= 60:
            av = av + ctm.fd_speed(st["n"].sum(1), info["cap"], vf_cell, net, smooth=smooth)[0]
            aq = aq + info["out_tot"][0] / ctm.DT_H
            ar = ar + info["ramp_in"][0] / ctm.DT_H
            ax = ax + info["exit_cell"][0] / ctm.DT_H
            if (t - 60) % ctm.STEPS_PER_5MIN == ctm.STEPS_PER_5MIN - 1:
                k = ctm.STEPS_PER_5MIN
                Vs.append(av / k); Qs.append(aq / k); Rs.append(ar / k); Xs.append(ax / k); av, aq, ar, ax = 0.0, 0.0, 0.0, 0.0
    return torch.stack(Vs), torch.stack(Qs), qpen, qmax, torch.stack(Rs), torch.stack(Xs)


def route_tt_err(V, Y, sens):
    out = {}
    for r, name in ((1, "A"), (2, "B")):
        m = (sens.route == r).values
        s = sens[m].reset_index(drop=True)
        cols = list(range(m.sum()))
        L = route_length_km(name)
        f = lambda A: instantaneous_tt_min(pd.DataFrame(A[:, m], columns=s.sensor_id.values), s, L)
        out[name] = float(np.abs(f(V) - f(Y)).mean())
    return out


def main(iters=300, lr=0.05, w_q=1.0, tag="_v4", anneal=25.0, w_spill=1.0, source="2017", split="interleave"):
    ramps = source == "2026ramps"
    T = targets() if source == "2017" else targets_2026(ramps=ramps, split=split)
    net, sens, hidden, has_q = T["net"], T["sens"], T["hidden"], T["has_q"]
    valid = T.get("valid", np.ones(len(hidden), bool))
    cell_of = torch.tensor([net.sensor_cell[int(s)] for s in sens.sensor_id])
    # elementwise masks: visible, valid detector, and a measured value in this slot
    mV = torch.tensor(np.isfinite(T["Vtr"]) & (~hidden & valid)[None, :])
    mQ = torch.tensor(np.isfinite(T["Qtr"]) & (~hidden & has_q)[None, :])
    Yv = torch.tensor(np.nan_to_num(T["Vtr"]), dtype=torch.float32)
    Yq = torch.tensor(np.nan_to_num(T["Qtr"]), dtype=torch.float32)
    bnd = boundary_profile(T, net)
    p0 = init_params(net, T)
    if ramps:   # start measured ramps at their measured profile; masks for ramp and exit losses
        mR = torch.tensor(np.isfinite(T["Rtr"]) & (T["has_r"] & ~T["hid_r"])[None, :])
        mX = torch.tensor(np.isfinite(T["Xtr"]) & (T["has_x"] & ~T["hid_x"])[None, :])
        Yr = torch.tensor(np.nan_to_num(T["Rtr"]), dtype=torch.float32)
        Yx = torch.tensor(np.nan_to_num(T["Xtr"]), dtype=torch.float32)
        blocks = np.nanmean(T["Rtr"].reshape(24, 3, -1), 1)               # 24 x R
        pre = ctm.N_BLOCKS - 24
        init = np.concatenate([np.repeat(blocks[:1], pre, 0), blocks], 0).T  # R x N_BLOCKS
        use = T["has_r"] & ~T["hid_r"]
        lr0 = p0["log_r"].numpy().copy()
        lr0[use] = np.log(np.clip(np.nan_to_num(init[use], nan=np.exp(lr0[use])), 20.0, 4000.0))
        p0["log_r"] = torch.tensor(lr0, dtype=torch.float32)
    p = {k: v.clone().requires_grad_(True) for k, v in p0.items()}
    opt = torch.optim.Adam(p.values(), lr=lr)
    log, t0 = [], time.time()
    for it in range(iters):
        opt.zero_grad()
        tau = max(2.0, anneal * (1 - it / (0.6 * iters))) if anneal > 0 else 2.0
        V, Q, qpen, qmax, Rr, Xx = simulate(p, net, bnd, smooth=tau)
        V, Q = V[:, cell_of], Q[:, cell_of]
        fv = (((V - Yv) / V_SCALE)[mV] ** 2).mean()
        fq = (((Q - Yq) / Q_SCALE)[mQ] ** 2).mean()
        reg = 0.5 * (p["cap_logit"] ** 2).mean() + 0.2 * (p["vf_logit"] ** 2).mean() \
            + 5.0 * (p["log_r"].diff(dim=1) ** 2).mean() + 5.0 * (p["log_d0"].diff() ** 2).mean()
        loss = fv + w_q * fq + reg + w_spill * qpen
        if ramps:
            loss = loss + w_q * (((Rr - Yr) / Q_SCALE)[mR] ** 2).mean() + w_q * (((Xx - Yx) / Q_SCALE)[mX] ** 2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            p["vf"].clamp_(70.0, 125.0); p["b1"].clamp_(0.0, 1.0); p["log_r"].clamp_(math.log(5.0), math.log(4000.0))
        if it % 10 == 0 or it == iters - 1:
            log.append({"iter": it, "tau": round(tau, 1),
                        "mae_v_vis": (V - Yv)[mV].abs().mean().item(),
                        "mae_q_vis": (Q - Yq)[mQ].abs().mean().item(), "max_q_over_storage": round(qmax, 2),
                        "sec": round(time.time() - t0, 1)})
            print(log[-1], flush=True)

    with torch.no_grad():
        V, Q, _, qmax, Rr, Xx = simulate(p, net, bnd, smooth=0.0)
        V, Q, Rr, Xx = V[:, cell_of].numpy(), Q[:, cell_of].numpy(), Rr.numpy(), Xx.numpy()
    hv, hq = hidden & valid, hidden & has_q
    vis_ok = ~hidden & valid
    bv_te = interp_nan(T["Vte"], sens, vis_ok)
    bq_te = interp_nan(T["Qte"], sens, ~hidden & has_q)
    mae = lambda P, Y, cols: float(np.nanmean(np.abs(P - Y)[:, cols]))
    Yte_full = interp_nan(T["Vte"], sens, valid)          # truth for route travel time: all measured detectors
    Ytr_full = interp_nan(T["Vtr"], sens, valid)
    res = {
        "source": source, "split": split if source != "2017" else "months", "days_dropped": T.get("days_dropped"),
        "days_test": T.get("days_test"), "w_q": w_q,
        "n_train_days": T["ntr"], "n_test_days": T["nte"], "n_hidden": int(hv.sum()),
        "n_valid_detectors": int(valid.sum()), "boundary": T.get("bnd_id"),
        "speed_mae": {"wm_vis_train": mae(V, T["Vtr"], vis_ok), "wm_vis_test": mae(V, T["Vte"], vis_ok),
                      "wm_hidden_test": mae(V, T["Vte"], hv), "interp_hidden_test": mae(bv_te, T["Vte"], hv),
                      "train_profile_hidden_test": mae(T["Vtr"], T["Vte"], hv)},
        "flow_mae_vehh": {"wm_vis_train": mae(Q, T["Qtr"], ~hidden & has_q), "wm_vis_test": mae(Q, T["Qte"], ~hidden & has_q),
                          "wm_hidden_test": mae(Q, T["Qte"], hq), "interp_hidden_test": mae(bq_te, T["Qte"], hq)},
        "route_tt_mae_min_test": {"wm": route_tt_err(V, Yte_full, sens),
                                  "train_profile": route_tt_err(Ytr_full, Yte_full, sens)},
        "ramps": None if not ramps else {
            "n_onramp_targets": int(T["has_r"].sum()), "n_hidden": int(T["hid_r"].sum()),
            "n_exit_targets": int(T["has_x"].sum()), "n_exit_hidden": int(T["hid_x"].sum()),
            "onramp_mae_vis_train": mae(Rr, T["Rtr"], T["has_r"] & ~T["hid_r"]),
            "onramp_mae_vis_test": mae(Rr, T["Rte"], T["has_r"] & ~T["hid_r"]),
            "onramp_mae_hidden_test": mae(Rr, T["Rte"], T["hid_r"]),
            "onramp_train_profile_hidden_test": mae(T["Rtr"], T["Rte"], T["hid_r"]),
            "exit_mae_vis_train": mae(Xx, T["Xtr"], T["has_x"] & ~T["hid_x"]),
            "exit_mae_hidden_test": mae(Xx, T["Xte"], T["hid_x"]),
            "exit_train_profile_hidden_test": mae(T["Xtr"], T["Xte"], T["hid_x"])},
        "vf": p["vf"].tolist(), "b0": float(p["b0"]), "b1": float(p["b1"]),
        "max_ramp_queue_over_storage_no_control": qmax,
        "cap_mult_range": [float(ctm.cap_mult(p).min()), float(ctm.cap_mult(p).max())],
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(res, indent=2))
    sv = {k: v.detach() for k, v in p.items()}
    sv["bnd_major_B"] = bnd
    torch.save(sv, os.path.join(ROOT, "results", f"wm_params{tag}.pt"))
    json.dump({"result": res, "log": log}, open(os.path.join(ROOT, "results", f"calibration{tag}.json"), "w"), indent=2)
    np.savez(os.path.join(ROOT, "results", f"calibration_arrays{tag}.npz"), V=V, Q=Q, Vtr=T["Vtr"], Vte=T["Vte"],
             Qtr=T["Qtr"], Qte=T["Qte"], hidden=hidden, has_q=has_q, valid=valid, x=sens.x_km.values, route=sens.route.values)


if __name__ == "__main__":
    a = sys.argv
    main(iters=int(a[1]) if len(a) > 1 else 300, lr=float(a[2]) if len(a) > 2 else 0.05,
         w_q=float(a[3]) if len(a) > 3 else 1.0, tag=a[4] if len(a) > 4 else "_v4",
         anneal=float(a[5]) if len(a) > 5 else 25.0, w_spill=float(a[6]) if len(a) > 6 else 1.0,
         source=a[7] if len(a) > 7 else "2017", split=a[8] if len(a) > 8 else "interleave")
