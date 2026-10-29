"""Fleet agent (ride-hailing / robotaxi operator) negotiating with the traffic management agent.

Design: docs/fleet_agent_design.md. Through demand at the origin is split into humans (O_H) and a fleet with share s_F,
of which a share e is empty (F_emp) and the rest occupied (F_occ). Private agentic vehicles are off here (p_agentic = 0).

Fleet behaviour levels:
  F0  self-interested fleet: routes occupied and empty vehicles on live travel times, ignores the operator.
  F1  obedient: the operator sets the fleet's route share to B (theta) and, with hold on, the share of new empties
      held at the depot (h).
  F2  negotiating: the operator offers sigma $/vehicle to switch route (sign gives the direction) and, with hold on,
      y $/held empty. The fleet accepts per vehicle when the payment covers its cost: occupied vehicles use the
      passengers' value of time, empties the fleet's operating cost c_E ($/h); holding costs c_E x mean hold time plus
      a later-next-pickup penalty.
  F3  as F2 but offers apply to empties only; occupied vehicles behave as in F0.
The operator plans with its world model and its belief about passenger value of time (VOT_BELIEF); the true world
uses VOT_TRUE. Fleet costs are known to both (c_E is published in operator reports).
"""
from __future__ import annotations
import math, time
import numpy as np
import torch
import ctm
from agentic import World, alinea, _lognorm_cdf, VOT_TRUE, VOT_BELIEF, PI_MAX, SPILL_Q, C1

C_EMP = 20.0            # fleet operating cost of an empty vehicle, $/h
C_EMP_SD = 0.3          # log-sd across vehicles (next-dispatch value differs by vehicle)
HOLD_PENALTY = 2.0      # $ per held empty for picking up its next rider later
Y_MAX = 12.0            # $ per held empty
H_MAX = 0.8             # max share of new empties held (F1)
EMPTY_SHARE = 0.41      # Waymo 2026Q2 statewide: P1 cruising 26% + P2 en route 15% of VMT


def hold_cost_mean():
    return C_EMP * ctm.T_HOLD_MIN / 60.0 + HOLD_PENALTY


def n_levers(level, hold):
    return {"F0": 0, "F1": 1, "F2": 1, "F3": 1}[level] + (1 if hold and level != "F0" else 0)


def lever_bounds(level, hold):
    lo, hi = [], []
    if level == "F1":
        lo, hi = [0.0], [1.0]
    elif level in ("F2", "F3"):
        lo, hi = [-PI_MAX], [PI_MAX]
    if hold and level != "F0":
        if level == "F1":
            lo.append(0.0); hi.append(H_MAX)
        else:
            lo.append(0.0); hi.append(Y_MAX)
    return lo, hi


def fleet_probs(level, lev, ttA, ttB, b0, b1, vot):
    """Route-B probabilities (B, 4) for (O_H, O_A, F_occ, F_emp) and induced (paid) shares (B, 2) for (F_occ, F_emp)."""
    pH = torch.sigmoid(b0 + b1 * (ttA - ttB))
    sF = torch.sigmoid(C1 * (ttA - ttB))
    Bsz = pH.shape[0]
    ind = torch.zeros(Bsz, 2)
    if level == "F0" or lev is None:
        pocc, pemp = sF, sF
    elif level == "F1":
        th = lev[:, 0].clamp(0.0, 1.0)
        pocc, pemp = th, th
    else:
        sig = lev[:, 0]
        pos, neg = sig.clamp_min(0.0), (-sig).clamp_min(0.0)
        extra_B = (ttB - ttA).clamp_min(0.1); extra_A = (ttA - ttB).clamp_min(0.1)
        def accept(mean, sd):
            FB = _lognorm_cdf(60.0 * pos / extra_B, mean, sd) * (pos > 0)
            FA = _lognorm_cdf(60.0 * neg / extra_A, mean, sd) * (neg > 0)
            to_B, to_A = (1 - sF) * FB, sF * FA
            return sF + to_B - to_A, to_B + to_A
        pemp, ind_e = accept(C_EMP, C_EMP_SD)
        if level == "F2":
            pocc, ind_o = accept(*vot)
        else:
            pocc, ind_o = sF, torch.zeros_like(sF)
        ind = torch.stack([ind_o, ind_e], 1)
    return torch.stack([pH, pH, pocc, pemp], 1), ind


def hold_share(level, lev, hold):
    """Share of new empties held at the depot, (B,) or None; and whether it is paid (offer y)."""
    if not hold or level == "F0" or lev is None:
        return None
    x = lev[:, -1]
    if level == "F1":
        return x.clamp(0.0, H_MAX)
    return _lognorm_cdf(x.clamp_min(1e-6), hold_cost_mean(), C_EMP_SD) * (x > 0)


class FleetWorld(World):
    """Truth (or model) world with a fleet; human route choice keeps the day-specific bias of World."""

    def __init__(self, net, p, level, s_f, hold, truth=False, day_seed=0, incident=False, truth_model="ctm",
                 empty_share=EMPTY_SHARE):
        super().__init__(net, p, 0, 0.0, 0.0, truth=truth, day_seed=day_seed, incident=incident, truth_model=truth_model)
        self.flevel, self.s_f, self.hold, self.e = level, s_f, hold, empty_share
        if truth_model == "metanet":
            import metanet
            self.state = metanet.init_state(net, 1, self.p)
        self.state["qh"] = torch.zeros(1)
        self.fm = {"tts_road": 0.0, "hold_vh": 0.0, "occ_vh": 0.0, "emp_vh": 0.0, "pay_route": 0.0, "pay_hold": 0.0,
                   "emp_vkt": 0.0, "occ_served": 0.0, "human_vh": 0.0, "human_served": 0.0}

    def step(self, meters, lever):
        net = self.net
        lev = None if lever is None else torch.as_tensor(lever, dtype=torch.float32).reshape(1, -1)
        holder = {}
        def rp(a, b):
            probs, ind = fleet_probs(self.flevel, lev, a, b, self.b0, self.p["b1"], self.vot)
            holder["ind"] = ind
            return probs
        h = hold_share(self.flevel, lev, self.hold)
        prev = self.state
        st, info = self.sim(self.state, self.p, net, self.t, meters.reshape(1, -1), rp, cap_factor=self.cap_factor(),
                            demand=self.demand(self.t), p_agentic=0.0, fleet=(self.s_f, self.e), hold=h)
        info["ind"] = holder.get("ind", torch.zeros(1, 2))       # realised accepted shares (F_occ, F_emp)
        if self.t >= 60:
            road = st["n"].sum().item() + st["qo"].sum().item() + st["qr"].sum().item()
            fm = self.fm
            fm["tts_road"] += road * ctm.DT_H
            fm["hold_vh"] += float(st["qh"].sum()) * ctm.DT_H
            fm["occ_vh"] += (st["n"][0, 4].sum() + st["qo"][0, 2]).item() * ctm.DT_H
            fm["emp_vh"] += (st["n"][0, 5].sum() + st["qo"][0, 3] + st["qh"][0]).item() * ctm.DT_H
            fm["human_vh"] += (st["n"][0, 0].sum() + st["qo"][0, 0]).item() * ctm.DT_H
            fm["human_served"] += info["exit"][0, 0].item()
            fm["occ_served"] += info["exit"][0, 4].item()
            n0 = prev["n"][0]
            fm["emp_vkt"] += (info["out_tot"][0] * n0[5] / n0.sum(0).clamp_min(1e-6) * net.L).sum().item()
            if self.flevel in ("F2", "F3") and lev is not None:
                fm["pay_route"] += float((info["o_F"][0] * holder["ind"][0]).sum()) * abs(float(lev[0, 0]))
            if self.hold and self.flevel in ("F2", "F3") and lev is not None:
                fm["pay_hold"] += float(info["held"].sum()) * float(lev[0, -1])
            q = st["qr"][0].sum(0).numpy() * (~net.ramp_major).numpy()
            self.m["ramp_wait"] += q * ctm.DT_H
            arr = self.demand(self.t)[1][0].numpy() * ctm.DT_H
            self.m["ramp_served"] += prev["qr"][0].sum(0).numpy() + arr - st["qr"][0].sum(0).numpy()
            self.m["max_q"] = max(self.m["max_q"], float(q.max()))
        self.state, self.t = st, self.t + 1
        return info

    def road_stock(self):
        st = self.state
        return st["n"].sum().item() + st["qo"].sum().item() + st["qr"].sum().item()


@torch.no_grad()
def fleet_rollout_cost(world, p, X, nb, nm, nlev, block_steps, ctrl_idx, level, hold, s_f, e):
    """Operator cost of candidate plans X (P, nb*(nm+nlev)): all vehicle-hours (road, queues and depot) +
    ramp spillback penalty + incentive payments converted at $25/veh-h."""
    net = world.net
    P = X.shape[0]
    st = {k: v.expand(P, *v.shape[1:]).clone() for k, v in world.state.items() if k in ("n", "qo", "qr", "qh")}
    cost = torch.zeros(P)
    per = nm + nlev
    for b in range(nb):
        u = X[:, b * per:(b + 1) * per]
        meters = net.ramp_cap.expand(P, -1).clone()
        meters[:, ctrl_idx] = u[:, :nm]
        lev = u[:, nm:] if nlev else None
        h = hold_share(level, lev, hold)
        for s in range(block_steps):
            t = world.t + b * block_steps + s
            holder = {}
            def rp(a, bb):
                probs, ind = fleet_probs(level, lev, a, bb, p["b0"], p["b1"], VOT_BELIEF)
                holder["ind"] = ind
                return probs
            d0, dr = ctm.demand_at(p, net, min(t, ctm.N_STEPS - 1))
            st, info = ctm.step(st, p, net, t, meters, rp, demand=(d0.expand(P), dr.expand(P, -1)),
                                fleet=(s_f, e), hold=h)
            stock = st["n"].sum((1, 2)) + st["qo"].sum(1) + st["qr"].sum((1, 2)) + st["qh"]
            q = st["qr"].sum(1)
            cost += stock * ctm.DT_H + 10.0 * ((q - SPILL_Q).clamp_min(0) * (~net.ramp_major)).sum(1) * ctm.DT_H
            if level in ("F2", "F3"):
                cost += (info["o_F"] * holder["ind"]).sum(1) * lev[:, 0].abs() / 25.0
                if hold:
                    cost += info["held"] * lev[:, -1] / 25.0
    cost += (st["n"].sum((1, 2)) + st["qo"].sum(1) + st["qr"].sum((1, 2)) + st["qh"]) * (10.0 / 60.0)
    return cost


def fleet_mpc_plan(world, wm_params, ctrl_idx, level, hold, s_f, e, horizon_steps=60, block_steps=20,
                   pop=48, iters=3, elite=8, seed=0, init=None):
    net = world.net
    nb = horizon_steps // block_steps
    nm = len(ctrl_idx)
    nlev = n_levers(level, hold)
    llo, lhi = lever_bounds(level, hold)
    lo = torch.tensor([300.0] * nm + llo).repeat(nb)
    hi = torch.tensor([1800.0] * nm + lhi).repeat(nb)
    mean = init.repeat(nb) if init is not None else (lo + hi) / 2
    std = (hi - lo) / 3
    g = torch.Generator().manual_seed(seed)
    p = dict(wm_params); p["kappa"] = torch.tensor(0.0)
    best = None
    for _ in range(iters):
        X = (mean + std * torch.randn(pop, nb * (nm + nlev), generator=g)).clamp(lo, hi)
        X[0] = mean.clamp(lo, hi)
        cost = fleet_rollout_cost(world, p, X, nb, nm, nlev, block_steps, ctrl_idx, level, hold, s_f, e)
        el = X[cost.argsort()[:elite]]
        mean, std = el.mean(0), el.std(0) + 1e-3
        best = X[cost.argmin()]
    first = best[:nm + nlev]
    meters = net.ramp_cap.clone()
    meters[ctrl_idx] = first[:nm]
    lever = first[nm:].tolist() if nlev else None
    return meters, lever, first


def run_day_fleet(net, wm_params, truth_params, agent, level, s_f, hold, day_seed, incident, ctrl_idx,
                  truth_model="ctm", empty_share=EMPTY_SHARE, decide_every=10):
    """agent: A0 (no control) or A2 (world-model MPC over meters and, for F1-F3, the fleet lever)."""
    torch.set_num_threads(1)
    world = FleetWorld(net, truth_params, level, s_f, hold, truth=True, day_seed=day_seed, incident=incident,
                       truth_model=truth_model, empty_share=empty_share)
    meters = net.ramp_cap.clone()
    if agent == "A2":
        meters[ctrl_idx] = 1200.0
    lever, init, n_dec, t_dec = None, None, 0, 0.0
    for t in range(ctm.N_STEPS):
        if agent == "A2" and t >= 60 and (t - 60) % decide_every == 0:
            c0 = time.time()
            meters, lever, init = fleet_mpc_plan(world, wm_params, ctrl_idx, level, hold, s_f, empty_share,
                                                 seed=day_seed * 1000 + t, init=init)
            t_dec += time.time() - c0; n_dec += 1
        world.step(meters, lever)
    fm = world.fm
    end_road = world.road_stock() * (10.0 / 60.0)
    end_hold = float(world.state["qh"].sum()) * (10.0 / 60.0)
    tts_road = fm["tts_road"] + end_road
    tts_all = tts_road + fm["hold_vh"] + end_hold
    ramp_avg_wait_min = world.m["ramp_wait"][ctrl_idx] / np.maximum(world.m["ramp_served"][ctrl_idx], 1.0) * 60.0
    pay = fm["pay_route"] + fm["pay_hold"]
    return {"agent": agent, "level": level, "s_f": s_f, "hold": hold, "truth": truth_model, "day": day_seed,
            "incident": incident, "tts_all_vh": tts_all, "tts_road_vh": tts_road, "hold_vh": fm["hold_vh"] + end_hold,
            "occ_vh": fm["occ_vh"], "emp_vh": fm["emp_vh"], "emp_vkt": fm["emp_vkt"],
            "human_avg_min": fm["human_vh"] / max(fm["human_served"], 1.0) * 60.0,
            "occ_avg_min": fm["occ_vh"] / max(fm["occ_served"], 1.0) * 60.0,
            "pay_route_usd": fm["pay_route"], "pay_hold_usd": fm["pay_hold"],
            "fleet_cost_usd": fm["occ_vh"] * VOT_TRUE[0] + fm["emp_vh"] * C_EMP - pay,
            "worst_ramp_wait_min": float(ramp_avg_wait_min.max()), "max_ramp_queue": world.m["max_q"],
            "decision_sec": t_dec / max(n_dec, 1)}
