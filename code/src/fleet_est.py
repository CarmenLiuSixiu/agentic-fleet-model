"""Detector-only operator for the fleet experiments on replayed 2026 days, with the A4 fallback.

Operator: plans with the typical-day world model (replay.typical_day_params); sees only noisy detector speeds up to
the decision time; re-estimates its state every 15 min (moving horizon, as operator_est) and plans with MPC every 5 min.
A4: negotiates (F2) by default and switches to empties-only negotiation (F3) while its realised prediction error is
high. Realised error = mean absolute difference, over the last 30 min of measured 5-min slots and all route detectors,
between what its internal model predicted for those slots (propagated from its last estimate) and what the detectors
then showed. The threshold is chosen on tuning days and locked for test days.
Extra metrics: route-choice regret of through vehicles (veh-h lost by being on the slower route at departure, using
true instantaneous route times), overall and only when the routes differ by more than 2 min.
"""
from __future__ import annotations
import time, types
from collections import deque
import numpy as np
import torch
import ctm
from operator_est import Estimator, NOISE_SD, EST_EVERY
from estimate import demand as scaled_demand
from fleet import fleet_probs, hold_share, fleet_mpc_plan, VOT_BELIEF, VOT_TRUE, C_EMP
from replay import ReplayFleetWorld

DECISION_GAP_MIN = 2.0


class FleetEstimator(Estimator):
    """nas=True ("negotiation as sensing"): the moving-horizon fit also matches the fleet's realised acceptance
    shares of the operator's offers, predicted through the operator's own acceptance model and route times."""

    def __init__(self, net, p, level, s_f, hold, e, cell_of, seed=0, nas=False, w_nas=4.0):
        super().__init__(net, p, 0, 0.0, 0.0, cell_of, seed=seed)
        self.flevel, self.s_f, self.hold, self.e = level, s_f, hold, e
        self.pred_acc, self.pred = 0.0, {}
        self.errs = deque(maxlen=6)
        self.nas, self.w_nas = nas, w_nas
        self.acc_obs = {}                      # step -> (accepted shares (2,), offer magnitude)
        self._ind = None

    def record_response(self, t, ind, lever):
        if lever is not None and self.flevel in ("F2", "F3") and abs(float(lever[0])) > 1e-3:
            self.acc_obs[t] = torch.as_tensor(ind, dtype=torch.float32).reshape(-1)[:2].clone()

    def _step(self, st, t, meters, lever, s=None):
        net, p = self.net, self.p
        lev = None if lever is None else torch.as_tensor(lever, dtype=torch.float32).reshape(1, -1)
        def rp(a, b):
            probs, ind = fleet_probs(self.flevel, lev, a, b, p["b0"], p["b1"], VOT_BELIEF)
            self._ind = ind
            return probs
        z = torch.zeros(1)
        dem = scaled_demand(p, net, t, s if s is not None else (z, z, z))
        return ctm.step(st, p, net, t, meters.reshape(1, -1), rp, demand=dem, fleet=(self.s_f, self.e),
                        hold=hold_share(self.flevel, lev, self.hold))

    @torch.no_grad()
    def predict(self, t, meters, lever):
        super().predict(t, meters, lever)
        if t >= 60:
            v = ctm.fd_speed(self.state["n"].sum(1), ctm.Q_LANE * self.net.lanes * ctm.cap_mult(self.p),
                             ctm.vf_cells(self.p, self.net), self.net)[0, self.cell_of]
            self.pred_acc = self.pred_acc + v
            if (t - 60) % ctm.STEPS_PER_5MIN == ctm.STEPS_PER_5MIN - 1:
                self.pred[t + 1] = self.pred_acc / ctm.STEPS_PER_5MIN
                self.pred_acc = 0.0

    def observe(self, t, true_speed_cells):
        super().observe(t, true_speed_cells)
        k = t + 1
        if k in self.obs and k in self.pred:
            self.errs.append(float((self.pred[k] - self.obs[k]).abs().mean()))

    def update(self, t):
        """Moving-horizon re-estimation (as operator_est.Estimator.update) plus, with nas, a term matching the
        fleet's realised acceptance shares over the window."""
        if not self.nas:
            return super().update(t)
        import time as _t
        t0 = t - 60
        ends = [e for e in range(t0 + ctm.STEPS_PER_5MIN, t + 1, ctm.STEPS_PER_5MIN) if e in self.obs]
        if t0 < 60 or len(ends) < 6 or t0 not in self.snap:
            return
        c0 = _t.time()
        prior = self.snap[t0]
        Y = torch.stack([self.obs[e] for e in ends])
        x = {"logm": torch.zeros(self.net.M), "logq": torch.zeros(1), "s0": torch.zeros(1), "sA": torch.zeros(1),
             "sB": torch.zeros(1)}
        x = {k: v.requires_grad_(True) for k, v in x.items()}
        opt = torch.optim.Adam(x.values(), lr=self.lr)
        vf = ctm.vf_cells(self.p, self.net)
        resp_steps = [k for k in range(60) if (t0 + k) in self.acc_obs]

        def run(sm):
            st = {"n": prior["n"] * torch.exp(x["logm"])[None, None, :], "qo": prior["qo"] * torch.exp(x["logq"]),
                  "qr": prior["qr"] * torch.exp(x["logq"])}
            Vs, acc, inds = [], 0.0, []
            for k in range(60):
                m, lv = self.ctrl[t0 + k]
                st, info = self._step(st, t0 + k, m, lv, (x["s0"], x["sA"], x["sB"]))
                if (t0 + k) in self.acc_obs:
                    inds.append((self._ind[0], self.acc_obs[t0 + k]))
                acc = acc + ctm.fd_speed(st["n"].sum(1), info["cap"], vf, self.net, smooth=sm)[0, self.cell_of]
                if k % ctm.STEPS_PER_5MIN == ctm.STEPS_PER_5MIN - 1:
                    Vs.append(acc / ctm.STEPS_PER_5MIN); acc = 0.0
            return st, torch.stack(Vs), inds

        for _ in range(self.iters):
            opt.zero_grad()
            _, V, inds = run(2.0)
            loss = (((V - Y) / 10.0) ** 2).mean() + 0.05 * (x["logm"] ** 2).mean() + 0.5 * (x["logm"].diff() ** 2).mean() \
                + 0.5 * sum((x[k] ** 2).mean() for k in ("logq", "s0", "sA", "sB"))
            if inds:
                pred = torch.stack([a for a, _ in inds]); obs = torch.stack([b for _, b in inds])
                loss = loss + self.w_nas * ((pred - obs) ** 2).mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                x["logm"].clamp_(-2.0, 2.0)
                for k in ("logq", "s0", "sA", "sB"):
                    x[k].clamp_(-1.0, 1.0)
        with torch.no_grad():
            st, _, _ = run(0.0)
        self.state = {k: v.detach() for k, v in st.items()}
        self.n_est += 1; self.sec += _t.time() - c0

    def realized_error(self):
        return float(np.mean(self.errs)) if len(self.errs) == self.errs.maxlen else 0.0


def run_day_replay(net, shared, day, p_op, agent, level, s_f, ctrl_idx, threshold=None, hold=False,
                   empty_share=0.41, decide_every=10, seed=0, oracle=False, nas=False, det_frac=1.0):
    """oracle: the planner knows the day's demand (p_op should be replay.day_folded_params) and the true state.
    nas: the estimator also uses the fleet's realised acceptance shares. det_frac: share of detector cells observed."""
    """agent A0 (no control) or A2 (detector-only MPC). level F0/F1/F2/F3, or 'A4' (F2 <-> F3 by realised error)."""
    torch.set_num_threads(1)
    adaptive = level == "A4"
    world = ReplayFleetWorld(net, shared, day, "F2" if adaptive else level, s_f, hold, empty_share)
    cell_of = torch.tensor(sorted(set(int(c) for c in net.sensor_cell.values())))
    if det_frac < 1.0:
        g = np.random.default_rng(1000 + seed)
        cell_of = torch.tensor(sorted(g.choice(cell_of.numpy(), max(2, int(round(det_frac * len(cell_of)))), replace=False)))
    est = FleetEstimator(net, p_op, "F2" if adaptive else level, s_f, hold, empty_share, cell_of, seed=seed, nas=nas)
    judge_err = []
    meters = net.ramp_cap.clone()
    if agent == "A2":
        meters[ctrl_idx] = 1200.0
    lever, init = None, None
    regret = {"all": 0.0, "decisive": 0.0, "human": 0.0, "fleet_occ": 0.0}
    decisive_steps, f3_decisions, n_dec, err_log = 0, 0, 0, []
    for t in range(ctm.N_STEPS):
        if agent == "A2" and not oracle and t >= 60 and (t - 60) % EST_EVERY == 0:
            est.update(t)
            with torch.no_grad():                 # operator's judgement of the two route times vs the truth
                ve = ctm.fd_speed(est.state["n"].sum(1), ctm.Q_LANE * net.lanes * ctm.cap_mult(p_op), ctm.vf_cells(p_op, net), net)
                a_e, b_e = ctm.route_tt_min(ve, net)
                vt = world.speed_cells({"cap": ctm.Q_LANE * net.lanes * ctm.cap_mult(world.p)})
                a_t, b_t = ctm.route_tt_min(vt, net)
                judge_err.append(float((a_e - a_t).abs() + (b_e - b_t).abs()) / 2.0)
        if agent == "A2" and t >= 60 and (t - 60) % decide_every == 0:
            if adaptive:
                mode = "F3" if est.realized_error() > threshold else "F2"
                world.flevel = est.flevel = mode
                f3_decisions += mode == "F3"
            err_log.append(est.realized_error())
            shim = types.SimpleNamespace(net=net, state=world.state if oracle else est.state, t=t)
            meters, lever, init = fleet_mpc_plan(shim, p_op, ctrl_idx, world.flevel, hold, s_f, empty_share,
                                                 seed=seed * 1000 + t, init=init)
            n_dec += 1
        if agent == "A2" and not oracle:
            est.predict(t, meters, lever)
        info = world.step(meters, lever)
        if agent == "A2" and not oracle:
            est.observe(t, world.speed_cells(info))
            est.record_response(t, info["ind"], lever)
        if t >= 60:                           # route-choice regret at the diverge, true instantaneous route times
            ttA, ttB = float(info["ttA"][0]), float(info["ttB"][0])
            gap = abs(ttA - ttB)
            pB = info["pB"][0]                                       # (O_H, O_A, F_occ, F_emp)
            flows = info["fwd"][0, ctm.THROUGH, 0]
            wrong = torch.where(torch.tensor(ttB > ttA), pB, 1 - pB)
            lost = (flows * wrong).numpy() * gap / 60.0               # veh-h
            regret["all"] += float(lost.sum())
            regret["human"] += float(lost[0]); regret["fleet_occ"] += float(lost[2])
            if gap > DECISION_GAP_MIN:
                regret["decisive"] += float(lost.sum()); decisive_steps += 1
    fm = world.fm
    end_road = world.road_stock() * (10.0 / 60.0)
    end_hold = float(world.state["qh"].sum()) * (10.0 / 60.0)
    tts_all = fm["tts_road"] + end_road + fm["hold_vh"] + end_hold
    ramp_wait = world.m["ramp_wait"][ctrl_idx] / np.maximum(world.m["ramp_served"][ctrl_idx], 1.0) * 60.0
    pay = fm["pay_route"] + fm["pay_hold"]
    return {"agent": agent, "level": level, "s_f": s_f, "threshold": threshold, "date": day["days"],
            "tts_all_vh": tts_all, "occ_avg_min": fm["occ_vh"] / max(fm["occ_served"], 1.0) * 60.0,
            "human_avg_min": fm["human_vh"] / max(fm["human_served"], 1.0) * 60.0,
            "pay_usd": pay, "regret_vh": regret["all"], "regret_decisive_vh": regret["decisive"],
            "regret_human_vh": regret["human"], "regret_fleet_occ_vh": regret["fleet_occ"],
            "decisive_share": decisive_steps / (ctm.N_STEPS - 60),
            "f3_share": f3_decisions / max(n_dec, 1), "realized_err_mean": float(np.mean(err_log)) if err_log else None,
            "judge_err_min": float(np.mean(judge_err)) if judge_err else None, "oracle": oracle, "nas": nas,
            "det_frac": det_frac,
            "worst_ramp_wait_min": float(ramp_wait.max())}
