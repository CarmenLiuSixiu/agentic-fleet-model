"""Quick test: does a negotiating fleet that judges offers with a forecast of the trip it will actually drive do
better than one that judges them with instantaneous route times?

The fleet (F2) normally evaluates route A vs B, both for its own route choice and for accepting the operator's offer,
with instantaneous route times (sum over cells of length / current speed). Here the fleet instead uses the
experienced travel time of a vehicle departing now, refreshed every 5 min right after the operator's decision:
  perfect - the true experienced time, from rolling the replayed world forward 40 min under the current controls
            (the fleet's information upper bound);
  noisy   - perfect + N(0, sd) per route and refresh, sd = 2.3 min, i.e. a Gaussian with the 1.85-min mean absolute
            error of the STID ensemble at the 30-min horizon on June (sd = MAE * sqrt(pi/2)).
Human drivers and the operator are unchanged (the operator still believes the fleet uses instantaneous times).
STID itself cannot be plugged in: it forecasts the real detectors, while the replayed world is a reconstruction.

Pre-registered reading (written before the run), 18 June test days, paired by day, 90% day-bootstrap:
- Primary: change in fleet riders' average minutes and in total vehicle-hours, F2-forecast minus F2-instantaneous.
  "Helps" if the rider-minute interval is below 0 and the vehicle-hour interval is not above +0.1%, or if the
  vehicle-hour interval is below 0.
- Secondary: route-choice regret, payments, and the gap to the obedient fleet (F1) in total vehicle-hours.
- If perfect does not help, the fleet's information does not matter on this corridor, and noisy is not interpreted.
Baselines (F1, F2-instantaneous) come from results/replay_test_runs_chrono.csv, same .venv environment, deterministic.
Usage: PEMS_SPLIT=chrono .venv/bin/python src/fleet_forecast.py [shares] [workers]
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PEMS_SPLIT", "chrono")
import numpy as np, pandas as pd, torch
from multiprocessing import Pool
import ctm, fleet_est
from fleet import fleet_probs, hold_share
from replay import ReplayFleetWorld, load_replay, day_slice, typical_path
from network import build_default_net
from run_experiments import controlled_ramps

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
HORIZON = 80            # 40 min of 30 s steps
NOISE_SD = 2.3          # min
REFRESH = 10            # steps (5 min), aligned with the operator's decisions


def experienced_tt(speeds, net, cells):
    """speeds: list of (M,) cell speeds over future steps; trajectory time (min) of a vehicle entering cells[0] now."""
    L = torch.as_tensor(net.L, dtype=torch.float32).expand(net.M)
    tau = 0.0                                          # elapsed steps
    for c in cells.tolist():
        v = speeds[min(int(tau), len(speeds) - 1)][c].clamp_min(3.0)
        tau += float(L[c] / v * 60.0) * 2.0              # minutes -> 30 s steps
    return tau / 2.0


class ForecastFleetWorld(ReplayFleetWorld):
    mode = "perfect"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.fc = None
        self.rng = np.random.default_rng(int("".join(ch for ch in str(self.day["days"]) if ch.isdigit())[-8:] or 0))

    def _rollout_speeds(self, meters, lever):
        lev = None if lever is None else torch.as_tensor(lever, dtype=torch.float32).reshape(1, -1)
        h = hold_share(self.flevel, lev, self.hold)
        st = {k: v.clone() for k, v in self.state.items()}
        def rp(a, b):
            return fleet_probs(self.flevel, lev, a, b, self.b0, self.p["b1"], self.vot)[0]
        out = []
        for k in range(HORIZON):
            t = min(self.t + k, ctm.N_STEPS - 1)
            st, info = self.sim(st, self.p, self.net, t, meters.reshape(1, -1), rp, cap_factor=self.cap_factor(),
                                demand=self.demand(t), p_agentic=0.0, fleet=(self.s_f, self.e), hold=h)
            out.append(info["speed"][0])
        return out

    def step(self, meters, lever):
        if self.t >= 60 and (self.t - 60) % REFRESH == 0:
            sp = self._rollout_speeds(meters, lever)
            fa, fb = experienced_tt(sp, self.net, self.net.a_idx), experienced_tt(sp, self.net, self.net.b_idx)
            if self.mode == "noisy":
                fa += self.rng.normal(0, NOISE_SD); fb += self.rng.normal(0, NOISE_SD)
            self.fc = (torch.tensor([fa]), torch.tensor([fb]))
        if self.fc is None:
            return super().step(meters, lever)
        # fleet columns (route choice and acceptance) use the forecast; humans keep instantaneous times
        import fleet as _fleet
        base = _fleet.fleet_probs
        fc = self.fc
        def patched(level, lev, a, b, b0, b1, vot):
            pH, _ = base(level, lev, a, b, b0, b1, vot)
            pF, ind = base(level, lev, fc[0].expand_as(a), fc[1].expand_as(b), b0, b1, vot)
            return torch.cat([pH[:, :2], pF[:, 2:]], 1), ind
        _fleet.fleet_probs = patched
        try:
            return super().step(meters, lever)
        finally:
            _fleet.fleet_probs = base


def job(a):
    i, s_f, mode = a
    torch.set_num_threads(1)
    import fleet as _fleet
    ForecastFleetWorld.mode = mode
    fleet_est.ReplayFleetWorld = ForecastFleetWorld
    net, _, _ = build_default_net()
    shared, days = load_replay("test")
    p_op = torch.load(typical_path())
    r = fleet_est.run_day_replay(net, shared, day_slice(days, i), p_op, "A2", "F2", s_f, controlled_ramps(net, p_op), seed=i)
    r["fc_mode"] = mode
    return r


if __name__ == "__main__":
    shares = tuple(float(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else (0.30,)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    shared, days = load_replay("test")
    jobs = list(itertools.product(range(len(days["days"])), shares, ("perfect", "noisy")))
    with Pool(workers) as p:
        df = pd.DataFrame(p.map(job, jobs))
    df.to_csv(os.path.join(ROOT, "results", "fleet_forecast_chrono.csv"), index=False)
    print("saved", len(df), "runs", flush=True)
