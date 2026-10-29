"""Replay worlds: a 2026 day reconstructed from its calibrated demand (calibrate_daily_2026, day_params_*), used as
the simulated 'true world'. The operator never sees these day parameters: it plans with a typical-day model (shared
road parameters + the mean of the training days' demand multipliers) and only observes detectors up to the decision
time. The replay is a simulation built from that day's data, not the real day itself.
"""
from __future__ import annotations
import os
import numpy as np
import torch
import ctm
from fleet import FleetWorld

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAG = "_d26_tau6s"


def load_replay(part):
    """part: 'train' | 'tune' | 'test'. File tags follow the active split (src/split.py)."""
    import split
    tg = TAG + split.tag()
    shared = {k: v.float() for k, v in torch.load(os.path.join(ROOT, "results", f"wm_params{tg}.pt")).items()}
    days = torch.load(os.path.join(ROOT, "results", f"day_params{tg}_{part}.pt"))
    return shared, days


def typical_path():
    import split
    return os.path.join(ROOT, "results", f"wm_params_2026_typical{split.tag()}.pt")


def day_slice(days, i):
    return {k: (v[i] if k != "days" else v[i]) for k, v in days.items()}


def typical_day_params(shared, net, train_days):
    """Operator's world model: shared road + mean training-day demand multipliers folded into log_d0 / log_r,
    mean measured US-101 inflow, mean route bias."""
    p = {k: v.clone() for k, v in shared.items()}
    z0, zA, zB = train_days["z0"].mean(0), train_days["zA"].mean(0), train_days["zB"].mean(0)
    p["log_d0"] = shared["log_d0"] + z0
    zr = torch.stack([zA, zB], 0)[net.ramp_route - 1]          # (R, N_BLOCKS)
    p["log_r"] = shared["log_r"] + zr
    p["bnd_major_B"] = train_days["bnd"].mean(0)
    p["b0"] = shared["b0"] + train_days["bias"].mean()
    return p


class ReplayFleetWorld(FleetWorld):
    """FleetWorld whose demand, US-101 inflow and human route bias come from one calibrated 2026 day.
    No synthetic incidents or capacity drop: the day multipliers already encode that day's conditions."""

    def __init__(self, net, shared, day, level, s_f, hold, empty_share=0.41):
        super().__init__(net, shared, level, s_f, hold, truth=True, day_seed=0, incident=False, empty_share=empty_share)
        self.day = day
        self.b0 = shared["b0"] + day["bias"]

    def demand(self, t):
        blk = min(ctm.N_BLOCKS - 1, t // (2 * ctm.BLOCK_MIN))
        d0 = torch.exp(self.p["log_d0"][blk] + self.day["z0"][blk]).reshape(1)
        zr = torch.stack([self.day["zA"][blk], self.day["zB"][blk]])[self.net.ramp_route - 1]
        dr = torch.exp(self.p["log_r"][:, blk] + zr)
        majB = int(torch.where(self.net.ramp_major & (self.net.ramp_route == 2))[0])
        dr = torch.cat([dr[:majB], self.day["bnd"][blk].reshape(1), dr[majB + 1:]]).reshape(1, -1)
        return d0, dr

    def cap_factor(self):
        return None


def day_folded_params(shared, net, day):
    """Oracle planner: the day's own demand multipliers, US-101 inflow and route bias folded into the parameters."""
    p = {k: v.clone() for k, v in shared.items()}
    p["log_d0"] = shared["log_d0"] + day["z0"]
    p["log_r"] = shared["log_r"] + torch.stack([day["zA"], day["zB"]], 0)[net.ramp_route - 1]
    p["bnd_major_B"] = day["bnd"].clone()
    p["b0"] = shared["b0"] + day["bias"]
    return p


if __name__ == "__main__":
    # build the operator's typical-day parameters from the training days of the active split
    from network import build_default_net
    net, _, _ = build_default_net()
    shared, tr = load_replay("train")
    torch.save(typical_day_params(shared, net, tr), typical_path())
    print("saved", typical_path(), "from", len(tr["days"]), "training days")
