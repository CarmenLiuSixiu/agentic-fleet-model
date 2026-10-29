import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
torch.set_default_dtype(torch.float64)
import pytest
import ctm
from network import build_default_net

NET, SA, SB = build_default_net()


def run(p, steps, Bsz=2, p_agentic=0.3, meters=None, prob=0.5, cap_factor=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    st = ctm.init_state(NET, Bsz)
    hist = []
    for t in range(steps):
        m = meters if meters is not None else 300 + 1500 * torch.rand(Bsz, NET.R, generator=g)
        before = ctm.total_vehicles(st)
        d0, dr = ctm.demand_at(p, NET, t)
        st, info = ctm.step(st, p, NET, t, m, lambda a, b: torch.full((Bsz, 2), prob), cap_factor=cap_factor, p_agentic=p_agentic)
        after = ctm.total_vehicles(st)
        arrivals = (d0 + dr.sum()) * ctm.DT_H
        hist.append((before + arrivals - info["exit"].sum(1) - after).abs().max().item())
    return st, info, max(hist)


def test_cfl():
    vf = 110.0
    assert (vf * ctm.DT_H <= NET.L).all(), "free-flow travel per step must not exceed cell length"


def test_conservation():
    p = ctm.default_params(NET)
    _, _, err = run(p, 400)
    assert err < 1e-6, f"conservation error {err}"


def test_empty_network_free_flow():
    p = ctm.default_params(NET)
    st = ctm.init_state(NET, 1)
    cap = ctm.Q_LANE * NET.lanes * ctm.cap_mult(p)
    v = ctm.fd_speed(st["n"].sum(1), cap, torch.tensor(105.0), NET)
    assert torch.allclose(v, torch.full_like(v, 105.0))


def test_bottleneck_spillback():
    p = ctm.default_params(NET)
    p["log_d0"] = torch.full_like(p["log_d0"], torch.log(torch.tensor(5200.0)).item())
    bott = int(NET.a_idx[15])
    cf = torch.ones(1, NET.M); cf[0, bott] = 0.45
    st, info, err = run(p, 240, Bsz=1, prob=0.2, cap_factor=cf, meters=torch.full((1, NET.R), 1800.0))
    v = info["speed"][0]
    up = v[NET.a_idx[10:15]].mean().item(); down = v[NET.a_idx[17:22]].mean().item()
    assert up < 50 and down > 80, (up, down)
    assert err < 1e-6


def test_gradients_finite():
    p = {k: v.clone().requires_grad_(True) for k, v in ctm.default_params(NET).items()}
    st = ctm.init_state(NET, 1)
    loss = 0.0
    for t in range(120):
        st, info = ctm.step(st, p, NET, t, torch.full((1, NET.R), 1200.0),
                            lambda a, b: torch.sigmoid(p["b0"] + p["b1"] * (a - b)).reshape(1, 1).expand(1, 2))
        cap = info["cap"]
        loss = loss + ctm.fd_speed(st["n"].sum(1), cap, p["vf"][0], NET, smooth=3.0).mean()
    loss.backward()
    for k, v in p.items():
        assert v.grad is not None and torch.isfinite(v.grad).all(), k


def test_conservation_with_transit_exit():
    p = ctm.default_params(NET)
    cell = int(NET.b_idx[9])
    st = ctm.init_state(NET, 2)
    err, total_transit = 0.0, 0.0
    for t in range(300):
        before = ctm.total_vehicles(st)
        d0, dr = ctm.demand_at(p, NET, t)
        st, info = ctm.step(st, p, NET, t, torch.full((2, NET.R), 1500.0), lambda a, b: torch.full((2, 2), 0.7),
                            p_agentic=0.4, transit=lambda v: (cell, torch.tensor([0.3, 0.6])))
        after = ctm.total_vehicles(st)
        err = max(err, (before + (d0 + dr.sum()) * ctm.DT_H - info["exit"].sum(1) - after).abs().max().item())
        total_transit += info["transit_exit"].sum().item()
    assert err < 1e-6 and total_transit > 0
