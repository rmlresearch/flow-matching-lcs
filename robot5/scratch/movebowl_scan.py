"""Step-by-step scan of the steerable window, 'move the bowl' state 2 seed 0, along the base trajectory
(same noise as every movebowl_* script). At every control step t in [T0, T1]: fresh v1 with 1 and 4
power iterations, sigma1, the 2.5% jump along the fresh (4-iter) direction and along the directions found
at the replans t 40 / 50 / 60, and, for t in HOLD, the destination when the prompt embedding is held on
either side of (a) the fresh 4-iter direction, (b) the t-50 direction: N_SIDE noises each.
Question: is steerability intermittent along the trajectory (t 50 yes, t 57 no, t 60 yes, t 66+ no) or
does the one-iteration direction miss it at mid-chunk states?"""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/james/Projects/flow-matching-lcs/robot5")
import pi05flow
from pi05flow import H, D_ACT
from env5 import Scene, GOAL_NAMES, EVENT_NAMES, TERMINAL
from pridge5 import prompt_slots, plans, top_dir_E
import monitor5 as M

TEXT, SUITE, INIT, SEED = "move the bowl", "libero_goal", 2, 0
T0, T1, ALPHA, N_SIDE, T_BR = 46, 70, 0.025, 2, 110
HOLD = range(50, 71)
A = H * D_ACT
names = GOAL_NAMES + EVENT_NAMES
OUTJ = "/home/james/Projects/flow-matching-lcs/robot5/out/movebowl_scan.json"


def side_rollout(f, s, x, t0, slots, dE, rng):
    obs = s.set_state(x); tt, label = t0, None
    while tt < t0 + T_BR and label is None:
        o = s.obs_to_lerobot(obs, TEXT)
        E, pad, att = f.embed([o], [TEXT]); E[0, slots] += dE
        with torch.no_grad():
            c = f.prefix_from(E, pad, att)
        a = f.unnorm(f.flow_np(rng.standard_normal((1, A)), c)[0])
        for j in range(10):
            obs, _, _ = s.step(a[j]); tt += 1
            for i, q in enumerate(s.preds()):
                if q and TERMINAL[i]:
                    label = names[i]; break
            if label is not None:
                break
    return label or "none"


def main():
    f = pi05flow.load()
    zp = np.random.default_rng(777).standard_normal((M.N_Z, A))
    s = Scene(0, INIT, 0, suite_name=SUITE); obs = s.reset(); x0 = s.get_state(); obs = s.set_state(x0)
    rng = np.random.default_rng(1000 + SEED)
    states, t = {}, 0
    while t <= T1:                                       # the base trajectory; keep every state in [T0, T1]
        o = s.obs_to_lerobot(obs, TEXT)
        z = rng.standard_normal((1, A))
        a = f.unnorm(f.flow_np(z, f.prefix([o], [TEXT]))[0])
        for j in range(10):
            if T0 <= t <= T1:
                states[t] = (s.get_state(), obs)
            obs, _, _ = s.step(a[j]); t += 1
    M.log(f"states kept: {sorted(states)}")
    dirs, rows = {}, []
    for t in sorted(states):
        T = time.time(); x, obs = states[t]
        o = s.obs_to_lerobot(obs, TEXT)
        E, pad, att = f.embed([o], [TEXT]); slots, _ = prompt_slots(f, o, TEXT)
        torch.manual_seed(0); v4, s4, r = top_dir_E(f, E, pad, att, slots, zp, n_iter=4)
        torch.manual_seed(0); v1, s1, _ = top_dir_E(f, E, pad, att, slots, zp, n_iter=1)
        dirs[t] = v4
        keys = [k for k in (40, 50, 60) if k in dirs] if t >= 50 else []
        ds = {"fresh4": v4, "fresh1": v1, **{f"t{k}": dirs[k] for k in keys}}
        Es = []
        for v in ds.values():
            for sgn in (1, -1):
                Ep = E.clone(); Ep[0, slots] += sgn * ALPHA * r * v; Es.append(Ep)
        P = plans(f, torch.cat(Es), pad, att, zp).mean(1).view(len(ds), 2, -1)
        jump = (P[:, 0] - P[:, 1]).norm(dim=1).double().cpu().numpy()
        row = dict(t=t, sigma1_4=s4, sigma1_1=s1, cos_1_4=float((v1 * v4).sum().abs()),
                   cos_t50=float((dirs[50] * v4).sum().abs()) if 50 in dirs and t != 50 else None,
                   jump={k: float(jump[i]) for i, k in enumerate(ds)},
                   bowl_z=float(s.body_pos("akita_black_bowl_1")[2]))
        if t in HOLD:
            hold = {}
            for dk in ("fresh4", "t50"):
                if dk not in ds:
                    continue
                for tag, sgn in (("+", 1), ("-", -1)):
                    hold[f"{dk}{tag}"] = [side_rollout(f, s, x, t, slots, sgn * ALPHA * r * ds[dk], np.random.default_rng(95_000 + 10 * t + b)) for b in range(N_SIDE)]
            row["hold"] = hold
        rows.append(row)
        M.log(f"t {t}: s1 {s4:.1f} (1 iter {s1:.1f}, |cos| {row['cos_1_4']:.2f}); jump " +
              " ".join(f"{k} {v:.0f}" for k, v in row["jump"].items()) +
              (f"; |cos(v1, v1(t50))| {row['cos_t50']:.2f}" if row["cos_t50"] is not None else "") +
              (" | hold " + "  ".join(f"{k}: {'/'.join(l.replace('bowl->', '') for l in v)}" for k, v in row["hold"].items()) if "hold" in row else "") +
              f"  ({time.time() - T:.0f}s)")
        json.dump(dict(rows=rows, alpha=ALPHA, n_side=N_SIDE), open(OUTJ, "w"))
    s.close()


if __name__ == "__main__":
    main()
