"""At undecided replans of 'move the bowl' (state 2): is the eventual outcome in the chunk at all?
Replay the seed to the replan, draw 12 noises, roll each to the end (that chunk first, then fresh noise),
and compare between-outcome vs within-outcome plan distances, in cm (50-step plan, and the 10 executed steps)."""
import json, sys, time
import numpy as np
sys.path.insert(0, "/home/james/Projects/flow-matching-lcs/robot5")
import pi05flow
from pi05flow import H, D_ACT
from env5 import Scene, GOAL_NAMES, EVENT_NAMES, TERMINAL
from goal5 import path_cm
import monitor5 as M

TEXT, SUITE, INIT = "move the bowl", "libero_goal", 2
CASES = [(0, 40), (3, 30), (2, 20), (0, 0)]          # (seed, replan t)
N, T_BR = 12, 110
A = H * D_ACT
names = GOAL_NAMES + EVENT_NAMES
OUTJ = "/home/james/Projects/flow-matching-lcs/robot5/out/movebowl_spread.json"


def run_to(s, f, rng, t_stop):
    obs = s.set_state(s._x0); t = 0
    while t < t_stop:
        o = s.obs_to_lerobot(obs, TEXT)
        a = f.unnorm(f.flow_np(rng.standard_normal((1, A)), f.prefix([o], [TEXT]))[0])
        for j in range(10):
            obs, _, _ = s.step(a[j]); t += 1
    return obs


def main():
    f = pi05flow.load()
    out = {}
    s = Scene(0, INIT, 0, suite_name=SUITE); s.reset(); s._x0 = s.get_state()
    for seed, t0 in CASES:
        T = time.time()
        obs = run_to(s, f, np.random.default_rng(1000 + seed), t0)
        x = s.get_state()
        o = s.obs_to_lerobot(obs, TEXT); c = f.prefix([o], [TEXT])
        labs, P, P10 = [], [], []
        for b in range(N):
            rng = np.random.default_rng(90_000 + 1000 * seed + 10 * t0 + b)
            obs = s.set_state(x); tt, label = t0, None
            while tt < t0 + T_BR and label is None:
                oo = s.obs_to_lerobot(obs, TEXT)
                a = f.unnorm(f.flow_np(rng.standard_normal((1, A)), c if tt == t0 else f.prefix([oo], [TEXT]))[0])
                if tt == t0:
                    p = path_cm(a); P.append(p[:50].flatten()); P10.append(p[:10].flatten())
                for j in range(10):
                    obs, _, _ = s.step(a[j]); tt += 1
                    for i, q in enumerate(s.preds()):
                        if q and TERMINAL[i]:
                            label = names[i]; break
                    if label is not None:
                        break
            labs.append(label or "none")
        P, P10 = np.array(P), np.array(P10)
        D, D10 = np.linalg.norm(P[:, None] - P[None], axis=2), np.linalg.norm(P10[:, None] - P10[None], axis=2)
        L = np.array(labs); same = L[:, None] == L[None]; off = ~np.eye(N, dtype=bool)
        row = dict(seed=seed, t=t0, labels=labs,
                   within50=float(np.median(D[same & off])) if (same & off).any() else None, between50=float(np.median(D[~same])) if (~same).any() else None,
                   within10=float(np.median(D10[same & off])) if (same & off).any() else None, between10=float(np.median(D10[~same])) if (~same).any() else None)
        if (~same).any():
            m = sorted(set(labs)); Pm = [P[L == k].mean(0) for k in m]
            d = Pm[0] - Pm[1]; dh = d / np.linalg.norm(d)
            proj = (P - P.mean(0)) @ dh
            row["modes"], row["mean_sep50"] = m, float(np.linalg.norm(d))
            row["proj_by_label"] = {k: [round(float(v), 1) for v in proj[L == k]] for k in m}
        out[f"{seed}|{t0}"] = row
        M.log(f"seed {seed} t {t0}: {dict(zip(*np.unique(L, return_counts=True)))}  plan50 within {row['within50']} between {row['between50']} cm | "
              f"first10 within {row['within10']} between {row['between10']} cm | mean sep {row.get('mean_sep50')} proj {row.get('proj_by_label')}  ({time.time() - T:.0f}s)")
        json.dump(out, open(OUTJ, "w"))
    s.close()


if __name__ == "__main__":
    main()
