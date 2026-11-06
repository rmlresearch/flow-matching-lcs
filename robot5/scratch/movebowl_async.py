"""Two numbers for an asynchronous stop-and-ask monitor, 'move the bowl' state 2 seed 0:
(1) at every replan t 20..70: fresh v1 (1 and 4 iterations), cosine with the previous replan's v1, and the
    jump along the previous v1 vs along the fresh one (is the 0.31 s stale-direction test a usable trigger?);
(2) at t 50, 60, 70: hold the prompt embedding at +-alpha v1 for alpha in (0.025, 0.1, 0.4) for the rest of
    the episode, 3 noises each -> can the prompt still flip the destination after the ridge has decayed?"""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/james/Projects/flow-matching-lcs/robot5")
import pi05flow
from pi05flow import H, D_ACT
from env5 import Scene, GOAL_NAMES, EVENT_NAMES, TERMINAL
from pridge5 import prompt_slots, plans, top_dir_E
import monitor5 as M

TEXT, SUITE, INIT, SEED = "move the bowl", "libero_goal", 2, 0
A = H * D_ACT
names = GOAL_NAMES + EVENT_NAMES
HOLD_T, HOLD_A, N_SIDE, T_BR = (50, 60, 70), (0.025, 0.1, 0.4), 3, 110
OUTJ = "/home/james/Projects/flow-matching-lcs/robot5/out/movebowl_async.json"


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
    out, t, prev = {"ridge": [], "hold": {}}, 0, None
    while t <= 70:
        o = s.obs_to_lerobot(obs, TEXT)
        z = rng.standard_normal((1, A))
        if t >= 20:
            x = s.get_state()
            E, pad, att = f.embed([o], [TEXT]); slots, _ = prompt_slots(f, o, TEXT)
            torch.manual_seed(0); v4, s4, r = top_dir_E(f, E, pad, att, slots, zp, n_iter=4)
            torch.manual_seed(0); v1, s1, _ = top_dir_E(f, E, pad, att, slots, zp, n_iter=1)
            dirs = {"fresh4": v4, "fresh1": v1}
            if prev is not None:
                dirs["prev4"] = prev
            Es = []
            for v in dirs.values():
                for a_ in M.ALPHAS_ON:
                    for sgn in (1, -1):
                        Ep = E.clone(); Ep[0, slots] += sgn * a_ * r * v; Es.append(Ep)
            P = plans(f, torch.cat(Es), pad, att, zp).mean(1).view(len(dirs), len(M.ALPHAS_ON), 2, -1)
            jump = (P[:, :, 0] - P[:, :, 1]).norm(dim=2).double().cpu().numpy()
            row = dict(t=t, sigma1_4=s4, sigma1_1=s1, cos_1_4=float((v1 * v4).sum().abs()),
                       cos_prev=None if prev is None else float((prev * v4).sum().abs()),
                       **{f"jump_{k}": jump[i].tolist() for i, k in enumerate(dirs)})
            out["ridge"].append(row)
            M.log(f"t {t}: s1 {s4:.1f} (1 iter {s1:.1f}, |cos| {row['cos_1_4']:.2f}); |cos(v1, prev)| {row['cos_prev']}; "
                  f"jump fresh " + " ".join(f"{x:.0f}" for x in jump[0]) + (" | along prev " + " ".join(f"{x:.0f}" for x in jump[-1]) if prev is not None else ""))
            prev = v4
            if t in HOLD_T:
                res = {}
                for a_ in HOLD_A:
                    for tag, sgn in (("+", 1), ("-", -1)):
                        res[f"{tag}{a_}"] = [side_rollout(f, s, x, t, slots, sgn * a_ * r * v4, np.random.default_rng(80_000 + 10 * t + b)) for b in range(N_SIDE)]
                out["hold"][str(t)] = res
                M.log(f"   hold at t {t}: " + "  ".join(f"{k}: {'/'.join(l.replace('bowl->', '') for l in v)}" for k, v in res.items()))
                json.dump(out, open(OUTJ, "w"))
            obs = s.set_state(x)
        a = f.unnorm(f.flow_np(z, f.prefix([o], [TEXT]))[0])
        for j in range(10):
            obs, _, _ = s.step(a[j]); t += 1
    json.dump(out, open(OUTJ, "w"))
    s.close()


if __name__ == "__main__":
    main()
