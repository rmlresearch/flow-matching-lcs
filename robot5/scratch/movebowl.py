"""'move the bowl' replan by replan: replay the monitor's rollouts (same seeds) and at every replan
record everything the approach has: ridge along the fixed t-0 v1, ridge along a refreshed v1 (sigma1,
cos to the t-0 direction), branching from the state (6 fresh-noise rollouts to the end, the ground
truth for 'still undecided'), the alignment of the refreshed v1 with the stove/plate gradient, and
what decode2 says when given the branched modes."""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/james/Projects/flow-matching-lcs/robot5")
import pi05flow
from pi05flow import H, D_ACT
from env5 import Scene, GOAL_NAMES, EVENT_NAMES, TERMINAL
from goal5 import path_cm
from pridge5 import prompt_slots, plans, top_dir_E
import monitor5 as M

TEXT, SUITE = "move the bowl", "libero_goal"
CASES = [(2, 0), (2, 2), (2, 1), (2, 3)]      # state 2, the four monitor seeds (0, 1, 3 stove; 2 plate)
N_BR, T_BR = 6, 110
OUTJ = "/home/james/Projects/flow-matching-lcs/robot5/out/movebowl_steps.json"
A = H * D_ACT
names = GOAL_NAMES + EVENT_NAMES


def branch(f, s, text, t, n, rng_base):
    """n rollouts from the current state with fresh noise until a terminal event or T_BR steps; returns labels
    and the first-chunk plans (cm, 150) of each branch; restores the state."""
    x = s.get_state()
    labs, P = [], []
    for b in range(n):
        obs = s.set_state(x); rng = np.random.default_rng(rng_base + b)
        tt, label = t, None
        while tt < t + T_BR and label is None:
            o = s.obs_to_lerobot(obs, text)
            a = f.unnorm(f.flow_np(rng.standard_normal((1, A)), f.prefix([o], [text]))[0])
            if tt == t:
                P.append(path_cm(a)[:50].flatten())
            for j in range(10):
                obs, _, _ = s.step(a[j]); tt += 1
                for i, q in enumerate(s.preds()):
                    if q and TERMINAL[i]:
                        label = names[i]; break
                if label is not None:
                    break
        labs.append(label or "none")
    s.set_state(x)
    return labs, np.array(P)


def main():
    f = pi05flow.load(); voc = M.dictionary(f)
    zp = np.random.default_rng(777).standard_normal((M.N_Z, A))
    out = json.load(open(OUTJ)) if __import__("os").path.exists(OUTJ) else {}
    for init, seed in CASES:
        key = f"{init}|{seed}"
        if key in out:
            continue
        s = Scene(0, init, 0, suite_name=SUITE); obs = s.reset(); x0 = s.get_state()
        obs = s.set_state(x0)
        rng = np.random.default_rng(1000 + seed)
        v10 = None
        t, label, rows = 0, None, []
        while t < s.max_steps and label is None:
            T0 = time.time()
            o = s.obs_to_lerobot(obs, TEXT)
            z = rng.standard_normal((1, A))
            E, pad, att = f.embed([o], [TEXT]); slots, _ = prompt_slots(f, o, TEXT)
            v1, s1, r = top_dir_E(f, E, pad, att, slots, zp)
            if v10 is None:
                v10 = v1.clone()
            Es = []
            for v in (v10, v1):
                for a_ in M.ALPHAS_ON:
                    for sgn in (1, -1):
                        Ep = E.clone(); Ep[0, slots] += sgn * a_ * r * v; Es.append(Ep)
            P = plans(f, torch.cat(Es), pad, att, zp).mean(1).view(2, len(M.ALPHAS_ON), 2, -1)
            jump = (P[:, :, 0] - P[:, :, 1]).norm(dim=2).double().cpu().numpy()      # [fixed/refreshed, alpha]
            labs, Pb = branch(f, s, TEXT, t, N_BR, 50_000 + 100 * seed + t)
            cnt = {l: labs.count(l) for l in set(labs)}
            a = f.unnorm(f.flow_np(z, f.prefix([o], [TEXT]))[0]); p_exec = path_cm(a)[:50]
            row = dict(t=t, sigma1=s1, cos_v1_v10=float((v1 * v10).sum()), jump_fixed=jump[0].tolist(), jump_fresh=jump[1].tolist(),
                       step_fixed=bool(M.is_step(jump[0])), step_fresh=bool(M.is_step(jump[1])), branches=cnt,
                       plan_end=p_exec[-1].tolist())
            modes = [l for l in cnt if l != "none"]
            if len(modes) >= 2:
                m0, m1 = sorted(modes)[:2]
                Pm = [Pb[[i for i, l in enumerate(labs) if l == m]].mean(0) for m in (m0, m1)]
                d = Pm[0] - Pm[1]; row["mode_sep_cm"] = float(np.linalg.norm(d)); row["modes"] = [m0, m1]
                row["exec_toward"] = float((np.linalg.norm(p_exec.flatten() - Pm[1]) - np.linalg.norm(p_exec.flatten() - Pm[0])) / np.linalg.norm(d))
                dh = torch.as_tensor(d / np.linalg.norm(d), dtype=torch.float32, device=f.dev)
                g = M.grad_E(f, E, pad, att, zp, dh)[slots]
                row["cos_v1_modegrad"] = float((v1 * g / g.norm()).sum()); row["modegrad_norm"] = float(g.norm())
                d2 = M.decode2(f, o, TEXT, zp, voc, sides=(torch.as_tensor(Pm[0], dtype=torch.float32, device=f.dev),
                                                           torch.as_tensor(Pm[1], dtype=torch.float32, device=f.dev)))
                row["decode"] = dict(cplus=d2["cplus"], cminus=d2["cminus"], toward_plus=d2["toward_plus"], toward_minus=d2["toward_minus"])
            rows.append(row)
            M.log(f"init {init} seed {seed} t {t:3d}: s1 {s1:5.1f} cos(v1,v1_0) {row['cos_v1_v10']:+.2f} jump fixed "
                  + " ".join(f"{x:.0f}" for x in jump[0]) + ("*" if row["step_fixed"] else "") + " fresh "
                  + " ".join(f"{x:.0f}" for x in jump[1]) + ("*" if row["step_fresh"] else "") + f" | branches {cnt}"
                  + (f" | sep {row['mode_sep_cm']:.0f} cm exec {row['exec_toward']:+.2f} cos(v1,grad) {row['cos_v1_modegrad']:+.2f} "
                     f"decode '{row['decode']['cplus']}' ({row['decode']['toward_plus']:+.2f}) / '{row['decode']['cminus']}' ({row['decode']['toward_minus']:+.2f})"
                     if "decode" in row else "") + f"  ({time.time() - T0:.0f}s)")
            for j in range(10):
                obs, _, _ = s.step(a[j]); t += 1
                for i, q in enumerate(s.preds()):
                    if q and TERMINAL[i]:
                        label = names[i]; break
                if label is not None or t >= s.max_steps:
                    break
        M.log(f"init {init} seed {seed}: {label} at t {t}")
        out[key] = dict(init=init, seed=seed, label=label, t=t, rows=rows)
        json.dump(out, open(OUTJ, "w"))
        s.close()


if __name__ == "__main__":
    main()
