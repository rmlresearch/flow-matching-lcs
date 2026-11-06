"""Open frames for the VLM naming test on more cases. For each (prompt, init) and seed: replay the
episode; at every replan the fresh v1 (1 iteration) and the 2.5 % jump along it; where the jump is
>= OPEN cm (at most N_OPEN replans per episode) the 4-iteration v1, the two side plans (world and
projected (row, col)), the policy's image, and the destination with the embedding held on either
side (N_SIDE noises each; the truth for grading) -> out/vlm_cases.npz + out/vlm_cases.json."""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/james/Projects/flow-matching-lcs/robot5")
import pi05flow
from pi05flow import H, D_ACT
from env5 import Scene, GOAL_NAMES, EVENT_NAMES, TERMINAL
from goal5 import path_cm
from pridge5 import prompt_slots, plans, top_dir_E
import monitor5 as M
from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera

CASES = [("move the bowl", 3), ("tidy up the table", 2), ("put something on the plate", 0), ("put the cream cheese on the stove", 2),
         ("put the cream cheese on the plate", 3), ("open a drawer", 1), ("open the drawer", 3), ("move the bowl", 2)]
SEEDS = (0, 1)
SUITE, ALPHA, OPEN, N_OPEN, N_SIDE, T_BR, T_MAX = "libero_goal", 0.025, 50.0, 3, 2, 110, 130
A = H * D_ACT
names = GOAL_NAMES + EVENT_NAMES
OUTN = "/home/james/Projects/flow-matching-lcs/robot5/out/vlm_cases.npz"
OUTJ = "/home/james/Projects/flow-matching-lcs/robot5/out/vlm_cases.json"


def side_rollout(f, s, x, t0, text, slots, dE, rng):
    obs = s.set_state(x); tt, label = t0, None
    while tt < t0 + T_BR and label is None:
        o = s.obs_to_lerobot(obs, text)
        E, pad, att = f.embed([o], [text]); E[0, slots] += dE
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
    data, out = {}, {}
    for text, init in CASES:
        for seed in SEEDS:
            key = f"{text}|{init}|{seed}"
            if key == "move the bowl|2|0":
                continue                                     # done (movebowl_fig / movebowl_overlay)
            T0 = time.time()
            s = Scene(0, init, 0, suite_name=SUITE); obs = s.reset(); x0 = s.get_state(); obs = s.set_state(x0)
            W2P = get_camera_transform_matrix(s.sim, "agentview", 256, 256)
            proj = lambda v: project_points_from_world_to_camera(np.asarray(v, float).reshape(-1, 3), W2P, 256, 256)
            rng = np.random.default_rng(1000 + seed)
            rows, n_open, t, label = [], 0, 0, None
            while t < T_MAX and label is None:
                o = s.obs_to_lerobot(obs, text)
                z = rng.standard_normal((1, A)); x = s.get_state()
                E, pad, att = f.embed([o], [text]); slots, _ = prompt_slots(f, o, text)
                torch.manual_seed(0); v1, s1, r = top_dir_E(f, E, pad, att, slots, zp, n_iter=1)
                Es = [E.clone(), E.clone()]; Es[0][0, slots] += ALPHA * r * v1; Es[1][0, slots] -= ALPHA * r * v1
                P = plans(f, torch.cat(Es), pad, att, zp).mean(1)
                jump1 = float((P[0] - P[1]).norm())
                row = dict(t=t, sigma1_1=s1, jump1=jump1)
                if jump1 >= OPEN and n_open < N_OPEN:
                    n_open += 1
                    torch.manual_seed(0); v4, s4, r = top_dir_E(f, E, pad, att, slots, zp, n_iter=4)
                    Es = [E.clone(), E.clone()]; Es[0][0, slots] += ALPHA * r * v4; Es[1][0, slots] -= ALPHA * r * v4
                    Pm = plans(f, torch.cat(Es), pad, att, zp).mean(1).view(2, 50, 3).double().cpu().numpy() / 100
                    eef = s.eef(); img = np.asarray(obs["pixels"]["image"])
                    hold = {}
                    for tag, sgn in (("plus", 1), ("minus", -1)):
                        hold[tag] = [side_rollout(f, s, x, t, text, slots, sgn * ALPHA * r * v4, np.random.default_rng(60_000 + 10 * t + b)) for b in range(N_SIDE)]
                    obs = s.set_state(x)
                    fk = f"{key}|{t}"
                    data[f"img|{fk}"] = img
                    for tag, i in (("plus", 0), ("minus", 1)):
                        data[f"path_{tag}|{fk}"] = eef + Pm[i]; data[f"pix_{tag}|{fk}"] = proj(eef + Pm[i])
                    row.update(open=True, sigma1_4=s4, jump4=float(np.linalg.norm(Pm[0] - Pm[1]) * 100), hold=hold, eef=eef.tolist())
                    M.log(f"  {key} t {t}: OPEN s1 {s4:.1f} jump {row['jump4']:.0f} cm; hold + {hold['plus']}  - {hold['minus']}")
                rows.append(row)
                a = f.unnorm(f.flow_np(z, f.prefix([o], [text]))[0])
                for j in range(10):
                    obs, _, _ = s.step(a[j]); t += 1
                    for i, q in enumerate(s.preds()):
                        if q and TERMINAL[i]:
                            label = names[i]; break
                    if label is not None:
                        break
            out[key] = dict(text=text, init=init, seed=seed, label=label or "none", t_end=t, rows=rows)
            M.log(f"{key}: ended {label} at t {t}; open replans {[r['t'] for r in rows if r.get('open')]}; jumps " + " ".join(f"{r['jump1']:.0f}" for r in rows) + f"  ({time.time() - T0:.0f}s)")
            json.dump(out, open(OUTJ, "w")); np.savez(OUTN, **data)
            s.close()


if __name__ == "__main__":
    main()
