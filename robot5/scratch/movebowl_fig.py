"""Data for figs/ask_pi05.png: the asynchronous stop-and-ask loop replayed on 'move the bowl', state 2,
seed 0, with the measured timings (RTX 5090, 20 Hz control, replan every 10 steps):
  - background v1 refresh, 1 power iteration = 1.0 s = 20 steps, serial: the refresh of the state at
    replan t_k is delivered at t_k + 20 and the next one starts on the state there;
  - at every replan, the jump test |P(+a r v) - P(-a r v)| along the newest delivered v, 0.31 s = 6 steps,
    fires when the a = 0.025 jump >= TRIG cm; the robot stops when the result arrives (t + 6);
  - on the frozen state: fresh v1 + the two side plans + projection, 1.3 s = 26 steps; the question;
  - the answer = hold the prompt embedding on the chosen side and resume (3 noises each side).
At every replan t 0..80: the agentview image, eef, executed chunk path, fresh v1 and sigma1 (1 iteration),
the side plans at a = 0.025 (world, projected), the jump along every earlier replan's v1, and the
jump along the direction the asynchronous loop actually holds."""
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

TEXT, SUITE, INIT, SEED = "move the bowl", "libero_goal", 2, 0
ALPHA, ALPHAS, N_SIDE, T_BR, T_MAX = 0.025, (0.025, 0.05, 0.1), 3, 110, 80
TAG = sys.argv[1] if len(sys.argv) > 1 else "r20a3"
REFRESH = int(sys.argv[2]) if len(sys.argv) > 2 else 20        # steps per background refresh (measured: 1.0 s = 20)
N_ALPHA = int(sys.argv[3]) if len(sys.argv) > 3 else 3         # alphas in the trigger test (3 = 0.31 s; 1 = timed here)
LAG_FIX = int(sys.argv[4]) if len(sys.argv) > 4 else None     # fixed test lag in steps (else measured, rounded up)
N_IT_FROZEN = int(sys.argv[5]) if len(sys.argv) > 5 else 1    # power iterations for the full test on the frozen state
FULL_LAG, TRIG = 26, 50.0                                      # steps; cm
A = H * D_ACT
names = GOAL_NAMES + EVENT_NAMES
OUTN = f"/home/james/Projects/flow-matching-lcs/robot5/out/movebowl_fig_{TAG}.npz"
OUTJ = f"/home/james/Projects/flow-matching-lcs/robot5/out/movebowl_fig_{TAG}.json"


def side_rollout(f, s, x, t0, slots, dE, rng):
    """Closed loop from state x with E[slots] += dE at every replan -> (label, t_end, final image, eef path)."""
    obs = s.set_state(x); tt, label, path = t0, None, []
    while tt < t0 + T_BR and label is None:
        o = s.obs_to_lerobot(obs, TEXT)
        E, pad, att = f.embed([o], [TEXT]); E[0, slots] += dE
        with torch.no_grad():
            c = f.prefix_from(E, pad, att)
        a = f.unnorm(f.flow_np(rng.standard_normal((1, A)), c)[0])
        for j in range(10):
            obs, _, _ = s.step(a[j]); tt += 1; path.append(s.eef().copy())
            for i, q in enumerate(s.preds()):
                if q and TERMINAL[i]:
                    label = names[i]; break
            if label is not None:
                break
    return label or "none", tt, np.asarray(obs["pixels"]["image"]), np.array(path)


def newest_dir(t):
    """Replan whose v1 the background refresh has delivered by replan t (serial, REFRESH steps each)."""
    k = (t - REFRESH) // REFRESH * REFRESH
    return k if t >= REFRESH else None


def main():
    f = pi05flow.load()
    zp = np.random.default_rng(777).standard_normal((M.N_Z, A))
    s = Scene(0, INIT, 0, suite_name=SUITE); obs = s.reset(); x0 = s.get_state(); obs = s.set_state(x0)
    W2P = get_camera_transform_matrix(s.sim, "agentview", 256, 256)
    proj = lambda v: project_points_from_world_to_camera(np.asarray(v, float).reshape(-1, 3), W2P, 256, 256)
    rng = np.random.default_rng(1000 + SEED)
    data, rows, dirs, t = {}, [], {}, 0
    stop = None
    while t <= T_MAX and stop is None:
        T0 = time.time()
        o = s.obs_to_lerobot(obs, TEXT)
        z = rng.standard_normal((1, A))
        x = s.get_state(); eef = s.eef()
        img = np.asarray(obs["pixels"]["image"])
        E, pad, att = f.embed([o], [TEXT]); slots, _ = prompt_slots(f, o, TEXT)
        torch.manual_seed(0); v1, s1, r = top_dir_E(f, E, pad, att, slots, zp, n_iter=1)
        dirs[t] = v1
        # jumps along every delivered-or-not direction so far (fresh = this replan's own)
        keys = sorted(dirs)
        Es = []
        for k in keys:
            for a_ in ALPHAS:
                for sgn in (1, -1):
                    Ep = E.clone(); Ep[0, slots] += sgn * a_ * r * dirs[k]; Es.append(Ep)
        Es.append(E.clone())
        P = plans(f, torch.cat(Es), pad, att, zp).mean(1)
        Pb = P[-1].view(50, 3).double().cpu().numpy() / 100
        Pj = P[:-1].view(len(keys), len(ALPHAS), 2, -1)
        jump = (Pj[:, :, 0] - Pj[:, :, 1]).norm(dim=2).double().cpu().numpy()
        jumps = {str(k): jump[i].tolist() for i, k in enumerate(keys)}
        i_f = keys.index(t); i_a = ALPHAS.index(ALPHA)
        Pside = Pj[i_f, i_a].view(2, 50, 3).double().cpu().numpy() / 100          # fresh v1, +/- ALPHA, m rel. eef
        a_exec = f.unnorm(f.flow_np(z, f.prefix([o], [TEXT]))[0]); p_exec = path_cm(a_exec)[:50] / 100
        nd = newest_dir(t)
        j_async, t_test, test_lag = None, None, None
        if nd is not None:                                   # the trigger test as the loop would run it, timed
            Et = []
            for a_ in ALPHAS[:N_ALPHA]:
                for sgn in (1, -1):
                    Ep = E.clone(); Ep[0, slots] += sgn * a_ * r * dirs[nd]; Et.append(Ep)
            torch.cuda.synchronize(); T1 = time.time()
            Pt = plans(f, torch.cat(Et), pad, att, zp).mean(1).view(N_ALPHA, 2, -1)
            torch.cuda.synchronize(); t_test = time.time() - T1
            j_async = (Pt[:, 0] - Pt[:, 1]).norm(dim=1).double().cpu().numpy().tolist()
            test_lag = LAG_FIX if LAG_FIX else max(1, int(np.ceil(t_test * 20)))
        fire = j_async is not None and j_async[0] >= TRIG
        row = dict(t=t, sigma1=s1, r=r, eef=eef.tolist(), jumps=jumps, newest_dir=nd, jump_async=j_async, fire=fire, t_test_s=t_test, test_lag=test_lag,
                   cos_prev={str(k): float((dirs[k] * v1).sum().abs()) for k in keys if k != t},
                   bowl=s.body_pos("akita_black_bowl_1").tolist(), plate=s.body_pos("plate_1").tolist(), stove=s.body_pos("flat_stove_1").tolist())
        rows.append(row)
        data[f"img_{t}"] = img
        for k, v in dict(exec=eef + p_exec, base=eef + Pb, plus=eef + Pside[0], minus=eef + Pside[1]).items():
            data[f"path_{k}_{t}"] = v; data[f"pix_{k}_{t}"] = proj(v)
        for k in ("eef", "bowl", "plate", "stove"):
            data[f"objpix_{k}_{t}"] = proj(row[k] if k != "eef" else eef)[0]
        M.log(f"t {t}: s1 {s1:.1f}; fresh jump " + "/".join(f"{x:.0f}" for x in jumps[str(t)]) +
              (f"; async dir t {nd}: " + "/".join(f"{x:.0f}" for x in j_async) + f" ({t_test:.2f}s = {test_lag} steps)" + (" FIRE" if fire else "") if nd is not None else "; async: no direction yet") +
              f"  ({time.time() - T0:.0f}s)")
        obs = s.set_state(x)
        a = f.unnorm(f.flow_np(z, f.prefix([o], [TEXT]))[0])
        n_step = test_lag if fire else 10
        for j in range(n_step):
            obs, _, _ = s.step(a[j]); t += 1
        if fire:
            stop = t
    out = dict(rows=rows, stop=stop, refresh=REFRESH, n_alpha=N_ALPHA, full_lag=FULL_LAG, trig=TRIG)
    if stop is not None:
        # the frozen state: the full test, the question, and the two answers
        T0 = time.time()
        o = s.obs_to_lerobot(obs, TEXT); x = s.get_state(); eef = s.eef()
        img = np.asarray(obs["pixels"]["image"])
        E, pad, att = f.embed([o], [TEXT]); slots, _ = prompt_slots(f, o, TEXT)
        torch.manual_seed(0); v1, s1, r = top_dir_E(f, E, pad, att, slots, zp, n_iter=N_IT_FROZEN)
        Es = [E.clone(), E.clone(), E.clone()]; Es[1][0, slots] += ALPHA * r * v1; Es[2][0, slots] -= ALPHA * r * v1
        Pm = plans(f, torch.cat(Es), pad, att, zp).mean(1).view(3, 50, 3).double().cpu().numpy() / 100
        t_full = time.time() - T0
        data[f"img_{stop}"] = img
        for k, v in dict(base=eef + Pm[0], plus=eef + Pm[1], minus=eef + Pm[2]).items():
            data[f"path_{k}_{stop}"] = v; data[f"pix_{k}_{stop}"] = proj(v)
        for k, v in dict(eef=eef, bowl=s.body_pos("akita_black_bowl_1"), plate=s.body_pos("plate_1"), stove=s.body_pos("flat_stove_1")).items():
            data[f"objpix_{k}_{stop}"] = proj(v)[0]
        res = {}
        for tag, sgn in (("plus", 1), ("minus", -1)):
            res[tag] = []
            for b in range(N_SIDE):
                lab, te, fimg, path = side_rollout(f, s, x, stop, slots, sgn * ALPHA * r * v1, np.random.default_rng(90_000 + b))
                res[tag].append((lab, te))
                data[f"final_{tag}_{b}"] = fimg; data[f"finalpath_{tag}_{b}"] = path; data[f"finalpix_{tag}_{b}"] = proj(path)
        s.set_state(x)
        out["frozen"] = dict(t=stop, sigma1=s1, r=r, jump=float(np.linalg.norm(Pm[1] - Pm[2]) * 100), t_full_s=t_full, n_iter=N_IT_FROZEN, hold=res)
        M.log(f"stopped at t {stop}: s1 {s1:.1f}, sides {out['frozen']['jump']:.0f} cm apart, full test {t_full:.1f}s; hold + {res['plus']}  - {res['minus']}")
    else:
        M.log("no trigger by t", T_MAX)
    np.savez(OUTN, **data); json.dump(out, open(OUTJ, "w"))
    s.close()


if __name__ == "__main__":
    main()
