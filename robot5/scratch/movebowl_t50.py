"""'move the bowl', state 2, seed 0: what the ridge is at t 40 / 50 / 60. At each replan: the scene
(eef, gripper, bowl, stove, plate), the fresh v1 and sigma1, the plans on both sides of the ridge
(endpoints in world cm) and the executed one, the words the decoder attaches to the two sides, and
closed-loop rollouts with the prompt embedding held on either side (3 noises each) -> outcomes."""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/james/Projects/flow-matching-lcs/robot5")
import pi05flow
from pi05flow import H, D_ACT
from env5 import Scene, GOAL_NAMES, EVENT_NAMES, TERMINAL
from goal5 import path_cm
from pridge5 import prompt_slots, plans, top_dir_E
import monitor5 as M

TEXT, SUITE, INIT, SEED = "move the bowl", "libero_goal", 2, 0
TS, ALPHA, N_SIDE, T_BR = (40, 50, 60), 0.025, 3, 110
A = H * D_ACT
names = GOAL_NAMES + EVENT_NAMES
OUTJ = "/home/james/Projects/flow-matching-lcs/robot5/out/movebowl_t50.json"


def body(s, name):
    try:
        return s.body_pos(name)
    except KeyError:
        return None


def side_rollout(f, s, x, t0, slots, dE, rng):
    """Closed loop from state x with E[slots] += dE at every replan; returns (label, t, eef path)."""
    obs = s.set_state(x); tt, label, path = t0, None, []
    while tt < t0 + T_BR and label is None:
        o = s.obs_to_lerobot(obs, TEXT)
        E, pad, att = f.embed([o], [TEXT]); E[0, slots] += dE
        with torch.no_grad():
            c = f.prefix_from(E, pad, att)
        a = f.unnorm(f.flow_np(rng.standard_normal((1, A)), c)[0])
        for j in range(10):
            obs, _, _ = s.step(a[j]); tt += 1; path.append(s.eef().tolist())
            for i, q in enumerate(s.preds()):
                if q and TERMINAL[i]:
                    label = names[i]; break
            if label is not None:
                break
    return label or "none", tt


def main():
    f = pi05flow.load(); voc = M.dictionary(f)
    zp = np.random.default_rng(777).standard_normal((M.N_Z, A))
    s = Scene(0, INIT, 0, suite_name=SUITE); obs = s.reset(); x0 = s.get_state(); obs = s.set_state(x0)
    e = s.env._env.env
    M.log("bodies:", [k for k in e.obj_body_id if any(w in k for w in ("stove", "plate", "bowl"))])
    rng = np.random.default_rng(1000 + SEED)
    out, t = {}, 0
    while t <= max(TS):
        o = s.obs_to_lerobot(obs, TEXT)
        z = rng.standard_normal((1, A))
        if t in TS:
            T = time.time(); x = s.get_state()
            eef = s.eef(); grip = np.asarray(obs["agent_pos"])[-2:] if "agent_pos" in obs else None
            scene = dict(eef=eef.tolist(), gripper=None if grip is None else grip.tolist(),
                         bowl=body(s, "akita_black_bowl_1").tolist(), plate=body(s, "plate_1").tolist(),
                         stove=body(s, "flat_stove_1").tolist())
            E, pad, att = f.embed([o], [TEXT]); slots, _ = prompt_slots(f, o, TEXT)
            v1, s1, r = top_dir_E(f, E, pad, att, slots, zp)
            Es = [E.clone(), E.clone(), E.clone()]; Es[1][0, slots] += ALPHA * r * v1; Es[2][0, slots] -= ALPHA * r * v1
            P = plans(f, torch.cat(Es), pad, att, zp)                       # [3, n_z, 150]
            Pm = P.mean(1).view(3, 50, 3).double().cpu().numpy()
            a_exec = f.unnorm(f.flow_np(z, f.prefix([o], [TEXT]))[0]); p_exec = path_cm(a_exec)[:50]
            ends = dict(exec=p_exec[-1].tolist(), base=Pm[0, -1].tolist(), plus=Pm[1, -1].tolist(), minus=Pm[2, -1].tolist())
            world = {k: (eef * 100 + np.array(v)).tolist() for k, v in ends.items()}   # cm
            d2 = M.decode2(f, o, TEXT, zp, voc, sides=(P[1].mean(0), P[2].mean(0)))
            res = {}
            for tag, sgn in (("plus", 1), ("minus", -1), ("base", 0)):
                res[tag] = [side_rollout(f, s, x, t, slots, sgn * ALPHA * r * v1, np.random.default_rng(70_000 + 10 * t + b)) for b in range(N_SIDE)]
            s.set_state(x)
            row = dict(t=t, sigma1=s1, r=r, scene=scene, jump=float(np.linalg.norm(Pm[1] - Pm[2])), plan_ends_rel_cm=ends, plan_ends_world_cm=world,
                       exec_toward=float((np.linalg.norm(p_exec.flatten() - Pm[2].flatten()) - np.linalg.norm(p_exec.flatten() - Pm[1].flatten())) / np.linalg.norm(Pm[1] - Pm[2])),
                       decode=dict(cplus=d2["cplus"], cminus=d2["cminus"], toward_plus=d2["toward_plus"], toward_minus=d2["toward_minus"], top=d2["top"][:6], bottom=d2["bottom"][-6:]),
                       side_rollouts=res)
            out[str(t)] = row
            M.log(f"t {t}: sigma1 {s1:.1f} jump {row['jump']:.0f} cm | eef {np.round(eef*100,1).tolist()} grip {grip} bowl {np.round(np.array(scene['bowl'])*100,1).tolist()} "
                  f"plate {np.round(np.array(scene['plate'])*100,1).tolist()} stove {np.round(np.array(scene['stove'])*100,1).tolist()}")
            M.log(f"   plan ends (world cm): exec {np.round(world['exec'],1).tolist()} base {np.round(world['base'],1).tolist()} + {np.round(world['plus'],1).tolist()} - {np.round(world['minus'],1).tolist()}  exec toward {row['exec_toward']:+.2f}")
            M.log(f"   decode: + '{d2['cplus']}' ({d2['toward_plus']:+.2f})  - '{d2['cminus']}' ({d2['toward_minus']:+.2f})")
            M.log(f"   side rollouts: + {res['plus']}  - {res['minus']}  base {res['base']}  ({time.time() - T:.0f}s)")
            json.dump(out, open(OUTJ, "w"))
            obs = s.set_state(x)
        a = f.unnorm(f.flow_np(z, f.prefix([o], [TEXT]))[0])
        for j in range(10):
            obs, _, _ = s.step(a[j]); t += 1
    s.close()


if __name__ == "__main__":
    main()
