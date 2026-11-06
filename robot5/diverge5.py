"""
Why do 1%-of-|z| perturbations flip the closed-loop label at a decision state (boundary5)?
Run a seed z0 and z0 + eps d (d a random unit direction) side by side from the same state, with the
same future stream, and log at every replan: eef distance between the two runs, whether pi0.5's
discretised state tokens in the prompt differ, how many image pixels differ, and the distance
between the two new plans' predicted eef paths.

    python diverge5.py "<prompt>" init seed_index eps [eps ...]   # t = 0, future stream 0
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SUITE, TASK0, LAYOUT, N_EXEC = "libero_goal", 0, 0, 10


def episode(f, s, text, a0, fut=0, max_t=300):
    from pi05flow import H, D_ACT
    from env5 import TERMINAL, GOAL_NAMES, EVENT_NAMES
    from goal5 import path_cm
    rng = np.random.default_rng(30_000 + fut)
    s.reset()
    a, t, log = a0, 0, []
    while t < max_t:
        for k in range(N_EXEC):
            obs, _, _ = s.step(a[k]); t += 1
            g = s.preds()
            if (g & TERMINAL).any():
                return (GOAL_NAMES + EVENT_NAMES)[int(np.argmax(g & TERMINAL))], log
        o = s.obs_to_lerobot(obs, text)
        b = f.batch([o], [text])
        c = f.prefix([o], [text])
        a = f.unnorm(f.flow_np(rng.standard_normal((1, H * D_ACT)), c)[0])
        imgs = np.concatenate([np.asarray(v).ravel() for k_, v in sorted(o.items()) if "image" in k_])
        log.append(dict(t=t, eef=s.eef(), tok=b["observation.language.tokens"][0].cpu().numpy(), img=imgs,
                        plan=path_cm(a)))
    return "none", log


def main(text, init, i, eps_list):
    import pi05flow
    from pi05flow import H, D_ACT
    from env5 import Scene
    f = pi05flow.load()
    s = Scene(TASK0, init, LAYOUT, suite_name=SUITE)
    o0 = s.obs_to_lerobot(s.reset(), text)
    c0 = f.prefix([o0], [text])
    z0 = np.random.default_rng(20_000).standard_normal((8, H * D_ACT))[i]
    d = np.random.default_rng(1).standard_normal(H * D_ACT); d /= np.linalg.norm(d)
    lab0, L0 = episode(f, s, text, f.unnorm(f.flow_np(z0[None], c0)[0]))
    print(f"base seed {i}: {lab0} after {L0[-1]['t'] if L0 else 0}+ steps", flush=True)
    for eps in eps_list:
        a = f.unnorm(f.flow_np((z0 + eps * d)[None], c0)[0])
        a_base = f.unnorm(f.flow_np(z0[None], c0)[0])
        lab, L = episode(f, s, text, a)
        print(f"\neps {eps:g}: label {lab} (base {lab0}); first-chunk action max diff {np.abs(a - a_base)[:N_EXEC].max():.2e}")
        for x, y in zip(L0, L):
            nt = int((x["tok"] != y["tok"]).sum())
            dimg = np.abs(x["img"].astype(float) - y["img"].astype(float))
            print(f"  t {x['t']:3d}: eef diff {1000 * np.linalg.norm(x['eef'] - y['eef']):7.3f} mm | state tokens differ: "
                  f"{nt} | image values differ: {int((dimg > 0).sum()):6d} (max {dimg.max():.3f}) | new plan path diff "
                  f"{np.abs(x['plan'] - y['plan']).max():6.2f} cm")
            if np.abs(x["plan"] - y["plan"]).max() > 5:
                break


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), [float(x) for x in sys.argv[4:]])
