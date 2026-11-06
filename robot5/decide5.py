"""
Where along a rollout does pi0.5 choose the bowl, and does the chunk FTLE see it?

Under the ambiguous prompt the first chunk of LIBERO-spatial scene (task 7, init 0) does not
commit: every seed's chunk ends between the two bowls, yet the closed-loop outcome is
seed-dependent. So the decision is made at a later replan. Here, along a main closed-loop
rollout, at every replan state s_k (every N_EXEC = 10 steps) until a bowl is lifted:

  - closed-loop basin of the current chunk's noise: for N_Z seeds z of this chunk, restore s_k
    exactly (physics + observable timers, see env5.Scene.get_state), execute z's chunk for 10
    steps, then continue closed loop with a *shared* future-noise stream until a bowl is
    lifted. Within one stream, outcome differences are caused by z alone (common random
    numbers). N_FUT streams.
  - FTLE of the chunk flow map at s_k for the same seeds: full chunk (350 rows), the executed
    10 steps (70 rows), and the translation rows (150).

    python decide5.py run [task] [init] [seed ...]     # default: 7 0 0 1
    python decide5.py report
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from env5 import Scene, AMBIG  # noqa: E402
from rollout5 import N_EXEC  # noqa: E402

OUT = os.path.join(HERE, "out")
N_Z, N_FUT = 8, 2
t00 = time.time()


def log(*a):
    print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)


def continue_closed_loop(f, s, obs, text, rng, t, t_max):
    """From the current state, closed loop until a bowl is lifted (or t_max). Returns the bowl."""
    from pi05flow import H, D_ACT
    while t < t_max:
        c = f.prefix([s.obs_to_lerobot(obs, text)], [text])
        a = f.unnorm(f.flow_np(rng.standard_normal((1, H * D_ACT)), c)[0])
        for k in range(N_EXEC):
            obs, _, _ = s.step(a[k]); t += 1
            b = s.lifted()
            if b >= 0:
                return b, t
    return -1, t


def run(task=7, init=0, seeds=(0, 1), text=AMBIG):
    import pi05flow
    from pi05flow import H, D_ACT
    A = H * D_ACT
    f = pi05flow.load()
    s = Scene(task, init)
    path = os.path.join(OUT, f"decide_{task}_{init}.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for seed in seeds:
        if str(seed) in res:
            continue
        rng = np.random.default_rng(seed)           # the main rollout's noise stream (as rollout5)
        obs = s.reset(); t = 0; states = []
        while t < s.max_steps:
            x = s.get_state()
            c = f.prefix([s.obs_to_lerobot(obs, text)], [text])
            k = t // N_EXEC
            zb = np.random.default_rng(20_000 + k).standard_normal((N_Z, A))
            ab, J = f.jac(zb, c, steps=H, bs=2, chunk=70)
            J4 = J.reshape(N_Z, H, 7, A)
            ftle = lambda M: np.log(np.linalg.svd(M, compute_uv=False)[:, 0])  # noqa: E731
            fl = dict(full=ftle(J), exec=ftle(J4[:, :N_EXEC].reshape(N_Z, -1, A)),
                      pos=ftle(J4[:, :, :3].reshape(N_Z, -1, A)))
            ab = f.unnorm(ab)
            out = np.full((N_Z, N_FUT), -1)
            for i in range(N_Z):
                for j in range(N_FUT):
                    s.set_state(x)
                    o, tt, b = obs, t, -1
                    for kk in range(N_EXEC):
                        o, _, _ = s.step(ab[i, kk]); tt += 1
                        b = s.lifted()
                        if b >= 0:
                            break
                    if b < 0:
                        b, _ = continue_closed_loop(f, s, o, text, np.random.default_rng(30_000 + j), tt, s.max_steps)
                    out[i, j] = b
            # resume the main rollout
            obs = s.set_state(x)
            a = f.unnorm(f.flow_np(rng.standard_normal((1, A)), c)[0])
            lifted = -1
            for kk in range(N_EXEC):
                obs, _, _ = s.step(a[kk]); t += 1
                lifted = s.lifted()
                if lifted >= 0:
                    break
            states.append(dict(t=t - kk - 1, eef=s.eef().tolist(), outcome=out.tolist(),
                               **{f"ftle_{k_}": v.tolist() for k_, v in fl.items()}))
            p1 = np.mean(out == 0); pn = np.mean(out < 0)
            dz = np.mean([len(set(out[:, j])) > 1 for j in range(N_FUT)])
            log(f"seed {seed} t {states[-1]['t']:3d}: P(bowl1) {p1:.2f} (none {pn:.2f}) | outcome depends on z "
                f"in {dz:.1f} of streams | FTLE full {np.median(fl['full']):+.2f} exec {np.median(fl['exec']):+.2f} "
                f"pos {np.median(fl['pos']):+.2f}")
            if lifted >= 0:
                break
        res[str(seed)] = dict(lifted=lifted, states=states)
        log(f"seed {seed}: main rollout lifted bowl {lifted + 1 if lifted >= 0 else 'none'} at t {t}")
        json.dump(res, open(path, "w"))
    s.close()


if __name__ == "__main__":
    if sys.argv[1] == "run":
        a = [int(x) for x in sys.argv[2:]]
        run(*(a[:2] if a else []), **({"seeds": a[2:]} if len(a) > 2 else {}))
