"""
pi0.5 on PushT as a flow map (robot5/pi05flow.Pi05Flow with the PushT checkpoint: chunk 16 x 2,
noise space 16 x 32 = 512-D, 10 Euler steps).

    python probe6.py check          # explicit flow vs lerobot sample_actions; dPhi/dz vs central FD
    python probe6.py modes [n_st]   # n_st initial states x 256 seeds: chunk spread and 2-means
                                    #   split of the planned chunks -> out/modes.npz

Chunks are absolute agent targets (pixels, 0..512).
"""
import os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "robot5"))
from eval6 import make_env, to_lerobot, TASK, POLICIES, OUT  # noqa: E402

REPO = POLICIES["pi05"]


def load(dtype=torch.float32):
    import pi05flow
    return pi05flow.load(dtype, repo=REPO)


def states(seeds):
    env = make_env()
    out = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        out.append((obs, to_lerobot(obs)))
    return out


def check(n=3, n_dir=8, eps=1e-4):
    from pi05flow import cslice, N_STEPS, ftle
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    f = load()
    rng = np.random.default_rng(0)
    obs = [o for _, o in states(range(1000, 1000 + n))]
    z = rng.standard_normal((n, f.H * 32))
    with torch.no_grad():
        b = f.batch(obs, [TASK] * n)
        images, img_masks = f.pol._preprocess_images(b)
        ref = f.m.sample_actions(images, img_masks, b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK],
                                 noise=f._z(z), num_steps=N_STEPS)[..., :f.A].double().cpu().numpy()
        c32 = f.prefix(obs, [TASK] * n)
        mine = f.flow_np(z, c32)
    print(f"explicit flow vs lerobot sample_actions (fp32): max abs diff {np.abs(mine - ref).max():.2e} "
          f"(|a| max {np.abs(ref).max():.2f})", flush=True)
    post = f.post(torch.as_tensor(ref.reshape(-1, f.A), dtype=torch.float32)).double().cpu().numpy()
    print(f"unnorm vs lerobot postprocessor: max abs diff {np.abs(f.unnorm(ref).reshape(-1, f.A) - post).max():.2e} px",
          flush=True)
    f.set_dtype(torch.float64)
    c = f.recast(c32)
    a, J = f.jac(z, c, bs=1)
    U = rng.standard_normal((n_dir, f.H * 32)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    err = []
    for i in range(n):
        ci = cslice(c, i, i + 1)
        fd = ((f.flow_np(z[i] + eps * U, ci) - f.flow_np(z[i] - eps * U, ci)) / (2 * eps)).reshape(n_dir, -1)
        err.append(np.abs(U @ J[i].T - fd).max() / np.abs(fd).max())
    print(f"dPhi/dz [{J.shape[1]} x {J.shape[2]}] (float64) vs central FD, {n_dir} directions x {n} states: "
          f"max rel err {max(err):.2e} | FTLE {np.round(ftle(J), 3)}", flush=True)


def two_means(X, iters=50, seed=0):
    """2-means on rows of X. Returns labels, centroid gap / pooled within-cluster std (per dim)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(5):
        C = X[rng.choice(len(X), 2, replace=False)]
        for _ in range(iters):
            L = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
            C = np.stack([X[L == k].mean(0) if np.any(L == k) else C[k] for k in range(2)])
        w = sum(((X[L == k] - C[k]) ** 2).sum() for k in range(2))
        if best is None or w < best[0]:
            best = (w, L, C)
    w, L, C = best
    within = np.sqrt(w / (len(X) * X.shape[1]))
    return L, np.linalg.norm(C[0] - C[1]) / np.sqrt(X.shape[1]) / within


def modes(n_st=20, n_seed=256):
    from pi05flow import ftle
    f = load()
    seeds = list(range(1000, 1000 + n_st))
    S = states(seeds)
    Z = np.random.default_rng(7).standard_normal((n_seed, f.H * 32))
    res = dict(seeds=np.array(seeds), Z=Z.astype(np.float32))
    A, agent, F = [], [], []
    for (raw, o), s in zip(S, seeds):
        c = f.prefix([o], [TASK])
        a = f.unnorm(f.flow_np(Z, c, bs=256))                        # [n_seed, H, 2] pixels
        _, J = f.jac(Z[:16], c, bs=8)
        L, sep = two_means(a.reshape(n_seed, -1))
        fl = ftle(J)
        print(f"state {s}: chunk-end spread (std) {a[:, -1].std(0).round(1)} px | 2-means split "
              f"{np.mean(L == 0):.2f}/{np.mean(L == 1):.2f}, separation {sep:.1f} | FTLE median {np.median(fl):+.2f} "
              f"max {fl.max():+.2f}", flush=True)
        A.append(a.astype(np.float32)); agent.append(raw["agent_pos"]); F.append(fl)
    res.update(chunks=np.stack(A), agent=np.stack(agent), ftle=np.stack(F))
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "modes.npz"), **res)


def screen(n_init=100, n_roll=12, n_seed=128, n_exec=8):
    """Noise spread of the plan at many states: n_init initial states, plus every replan state
    along n_roll closed-loop rollouts (noise stream = rollout index). Per state: spread = max over
    chunk steps of the RMS std across n_seed seeds (px), and the 2-means separation."""
    f = load()
    env = make_env()
    Z = np.random.default_rng(11).standard_normal((n_seed, f.H * 32))
    rows = []

    def probe(o, kind, seed, t, raw):
        c = f.prefix([o], [TASK])
        a = f.unnorm(f.flow_np(Z, c, bs=128))
        spread = float(np.sqrt((a.std(0) ** 2).sum(-1)).max())
        _, sep = two_means(a.reshape(n_seed, -1))
        rows.append(dict(kind=kind, seed=seed, t=t, spread=spread, sep=float(sep), agent=raw["agent_pos"].copy(),
                         state=np.array(_pose(env)), chunks=a.astype(np.float32)))
        return c

    t0 = time.time()
    for s in range(1000, 1000 + n_init):
        obs, _ = env.reset(seed=s)
        probe(to_lerobot(obs), "init", s, 0, obs)
    print(f"[{time.time() - t0:.0f}s] {n_init} initial states", flush=True)
    for r in range(n_roll):
        obs, _ = env.reset(seed=2000 + r)
        rng = np.random.default_rng(r)
        for t in range(0, 300, n_exec):
            c = probe(to_lerobot(obs), "roll", 2000 + r, t, obs)
            a = f.unnorm(f.flow_np(rng.standard_normal((1, f.H * 32)), c)[0])
            done = False
            for k in range(n_exec):
                obs, _, term, trunc, info = env.step(a[k])
                if term or info.get("is_success", False):
                    done = True
                    break
            if done:
                break
        print(f"[{time.time() - t0:.0f}s] rollout {r}: {t // n_exec + 1} states, success {info.get('is_success')}",
              flush=True)
    sp = np.array([r["spread"] for r in rows]); sep = np.array([r["sep"] for r in rows])
    print(f"{len(rows)} states: plan spread across seeds (px) median {np.median(sp):.1f}, 90% {np.percentile(sp, 90):.1f}, "
          f"max {sp.max():.1f} | 2-means separation median {np.median(sep):.2f}, max {sep.max():.2f}")
    for i in np.argsort(-sep)[:8]:
        r = rows[i]
        print(f"   {r['kind']} seed {r['seed']} t {r['t']}: spread {r['spread']:.1f} px, separation {r['sep']:.2f}, "
              f"agent {np.round(r['agent'])}")
    np.savez_compressed(os.path.join(OUT, "screen.npz"), **{k: np.array([r[k] for r in rows]) for k in rows[0]})


def _pose(env):
    """Agent xy, block xy, block angle (gym-pusht's _set_state layout)."""
    u = env.unwrapped
    return [*u.agent.position, *u.block.position, u.block.angle]


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    elif sys.argv[1] == "modes":
        modes(*[int(x) for x in sys.argv[2:3]])
    elif sys.argv[1] == "screen":
        screen(*[int(x) for x in sys.argv[2:4]])
