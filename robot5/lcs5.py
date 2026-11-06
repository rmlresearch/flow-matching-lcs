"""
LCS of pi0.5's first-chunk flow in noise space on LIBERO-spatial (robot4/lcs4.py ported).

    python lcs5.py calib     # chunk-level mode label vs the closed-loop outcome (rollouts_ambig)
    python lcs5.py scan      # FTLE, v1, line search along z + s v1, bisection of the label change
    python lcs5.py report

Chunk label. The chunk's position actions are OSC goal offsets of 0.05 m per unit, so the
predicted end-effector path is p_t = eef + GAIN * 0.05 * cumsum(a[:, :3]). The label is the bowl
nearest (in xy) to the predicted grasp point: p at the first step whose gripper command closes
(> 0), or the path's end if the chunk never closes the gripper. `calib` checks this against the
bowl that the closed-loop rollout actually lifts.
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from env5 import Scene, AMBIG  # noqa: E402

OUT = os.path.join(HERE, "out")
OSC = 0.05
GAIN = float(os.environ.get("GAIN", 1.0))
t00 = time.time()


def log(*a):
    print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)


def path(a_env, eef, gain=GAIN):
    """Predicted eef positions [..., H, 3] from env actions [..., H, 7]."""
    return eef + gain * OSC * np.cumsum(a_env[..., :3], axis=-2)


def grasp_index(a_env):
    """First step whose gripper command closes (> 0); H-1 if none. [...]"""
    close = a_env[..., 6] > 0
    return np.where(close.any(-1), close.argmax(-1), a_env.shape[-2] - 1)


def labels(a_env, eef, bowls, gain=GAIN, kind="grasp"):
    """Bowl (0/1) each chunk goes for. a_env [..., H, 7]; bowls [2, 3]."""
    p = path(a_env, eef, gain)[..., :2]                                    # [..., H, 2]
    d = np.linalg.norm(p[..., None, :] - bowls[:, :2], axis=-1)           # [..., H, 2]
    if kind == "grasp":
        g = grasp_index(a_env)
        dg = np.take_along_axis(d, g[..., None, None], axis=-2)[..., 0, :]
        return dg.argmin(-1)
    if kind == "end":
        return d[..., -1, :].argmin(-1)
    if kind == "closest":
        return d.min(-2).argmin(-1)
    raise ValueError(kind)


def initial(task, init, text):
    """Initial observation, eef, bowl positions of a scene (deterministic reset)."""
    s = Scene(task, init)
    obs = s.reset()
    o = s.obs_to_lerobot(obs, text)
    eef, bowls = s.eef(), s.bowls()
    s.close()
    return o, eef, bowls


def calib():
    import pi05flow
    from pi05flow import H, D_ACT
    R = json.load(open(os.path.join(OUT, "rollouts_ambig.json")))
    f = pi05flow.load()
    scenes = sorted({(r["task"], r["init"]) for r in R})
    rows = []
    for task, init in scenes:
        o, eef, bowls = initial(task, init, AMBIG)
        c = f.prefix([o], [AMBIG])
        rs = [r for r in R if (r["task"], r["init"]) == (task, init)]
        z = np.stack([np.random.default_rng(r["seed"]).standard_normal(H * D_ACT) for r in rs])
        a = f.unnorm(f.flow_np(z, c))
        g = grasp_index(a)
        for r, ai, gi in zip(rs, a, g):
            e = np.array(r["eef"])
            rows.append(dict(task=task, init=init, seed=r["seed"], lifted=r["lifted"], grasp_t=int(gi),
                             lift_t=r["lift_t"], a=ai, eef0=eef, bowls=bowls, eef_traj=e))
    lifted = np.array([r["lifted"] for r in rows])
    ok = lifted >= 0
    log(f"{len(rows)} episodes, {ok.sum()} lifted a bowl; first-chunk gripper closes within the chunk in "
        f"{np.mean([r['grasp_t'] < H - 1 for r in rows]):.2f} (median step {np.median([r['grasp_t'] for r in rows])}); "
        f"median lift step {np.median([r['lift_t'] for r in rows if r['lift_t'] > 0])}")
    # tracking gain: execute 2 first chunks per scene open loop (all 50 steps) and compare the
    # actual eef path with OSC * cumsum(a)
    pred, act = [], []
    for task, init in scenes:
        s = Scene(task, init)
        for r in [r for r in rows if (r["task"], r["init"]) == (task, init)][:2]:
            s.reset(); e = [s.eef()]
            for k in range(H):
                s.step(r["a"][k]); e.append(s.eef())
            pred.append(OSC * np.cumsum(r["a"][:, :3], 0)); act.append(np.array(e[1:]) - e[0])
        s.close()
    pred, act = np.stack(pred), np.stack(act)
    gain = float((pred * act).sum() / (pred * pred).sum())
    res = np.linalg.norm(act - gain * pred, axis=-1)
    log(f"tracking gain, open-loop first chunks (least squares, {len(pred)} chunks): {gain:.3f} | "
        f"path error at step 25 / 50: median {np.median(res[:, 24]):.3f} / {np.median(res[:, -1]):.3f} m "
        f"(path length {np.median(np.linalg.norm(act[:, -1], axis=-1)):.3f} m)")
    for kind in ("grasp", "end", "closest"):
        for gn in (0.5, 0.75, 1.0, gain):
            lab = np.array([labels(r["a"], r["eef0"], r["bowls"], gn, kind) for r in rows])
            log(f"label {kind:8s} gain {gn:.2f}: agreement with lifted bowl {np.mean(lab[ok] == lifted[ok]):.3f}"
                f" | P(label=bowl1) {np.mean(lab == 0):.2f}")
    np.savez_compressed(os.path.join(OUT, "calib.npz"), a=np.stack([r["a"] for r in rows]),
                        lifted=lifted, task=[r["task"] for r in rows], init=[r["init"] for r in rows],
                        eef0=np.stack([r["eef0"] for r in rows]), bowls=np.stack([r["bowls"] for r in rows]))


LABEL = os.environ.get("LABEL", "grasp")
S_GRID = np.round(np.arange(-40, 41) * 0.1, 10)
I0 = 40
N_BISECT = 12


def scan(n_init=2, n_seed=16, tag=""):
    """Per scene (10 tasks x n_init initial states) and prompt (ambiguous; the task's own):
    first-chunk Jacobian (all 350 task rows), FTLE, v1; labels on z + s v1; bisected margins."""
    import pi05flow, torch
    from pi05flow import H, D_ACT, ftle
    from env5 import suite
    A = H * D_ACT
    f = pi05flow.load()
    tasks = suite().tasks
    Z = np.random.default_rng(500).standard_normal((10, n_init, n_seed, A))
    path_ = os.path.join(OUT, f"lcs_scan{tag}.npz")
    res = dict(np.load(path_)) if os.path.exists(path_) else {}
    for task in range(10):
        for init in range(n_init):
            o, eef, bowls = initial(task, init, AMBIG)
            for pk, text in (("ambig", AMBIG), ("own", tasks[task].language)):
                key = f"{task}_{init}_{pk}"
                if f"ftle__{key}" in res:
                    continue
                c = f.prefix([o], [text])
                z = Z[task, init]

                def lab(zz):
                    a = f.unnorm(f.flow_np(zz, c, bs=128))
                    return labels(a, eef, bowls, GAIN, LABEL), a

                t0 = time.time()
                a0, J = f.jac(z, c, steps=H, bs=2, chunk=70)                       # [n, 350, 1600]
                tj = time.time() - t0
                _, sv, Vt = np.linalg.svd(J, full_matrices=False)
                v1 = Vt[:, 0]
                Jp = J.reshape(n_seed, H, 7, A)[:, :, :3].reshape(n_seed, -1, A)    # translation rows
                _, svp, Vtp = np.linalg.svd(Jp, full_matrices=False)
                zz = z[:, None] + S_GRID[None, :, None] * v1[:, None]
                L, a = lab(zz.reshape(-1, A))
                L = L.reshape(n_seed, len(S_GRID)); a = a.reshape(n_seed, len(S_GRID), -1)
                jump = np.linalg.norm(np.diff(a[..., [k for k in range(a.shape[-1]) if k % 7 < 3]], axis=1), axis=-1)
                l0 = L[:, I0]
                lo = np.full((n_seed, 2), np.nan); hi = lo.copy()
                for n in range(n_seed):
                    d = np.where(L[n] != l0[n])[0]
                    up, dn = d[d > I0], d[d < I0]
                    if len(up):
                        lo[n, 0], hi[n, 0] = S_GRID[up[0] - 1], S_GRID[up[0]]
                    if len(dn):
                        lo[n, 1], hi[n, 1] = S_GRID[dn[-1] + 1], S_GRID[dn[-1]]
                ok = ~np.isnan(lo); idx = np.argwhere(ok)
                for _ in range(N_BISECT if len(idx) else 0):
                    mid = 0.5 * (lo[ok] + hi[ok])
                    lm, _ = lab(z[idx[:, 0]] + mid[:, None] * v1[idx[:, 0]])
                    same = lm == l0[idx[:, 0]]
                    lo[ok] = np.where(same, mid, lo[ok]); hi[ok] = np.where(same, hi[ok], mid)
                hard = np.full((n_seed, 2), np.nan)
                if len(idx):
                    sg = np.sign(hi[ok] - lo[ok])
                    _, ain = lab(z[idx[:, 0]] + (lo[ok] - 0.05 * sg)[:, None] * v1[idx[:, 0]])
                    _, aout = lab(z[idx[:, 0]] + (hi[ok] + 0.05 * sg)[:, None] * v1[idx[:, 0]])
                    hard[ok] = np.linalg.norm((aout - ain)[..., :3].reshape(len(idx), -1), axis=1)
                sep = 0.5 * (lo + hi)
                out = dict(ftle=np.log(sv[:, 0]), ftle_pos=np.log(svp[:, 0]), sv=sv[:, :8], cos_v1=np.abs(
                    (v1 * Vtp[:, 0]).sum(1)), v1=v1.astype(np.float32), label=L, jump=jump.astype(np.float32),
                    d_plus=sep[:, 0], d_minus=-sep[:, 1], hard_plus=hard[:, 0], hard_minus=hard[:, 1],
                    eef=eef, bowls=bowls, z=z.astype(np.float32))
                res.update({f"{k}__{key}": v for k, v in out.items()})
                m = np.fmin(out["d_plus"], out["d_minus"])
                log(f"task {task} init {init} {pk:5s}: jac {tj:.0f}s | FTLE {np.median(out['ftle']):+.2f} "
                    f"| P(bowl1) {np.mean(l0 == 0):.2f} | boundary within 4: {np.mean(~np.isnan(m)):.2f} "
                    f"| median margin {np.nanmedian(m) if np.any(~np.isnan(m)) else np.nan:.2f}")
                np.savez_compressed(path_, **res)
                del J, Jp
                torch.cuda.empty_cache()


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "calib":
        calib()
    elif step == "scan":
        scan(*[int(x) for x in sys.argv[2:4]])
