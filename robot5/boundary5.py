"""
Closed-loop noise-space boundary at a decision state (branch5): does the chunk Jacobian's top
singular direction point at it?

At replan state t of goal5's / branch5's main rollout, with the future-noise stream fixed
(30000 + fut, as branch5), the label (goal5.label) is a deterministic function of this chunk's
noise z. From each of branch5's N_Z seeds z0 (rng 20000 + k), along rays z0 + s d:
  d = +v1, -v1     top right singular vector of the chunk's predicted eef path over the executed
                   10 steps (cm, order5.phys_jac) w.r.t. z
  d = +r1, -r1, +r2, -r2   random unit directions
at distances s in STEPS (|z| ~ 40). Record the first s at which the label differs from z0's.

    python boundary5.py run "<prompt>" init t [seed] [fut]   # -> out/boundary_<slug>_<init>_t<t>.json
    python boundary5.py report
"""
import glob, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "out")
SUITE, TASK0, LAYOUT = "libero_goal", 0, 0
W = int(os.environ.get("N_WORKERS", 8))
N_Z, N_EXEC = 8, 10
STEPS = [0.5, 1, 2, 4, 8, 16]
RAYS = ["+v1", "-v1", "+r1", "-r1", "+r2", "-r2"]
t00 = time.time()


def log(*a):
    print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)


def run(text, init, t_state, seed=0, fut=0):
    import pi05flow
    from pi05flow import H, D_ACT
    from vec5 import Vec
    from goal5 import label
    from order5 import phys_jac
    A = H * D_ACT
    f = pi05flow.load()
    v = Vec(W)
    idx = list(range(W))
    v.load(idx, [dict(task_id=TASK0, init_id=init, layout=LAYOUT, suite_name=SUITE)] * W)
    max_t = v.info[0]["max_steps"]
    obs = v.reset(idx)
    rng = np.random.default_rng(seed)                     # main rollout up to t_state (as branch5)
    for k in range(t_state // N_EXEC):
        c = f.prefix([obs[0]], [text])
        a = f.unnorm(f.flow_np(rng.standard_normal((1, A)), c)[0])
        outs = v._call(idx, "steps", [(a[:N_EXEC], "goal", max_t)] * W)
        obs = [o.pop("obs") for o in outs]
    v.save(idx, "s")
    k = t_state // N_EXEC
    c = f.prefix([obs[0]], [text])
    Z0 = np.random.default_rng(20_000 + k).standard_normal((N_Z, A))
    _, J = f.jac(Z0, c, steps=N_EXEC, bs=2, chunk=70)
    P = phys_jac(f, J, N_EXEC)                            # [N_Z, 30, A] cm per unit noise
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    v1 = Vt[:, 0]                                         # [N_Z, A]
    R = np.random.default_rng(40_000 + k).standard_normal((N_Z, 2, A))
    R /= np.linalg.norm(R, axis=-1, keepdims=True)
    dirs = {"+v1": v1, "-v1": -v1, "+r1": R[:, 0], "-r1": -R[:, 0], "+r2": R[:, 1], "-r2": -R[:, 1]}
    jobs = [(i, "base", 0.0) for i in range(N_Z)] + [(i, d, s) for i in range(N_Z) for d in RAYS for s in STEPS]
    lab = {}
    log(f"'{text}' init {init} t {t_state}: {len(jobs)} episodes; sigma1 (cm per unit noise) "
        f"{np.round(S[:, 0], 3).tolist()}, FTLE {np.round(np.log(S[:, 0]), 2).tolist()}")
    for b in range(0, len(jobs), W):
        jb = jobs[b:b + W]
        n = len(jb)
        z = np.stack([Z0[i] + (s * dirs[d][i] if d != "base" else 0) for i, d, s in jb])
        a = f.unnorm(f.flow_np(z, c))
        o = v.restore(idx[:n], "s")
        res = v.episodes(f, idx[:n], o, [text] * n, [np.random.default_rng(30_000 + fut) for _ in jb],
                         first=list(a), stop="goal")
        for (i, d, s), r in zip(jb, res):
            lab[(i, d, s)] = label(r)
        if (b // W) % 6 == 5 or b + W >= len(jobs):
            log(f"{b + n}/{len(jobs)} episodes")
    first = {}
    for i in range(N_Z):
        for d in RAYS:
            fl = [s for s in STEPS if lab[(i, d, s)] != lab[(i, "base", 0.0)]]
            first[f"{i}{d}"] = fl[0] if fl else None
    path = os.path.join(OUT, f"boundary_{text.replace(' ', '_')}_{init}_t{t_state}.json")
    json.dump(dict(text=text, init=init, t=t_state, seed=seed, fut=fut, steps=STEPS, rays=RAYS,
                   sigma1=S[:, 0].tolist(), sigma2=S[:, 1].tolist(),
                   labels={f"{i}|{d}|{s}": l for (i, d, s), l in lab.items()}, first_flip=first), open(path, "w"))
    v.close()
    report(path)


def report(path=None):
    for p in [path] if path else sorted(glob.glob(os.path.join(OUT, "boundary_*.json"))):
        R = json.load(open(p))
        L = R["labels"]
        print(f"\n'{R['text']}' init {R['init']} t {R['t']} (future stream {R['fut']})")
        names = sorted(set(L.values()))
        code = {l: "abcdefghij"[i] for i, l in enumerate(names)}
        print("  codes: " + ", ".join(f"{c} = {l}" for l, c in code.items()) + f"; steps {R['steps']}")
        for i in range(len(R["sigma1"])):
            base = L[f"{i}|base|0.0"]
            rays = "  ".join(f"{d} " + "".join(code[L[f'{i}|{d}|{s}']] for s in R["steps"]) for d in R["rays"])
            print(f"  seed {i} ({code[base]}, sigma1 {R['sigma1'][i]:.3f}): {rays}")
        F = R["first_flip"]
        for kind in ("v1", "r"):
            vals = [F[k] for k in F if k[2:].startswith(kind)]
            hit = [x for x in vals if x is not None]
            print(f"  rays along {'top singular vector' if kind == 'v1' else 'random directions':22s}: "
                  f"label flips within s <= {R['steps'][-1]} on {len(hit)}/{len(vals)}; first flip s "
                  f"median {np.median(hit) if hit else float('nan'):.1f}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        a = sys.argv[2:]
        run(a[0], *[int(x) for x in a[1:]])
    elif sys.argv[1] == "report":
        report()
