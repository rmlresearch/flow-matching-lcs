"""
Where along a rollout does the noise decide an underspecified LIBERO-goal prompt (goal5), and
does the chunk flow map see it?

Along a main closed-loop rollout (noise stream = `seed`, reproducing goal5's episode), at every
replan state s_k (every 10 steps) until the main rollout reaches a goal:

  - closed-loop basin of the current chunk's noise: N_Z seeds z of this chunk; restore s_k exactly,
    execute z's chunk for 10 steps, then continue closed loop with a *shared* future-noise stream
    (N_FUT streams) until a goal / placement event (goal5 labels). Within one stream, label
    differences are caused by z alone (common random numbers).
  - FTLE of the chunk flow map at s_k for the same seeds, on the predicted eef path (cm,
    order5.phys_jac): executed 10 steps and the full 50-step chunk.

    python branch5.py run "<prompt>" init seed     # -> out/branch_<slug>_<init>_<seed>.json
    python branch5.py report
"""
import glob, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "out")
SUITE, TASK0, LAYOUT = "libero_goal", 0, 0
W = int(os.environ.get("N_WORKERS", 8))
N_Z, N_FUT, N_EXEC = 8, 2, 10
t00 = time.time()


def log(*a):
    print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)


def slug(text):
    return text.replace(" ", "_")


def phys_ftle(f, J, rows):
    """FTLE of the predicted eef path (cm) over the first `rows` chunk steps, per seed."""
    from order5 import OSC, GAIN
    N, R, A = J.shape
    Jt = J.reshape(N, R // 7, 7, A)[:, :rows, :3] * f.a_scale[:3, None]
    P = 100 * GAIN * OSC * np.cumsum(Jt, axis=1).reshape(N, -1, A)
    sv = np.linalg.svd(P, compute_uv=False)
    return np.log(sv[:, 0]), sv[:, 1] / sv[:, 0]


def run(text, init, seed):
    import pi05flow
    from pi05flow import H, D_ACT
    from vec5 import Vec
    from goal5 import label
    A = H * D_ACT
    f = pi05flow.load()
    v = Vec(W)
    idx = list(range(W))
    v.load(idx, [dict(task_id=TASK0, init_id=init, layout=LAYOUT, suite_name=SUITE)] * W)
    max_t = v.info[0]["max_steps"]
    obs = v.reset(idx)
    rng = np.random.default_rng(seed)                    # main stream, as goal5.run / Vec.episodes
    pairs = [(i, j) for i in range(N_Z) for j in range(N_FUT)]
    rows, k = [], 0
    while True:
        v.save(idx, "k")
        c = f.prefix([obs[0]], [text])
        zb = np.random.default_rng(20_000 + k).standard_normal((N_Z, A))
        ab, J = f.jac(zb, c, steps=H, bs=2, chunk=70)
        ab = f.unnorm(ab)
        fe, r_exec = phys_ftle(f, J, N_EXEC)
        ff, _ = phys_ftle(f, J, H)
        lab = [[None] * N_FUT for _ in range(N_Z)]
        for b in range(0, len(pairs), W):
            pb = pairs[b:b + W]
            o = v.restore(idx[:len(pb)], "k")
            res = v.episodes(f, idx[:len(pb)], o, [text] * len(pb),
                             [np.random.default_rng(30_000 + j) for _, j in pb], first=[ab[i] for i, _ in pb],
                             stop="goal")
            for (i, j), r in zip(pb, res):
                lab[i][j] = label(r)
        # advance the main rollout by one chunk (all workers in lockstep)
        obs = v.restore(idx, "k")
        a = f.unnorm(f.flow_np(rng.standard_normal((1, A)), c)[0])
        outs = v._call(idx, "steps", [(a[:N_EXEC], "goal", max_t)] * W)
        obs = [o.pop("obs") for o in outs]
        dep = [len({lab[i][j] for i in range(N_Z)}) > 1 for j in range(N_FUT)]
        rows.append(dict(k=k, t=N_EXEC * k, eef=outs[0]["eef"].tolist(), labels=lab, ftle_exec=fe.tolist(),
                         ftle_full=ff.tolist(), sv2_sv1_exec=r_exec.tolist(), z_decides=dep))
        cnt = {}
        for l in sum(lab, []):
            cnt[l] = cnt.get(l, 0) + 1
        log(f"'{text}' init {init} seed {seed}, t {N_EXEC * k:3d}: {cnt} | z decides in {sum(dep)}/{N_FUT} streams | "
            f"FTLE exec {np.median(fe):+.2f} (max {fe.max():+.2f}), full {np.median(ff):+.2f}")
        k += 1
        if outs[0]["done"]:
            main = label(outs[0])
            log(f"main rollout: {main} at t {outs[0]['t']}")
            break
    path = os.path.join(OUT, f"branch_{slug(text)}_{init}_{seed}.json")
    json.dump(dict(text=text, init=init, seed=seed, main=main, n_z=N_Z, n_fut=N_FUT, states=rows), open(path, "w"))
    v.close()


def report():
    for p in sorted(glob.glob(os.path.join(OUT, "branch_*.json"))):
        R = json.load(open(p))
        print(f"\n'{R['text']}' init {R['init']} main seed {R['seed']} -> {R['main']}")
        names = sorted({l for s in R["states"] for l in sum(s["labels"], [])})
        code = {l: "abcdefghij"[i] for i, l in enumerate(names)}
        print("  codes: " + ", ".join(f"{c} = {l}" for l, c in code.items()))
        for s in R["states"]:
            streams = " | ".join("".join(code[s["labels"][i][j]] for i in range(R["n_z"])) for j in range(R["n_fut"]))
            print(f"  t {s['t']:3d}  z-labels per stream {streams}   FTLE exec {np.median(s['ftle_exec']):+.2f} "
                  f"[{min(s['ftle_exec']):+.2f}, {max(s['ftle_exec']):+.2f}]  full {np.median(s['ftle_full']):+.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif sys.argv[1] == "report":
        report()
