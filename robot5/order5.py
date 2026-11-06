"""
Subtask-order modes on LIBERO-10: which of the two objects does pi0.5 pick first, and does the
noise decide it?

    python order5.py screen [n_init] [n_seed]   # tasks in env5.TRACK10, layout 0: first object
                                                #   lifted per seed (episodes stop at the first lift)
    python order5.py report

Results: out/order_screen.json.
"""
import json, os, sys, time
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from env5 import TRACK10  # noqa: E402

OUT = os.path.join(HERE, "out")
SUITE = "libero_10"
W = int(os.environ.get("N_WORKERS", 8))   # ~3 GB RAM each (see vec5)
t00 = time.time()


def log(*a):
    print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)


def screen(n_init=4, n_seed=8, tasks=tuple(TRACK10)):
    import pi05flow
    from vec5 import Vec
    path = os.path.join(OUT, "order_screen.json")
    done = json.load(open(path)) if os.path.exists(path) else []
    have = {(r["task"], r["init"], r["layout"], r["seed"]) for r in done}
    jobs = [(t, i, 0, s) for t in tasks for i in range(n_init) for s in range(n_seed)
            if (t, i, 0, s) not in have]
    f = pi05flow.load()
    v = Vec(W)
    loaded = [None] * W
    log(f"{len(jobs)} episodes on {W} workers")
    for b in range(0, len(jobs), W):
        batch = jobs[b:b + W]
        idx = list(range(len(batch)))
        need = [k for k in idx if loaded[k] != batch[k][:3]]
        if need:
            v.load(need, [dict(task_id=batch[k][0], init_id=batch[k][1], layout=batch[k][2], suite_name=SUITE)
                          for k in need])
            for k in need:
                loaded[k] = batch[k][:3]
        obs = v.reset(idx)
        texts = [v.info[k]["task"] for k in idx]
        rngs = [np.random.default_rng(batch[k][3]) for k in idx]
        res = v.episodes(f, idx, obs, texts, rngs, stop="lift")
        for (task, init, layout, seed), r, text in zip(batch, res, texts):
            done.append(dict(task=task, init=init, layout=layout, seed=seed, text=text, first=r["first"],
                             lift_t=r["lift_t"], t=r["t"], track=list(TRACK10[task])))
        json.dump(done, open(path, "w"))
        log(f"batch {b // W + 1}/{-(-len(jobs) // W)}: " + " ".join(
            f"{t}.{i}.{s}:{r['first'] + 1 if r['first'] >= 0 else '-'}" for (t, i, _, s), r in zip(batch, res)))
    v.close()


OSC, GAIN = 0.05, 0.25           # OSC goal offset per unit action (m); measured tracking gain (lcs5 calib)


def phys_jac(f, J, n_exec=10):
    """dPhi/dz rows [N, 7*steps, A] -> predicted eef path over the executed steps, in cm:
    rows [N, n_exec*3, A] = 100 * GAIN * OSC * cumsum_t (std_d * dPhi_{t,d}/dz), d = x, y, z."""
    N, R, A = J.shape
    Jt = J.reshape(N, R // 7, 7, A)[:, :n_exec, :3] * f.a_scale[:3, None]
    return 100 * GAIN * OSC * np.cumsum(Jt, axis=1).reshape(N, -1, A)


def decide(task, init, seed=0, fut=0, layout=0, n_ftle=8):
    """Along a main closed-loop rollout (noise stream `seed`), at every replan state until the
    main rollout lifts an object: W branches, each executing a different noise seed for the
    current chunk, then the shared future stream `fut` until the first lift -> P(first object).
    Also the physical FTLE (executed eef path, cm) of n_ftle of those seeds."""
    import pi05flow
    from pi05flow import H, D_ACT
    from vec5 import Vec
    A = H * D_ACT
    f = pi05flow.load()
    v = Vec(W)
    idx = list(range(W))
    v.load(idx, [dict(task_id=task, init_id=init, layout=layout, suite_name=SUITE)] * W)
    text = v.info[0]["task"]; max_t = v.info[0]["max_steps"]
    obs = v.reset(idx)
    rng = np.random.default_rng(seed)
    rows, k = [], 0
    while True:
        v.save(idx, "k")
        c = f.prefix([obs[0]], [text])
        zb = np.random.default_rng(20_000 + k).standard_normal((W, A))
        ab = f.unnorm(f.flow_np(zb, c))
        res = v.episodes(f, idx, obs, [text] * W, [np.random.default_rng(30_000 + fut) for _ in idx],
                         first=list(ab), stop="lift")
        first = np.array([r["first"] for r in res])
        _, J = f.jac(zb[:n_ftle], c, steps=N_EXEC_ROWS, bs=2, chunk=70)
        sv = np.linalg.svd(phys_jac(f, J), compute_uv=False)
        rows.append(dict(k=k, t=10 * k, first=first.tolist(), lift_t=[r["lift_t"] for r in res],
                         ftle_phys=np.log(sv[:, 0]).tolist(), sv2_sv1=(sv[:, 1] / sv[:, 0]).tolist()))
        p = [np.mean(first == j) for j in (0, 1)]
        log(f"task {task} init {init} main seed {seed}, replan {k} (t={10 * k}): first object 1/2/none "
            f"{p[0]:.2f}/{p[1]:.2f}/{np.mean(first < 0):.2f} | FTLE_phys median {np.median(np.log(sv[:, 0])):+.2f} "
            f"(sigma1 {np.median(sv[:, 0]):.2f} cm per unit noise)")
        # advance the main rollout by one chunk (all workers in lockstep)
        obs = v.restore(idx, "k")
        a = f.unnorm(f.flow_np(rng.standard_normal((1, A)), c)[0])
        outs = v._call(idx, "steps", [(a[:10], "none", max_t)] * W)
        obs = [o.pop("obs") for o in outs]
        k += 1
        if outs[0]["first"] >= 0 or outs[0]["done"]:
            log(f"main rollout lifted object {outs[0]['first'] + 1} at t={outs[0]['lift_t']}")
            break
    path = os.path.join(OUT, f"order_decide_{task}_{init}_{seed}.json")
    json.dump(dict(task=task, init=init, seed=seed, fut=fut, main_first=outs[0]["first"], states=rows), open(path, "w"))
    v.close()


N_EXEC_ROWS = 10


def report():
    R = json.load(open(os.path.join(OUT, "order_screen.json")))
    sc = defaultdict(list)
    for r in sorted(R, key=lambda r: r["seed"]):
        sc[(r["task"], r["init"])].append(r)
    for (task, init), rs in sorted(sc.items()):
        s = "".join(str(r["first"] + 1) if r["first"] >= 0 else "." for r in rs)
        lt = [r["lift_t"] for r in rs if r["lift_t"] > 0]
        print(f"task {task} init {init}: first object per seed {s}   (1 = {rs[0]['track'][0]}, 2 = {rs[0]['track'][1]}; "
              f"lift step {min(lt) if lt else '-'}-{max(lt) if lt else '-'})")
    mixed = [k for k, rs in sc.items() if len({r["first"] for r in rs if r["first"] >= 0}) > 1]
    print(f"scenes where the noise changes the order: {len(mixed)}/{len(sc)} {sorted(mixed)}")


if __name__ == "__main__":
    if sys.argv[1] == "screen":
        screen(*[int(x) for x in sys.argv[2:4]])
    elif sys.argv[1] == "decide":
        decide(*[int(x) for x in sys.argv[2:]])
    elif sys.argv[1] == "report":
        report()
