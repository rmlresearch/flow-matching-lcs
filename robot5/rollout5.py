"""
Closed-loop pi0.5 rollouts on LIBERO-spatial through the explicit flow map (pi05flow), with the
noise of every replan drawn from a seeded generator so that runs are reproducible.

    python rollout5.py sanity        # each task's own instruction: success rate (checkpoint works?)
    python rollout5.py ambig         # "pick up the black bowl and place it on the plate":
                                     #   which bowl gets lifted, per scene, over seeds
    python rollout5.py report        # tables from the cached results

Mode = the bowl that gets lifted (0 = the one the task's own instruction names, 1 = the other).
Results: out/rollouts_<name>.json (one record per episode).
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from env5 import Scene, AMBIG  # noqa: E402

OUT = os.path.join(HERE, "out")
N_EXEC = 10                       # actions executed per chunk before replanning


def rollout(f, scene, text, seed, n_exec=N_EXEC, max_steps=None, record=False, frames=None):
    """One episode. Returns a dict: success, lifted bowl, steps, step of the first lift.
    frames: optional list; gets the camera images the policy sees ({name: HxWx3 uint8}) at every step."""
    from pi05flow import H, D_ACT
    rng = np.random.default_rng(seed)
    obs = scene.reset()
    if frames is not None:
        frames.append({k: v.copy() for k, v in obs["pixels"].items()})
    max_steps = max_steps or scene.max_steps
    t, lift_t, first, eef, chunks = 0, -1, -1, [scene.eef()], []
    success = False
    while t < max_steps:
        c = f.prefix([scene.obs_to_lerobot(obs, text)], [text])
        z = rng.standard_normal((1, H * D_ACT))
        a = f.unnorm(f.flow_np(z, c)[0])                   # [H, 7] env actions
        if record:
            chunks.append(dict(t=t, z=z[0].astype(np.float32), a=a.astype(np.float32)))
        for k in range(n_exec):
            obs, success, term = scene.step(a[k])
            t += 1
            if frames is not None:
                frames.append({k_: v.copy() for k_, v in obs["pixels"].items()})
            eef.append(scene.eef())
            if first < 0:                               # the bowl lifted first is the mode
                first = scene.lifted()
                lift_t = t if first >= 0 else -1
            if success or term or t >= max_steps:
                break
        if success or term:
            break
    out = dict(task=scene.task_id, init=scene.init_id, layout=scene.layout, text=text, seed=int(seed), success=bool(success),
               lifted=first, lift_t=lift_t, steps=t,
               eef=np.round(np.array(eef), 4).tolist())
    if record:
        out["chunks"] = chunks
    return out


def run(name, jobs, f=None):
    """jobs: list of (task, init, layout, text, seed). Appends to out/rollouts_<name>.json, skipping done."""
    import pi05flow
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"rollouts_{name}.json")
    done = json.load(open(path)) if os.path.exists(path) else []
    key = lambda r: (r["task"], r["init"], r["layout"], r["text"], r["seed"])  # noqa: E731
    have = {key(r) for r in done}
    jobs = [j for j in jobs if j not in have]
    f = f or pi05flow.load()
    scenes = {}
    t0 = time.time()
    for n, (task, init, layout, text, seed) in enumerate(jobs):
        if (task, init, layout) not in scenes:
            for s in scenes.values():
                s.close()
            scenes = {(task, init, layout): Scene(task, init, layout)}
        r = rollout(f, scenes[(task, init, layout)], text, seed)
        done.append(r)
        print(f"[{n + 1}/{len(jobs)} {time.time() - t0:.0f}s] task {task} init {init} layout {layout} seed {seed} "
              f"'{text[:40]}' -> success {r['success']} lifted {r['lifted']} steps {r['steps']}", flush=True)
        if (n + 1) % 10 == 0 or n + 1 == len(jobs):
            json.dump(done, open(path, "w"))
    return done


def report():
    from collections import defaultdict
    for name in ("sanity", "ambig", "layouts"):
        path = os.path.join(OUT, f"rollouts_{name}.json")
        if not os.path.exists(path):
            continue
        R = json.load(open(path))
        print(f"== {name}: {len(R)} episodes")
        by = defaultdict(list)
        for r in R:
            by[r["task"]].append(r)
        for task in sorted(by):
            rs = by[task]
            lift = np.array([r["lifted"] for r in rs])
            print(f"  task {task}: success {np.mean([r['success'] for r in rs]):.2f}  "
                  f"lifted bowl1 {np.mean(lift == 0):.2f} bowl2 {np.mean(lift == 1):.2f} none {np.mean(lift < 0):.2f}"
                  f"  (n={len(rs)})")
        if name in ("ambig", "layouts"):
            # per scene (task, init): is the choice of bowl seed-dependent?
            sc = defaultdict(list)
            for r in R:
                sc[(r["task"], r["init"], r["layout"])].append(r["lifted"])
            mixed = [k for k, v in sc.items() if len(set(x for x in v if x >= 0)) > 1]
            print(f"  scenes where the lifted bowl depends on the seed: {len(mixed)}/{len(sc)}")


if __name__ == "__main__":
    step = sys.argv[1]
    from env5 import suite
    tasks = suite().tasks
    if step == "sanity":
        n_init = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        run("sanity", [(k, i, 0, tasks[k].language, 0) for k in range(10) for i in range(n_init)])
    elif step == "ambig":
        n_init = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        n_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        run("ambig", [(k, i, 0, AMBIG, s) for k in range(10) for i in range(n_init) for s in range(n_seed)])
    elif step == "layouts":
        # noise vs fixture layout: scene (task, init), layouts 0..n_lay-1, seeds 0..n_seed-1
        task, init, n_lay, n_seed = [int(x) for x in sys.argv[2:6]]
        run("layouts", [(task, init, l, AMBIG, s) for l in range(n_lay) for s in range(n_seed)])
    elif step == "report":
        report()
