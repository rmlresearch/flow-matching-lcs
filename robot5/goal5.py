"""
Language and noise on LIBERO-goal: one fixed kitchen scene, 10 trained instructions (all in
pi0.5's training set, HuggingFaceVLA/libero), plus paraphrases, underspecified prompts and novel
compositions an operator might type. Which goal does each prompt produce, and does the noise
seed ever decide it?

    python goal5.py screen [n_init] [n_seed]   # open loop: first plan under n_seed noise seeds per
                                               #   (prompt, initial state): spread of the predicted
                                               #   eef path (cm), bimodality -> out/goal_screen.npz
    python goal5.py run [n_init] [n_seed]      # closed loop, seeds shared across prompts (common
                                               #   random numbers): first goal / placement event
                                               #   reached -> out/goal_run.json
    python goal5.py report

Scene: LIBERO-goal task 0's environment (the BDDL scene is identical for all 10 tasks; only the
goal and wording differ), its initial states 0..n_init-1, fixture layout 0. The prompt is the
only thing that changes. Labels: env5.GOALS (the 10 task goals) + env5.EVENTS (other drawers
opened, placements no task asks for); an episode's label is the first goal or placement event
reached (drawer opening alone is not terminal), else the last drawer opened, else 'none'.
"""
import json, os, sys, time
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from env5 import GOAL_NAMES, EVENT_NAMES  # noqa: E402

OUT = os.path.join(HERE, "out")
SUITE, TASK0, LAYOUT = "libero_goal", 0, 0
W = int(os.environ.get("N_WORKERS", 8))
NAMES = GOAL_NAMES + EVENT_NAMES
t00 = time.time()

# (prompt, kind, labels that count as doing what was asked)
ORIG = ["open the middle drawer of the cabinet", "put the bowl on the stove", "put the wine bottle on top of the cabinet",
        "open the top drawer and put the bowl inside", "put the bowl on top of the cabinet",
        "push the plate to the front of the stove", "put the cream cheese in the bowl", "turn on the stove",
        "put the bowl on the plate", "put the wine bottle on the rack"]
PROMPTS = [(p, "trained", [GOAL_NAMES[i]]) for i, p in enumerate(ORIG)] + [
    ("open the second drawer of the cabinet", "paraphrase", ["middle drawer"]),
    ("pull out the middle drawer", "paraphrase", ["middle drawer"]),
    ("place the bowl on the stove", "paraphrase", ["bowl->stove"]),
    ("put the bowl on the burner", "paraphrase", ["bowl->stove"]),
    ("put the bottle on top of the cabinet", "paraphrase", ["bottle->cabinet top"]),
    ("push the plate towards the stove", "paraphrase", ["plate->stove front"]),
    ("put the cheese in the bowl", "paraphrase", ["cheese->bowl"]),
    ("switch on the stove", "paraphrase", ["stove on"]),
    ("place the bowl onto the plate", "paraphrase", ["bowl->plate"]),
    ("put the wine bottle in the rack", "paraphrase", ["bottle->rack"]),
    ("open a drawer", "underspecified", ["middle drawer", "top drawer open", "bottom drawer open"]),
    ("open the drawer", "underspecified", ["middle drawer", "top drawer open", "bottom drawer open"]),
    ("put the bowl somewhere", "underspecified", ["bowl->stove", "bowl->top drawer", "bowl->cabinet top", "bowl->plate",
                                                  "bowl->middle drawer"]),
    ("move the bowl", "underspecified", ["bowl->stove", "bowl->top drawer", "bowl->cabinet top", "bowl->plate",
                                         "bowl->middle drawer"]),
    ("put the wine bottle somewhere", "underspecified", ["bottle->cabinet top", "bottle->rack", "bottle->stove",
                                                         "bottle->plate"]),
    ("move the wine bottle", "underspecified", ["bottle->cabinet top", "bottle->rack", "bottle->stove", "bottle->plate"]),
    ("put something on the stove", "underspecified", ["bowl->stove", "bottle->stove", "cheese->stove"]),
    ("put something on the plate", "underspecified", ["bowl->plate", "cheese->plate", "bottle->plate"]),
    ("put something on top of the cabinet", "underspecified", ["bottle->cabinet top", "bowl->cabinet top"]),
    ("put the bowl on the cabinet", "underspecified", ["bowl->cabinet top", "bowl->top drawer", "bowl->middle drawer"]),
    ("tidy up the table", "underspecified", None),
    ("put the wine bottle on the stove", "novel", ["bottle->stove"]),
    ("put the cream cheese on the plate", "novel", ["cheese->plate"]),
    ("put the cream cheese on the stove", "novel", ["cheese->stove"]),
]


def log(*a):
    print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)


def path_cm(a):
    """Env-action chunks [..., H, 7] -> predicted eef path [..., H, 3] in cm (order5 calibration)."""
    from order5 import OSC, GAIN
    return 100 * GAIN * OSC * np.cumsum(a[..., :3], axis=-2)


def bimodality(x):
    """Sarle's coefficient of the samples' projection on their first principal component
    (> 5/9 suggests two modes; uniform 5/9, Gaussian 1/3)."""
    x = x.reshape(len(x), -1); x = x - x.mean(0)
    p = x @ np.linalg.svd(x, full_matrices=False)[2][0]
    n = len(p); z = (p - p.mean()) / p.std()
    g, k = (z ** 3).mean(), (z ** 4).mean() - 3
    return (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def screen(n_init=4, n_seed=128):
    import pi05flow
    from env5 import Scene
    f = pi05flow.load()
    Z = np.random.default_rng(11).standard_normal((n_seed, pi05flow.H * pi05flow.D_ACT))
    rows = []
    for init in range(n_init):
        s = Scene(TASK0, init, LAYOUT, suite_name=SUITE)
        obs = s.reset()
        o = s.obs_to_lerobot(obs, "")
        for text, kind, _ in PROMPTS:
            c = f.prefix([o], [text])
            a = f.unnorm(f.flow_np(Z, c, bs=64))                         # [n_seed, H, 7]
            p = path_cm(a)
            spread = float(np.sqrt((p.std(0) ** 2).sum(-1)).max())
            rows.append(dict(init=init, text=text, kind=kind, spread=spread, bc=float(bimodality(p)),
                             end=p[:, -1].mean(0), grip=a[:, :, 6].mean(), chunks=a.astype(np.float32)))
        log(f"init {init}: " + " | ".join(f"{r['text'][:22]}: {r['spread']:.1f}cm bc {r['bc']:.2f}"
                                          for r in rows[-len(PROMPTS):][:6]) + " ...")
        s.close()
    np.savez_compressed(os.path.join(OUT, "goal_screen.npz"), **{k: np.array([r[k] for r in rows]) for k in rows[0]})


def label(r):
    if r["goal"] >= 0:
        return NAMES[r["goal"]]
    opened = [(t, NAMES[i]) for i, t in enumerate(r["ever_t"]) if t >= 0]
    return max(opened)[1] if opened else "none"


def run(n_init=4, n_seed=8):
    import pi05flow
    from vec5 import Vec
    path = os.path.join(OUT, "goal_run.json")
    done = json.load(open(path)) if os.path.exists(path) else []
    have = {(r["text"], r["init"], r["seed"]) for r in done}
    assert W % n_init == 0, "each worker holds one initial state"
    per = W // n_init                                       # seeds per prompt per batch
    jobs = [(text, s0) for text, _, _ in PROMPTS for s0 in range(0, n_seed, per)
            if any((text, i, s) not in have for i in range(n_init) for s in range(s0, s0 + per))]
    f = pi05flow.load()
    v = Vec(W)
    idx = list(range(W))
    inits = [k % n_init for k in idx]
    v.load(idx, [dict(task_id=TASK0, init_id=i, layout=LAYOUT, suite_name=SUITE) for i in inits])
    v.reset(idx)
    v.save(idx, "init")
    log(f"{len(jobs)} batches of {W} episodes")
    for b, (text, s0) in enumerate(jobs):
        obs = v.restore(idx, "init")
        seeds = [s0 + k // n_init for k in idx]
        res = v.episodes(f, idx, obs, [text] * W, [np.random.default_rng(s) for s in seeds], stop="goal")
        for k, r in enumerate(res):
            done.append(dict(text=text, init=inits[k], seed=seeds[k], label=label(r), goal=r["goal"], goal_t=r["goal_t"],
                             t=r["t"], ever_t=r["ever_t"], final=r["final"]))
        json.dump(done, open(path, "w"))
        log(f"batch {b + 1}/{len(jobs)} '{text}' seeds {s0}-{s0 + per - 1}: " +
            ", ".join(f"{label(r)}@{r['t']}" for r in res))
    v.close()


def report():
    S = np.load(os.path.join(OUT, "goal_screen.npz"), allow_pickle=True) if os.path.exists(
        os.path.join(OUT, "goal_screen.npz")) else None
    R = json.load(open(os.path.join(OUT, "goal_run.json"))) if os.path.exists(os.path.join(OUT, "goal_run.json")) else []
    by = defaultdict(list)
    for r in sorted(R, key=lambda r: (r["init"], r["seed"])):
        by[r["text"]].append(r)
    codes = {}
    print(f"{'prompt':42s} {'kind':14s} {'asked-for':>9s} {'noise-split':>11s} {'spread cm':>9s} {'BC max':>6s}  labels")
    for text, kind, ok in PROMPTS:
        rs = by.get(text, [])
        sp = bc = ""
        if S is not None:
            m = S["text"] == text
            sp, bc = f"{np.median(S['spread'][m]):.1f}", f"{S['bc'][m].max():.2f}"
        if not rs:
            print(f"{text:42s} {kind:14s} {'-':>9s} {'-':>11s} {sp:>9s} {bc:>6s}")
            continue
        labs = [r["label"] for r in rs]
        good = np.mean([l in ok for l in labs]) if ok else float("nan")
        per_init = defaultdict(list)
        for r in rs:
            per_init[r["init"]].append(r["label"])
        split = sum(len(set(v)) > 1 for v in per_init.values())
        for l in labs:
            codes.setdefault(l, "abcdefghijklmnopqrstuvwxyz"[len(codes)])
        cnt = Counter(labs).most_common()
        print(f"{text:42s} {kind:14s} {good:9.2f} {split:>6d}/{len(per_init):<4d} {sp:>9s} {bc:>6s}  " +
              ", ".join(f"{l} {n}" for l, n in cnt))
        print(f"{'':42s} {'':14s} per initial state (seeds): " +
              " | ".join("".join(codes[l] for l in v) for _, v in sorted(per_init.items())))
    print("codes: " + ", ".join(f"{c} = {l}" for l, c in codes.items()))


if __name__ == "__main__":
    cmd = sys.argv[1]
    args = [int(x) for x in sys.argv[2:4]]
    {"screen": screen, "run": run, "report": report}[cmd](*args)
