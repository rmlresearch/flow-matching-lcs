"""
Does the one-replan closed-loop gain (latent5, fast mode) recognise ambiguous prompts?

For every goal5 prompt x initial state (34 x 4), the gain at the first N_CHUNKS replans of the
main rollout (noise stream 0, shared across prompts), delta = 1 cm of predicted path. Ground
truth: goal5's closed-loop run (out/goal_run.json) -- a (prompt, init) pair is "split" if its 8
seeds reached more than one label. Scored by AUROC of the gain over the window, overall and
within initial state, with 95% bootstrap intervals clustered by initial state.

    python screen5.py run [n_chunks]     # -> out/screen_gain.json (resumable per prompt x init)
    python screen5.py report
"""
import json, os, sys, time
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "out")
SUITE, TASK0, LAYOUT, SEED, DELTA = "libero_goal", 0, 0, 0, 1.0
N_INIT, N_CHUNKS = 4, 5
t00 = time.time()


def log(*a):
    print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)


def run(n_chunks=N_CHUNKS):
    import pi05flow
    from env5 import Scene
    from goal5 import PROMPTS
    from latent5 import gain_rollout
    path = os.path.join(OUT, "screen_gain.json")
    R = json.load(open(path)) if os.path.exists(path) else {}
    f = pi05flow.load()
    for init in range(N_INIT):
        todo = [p for p in PROMPTS if f"{p[0]}|{init}" not in R]
        if not todo:
            continue
        s = Scene(TASK0, init, LAYOUT, suite_name=SUITE)
        for text, kind, _ in todo:
            rows, main, t = gain_rollout(f, s, text, SEED, n_states=n_chunks + 1, delta=DELTA, attrib=False, quiet=True)
            R[f"{text}|{init}"] = dict(text=text, kind=kind, init=init, main=main, t_end=t,
                                       states=[{k: v for k, v in r.items() if k != "layout"} for r in rows])
            json.dump(R, open(path, "w"))
            g = [r["gain"] for r in rows if "gain" in r]
            log(f"init {init} '{text}': gains " + " ".join(f"{x:.1f}" for x in g) + f"  ({kind})")
        s.close()


def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def boot(score, y, groups, n=2000, seed=0):
    """95% bootstrap interval of the AUROC, resampling groups (initial states) with replacement."""
    rng = np.random.default_rng(seed)
    score, y, groups = map(np.asarray, (score, y, groups))
    G = np.unique(groups)
    vals = []
    for _ in range(n):
        pick = rng.choice(G, len(G))
        idx = np.concatenate([np.where(groups == g)[0] for g in pick])
        vals.append(auroc(score[idx], y[idx]))
    vals = np.array(vals)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def report():
    R = json.load(open(os.path.join(OUT, "screen_gain.json")))
    G = json.load(open(os.path.join(OUT, "goal_run.json")))
    labels = defaultdict(set)
    for r in G:
        labels[(r["text"], r["init"])].add(r["label"])
    rows = []
    for key, v in R.items():
        split = labels.get((v["text"], v["init"]))
        if split is None:
            continue
        g = [st["gain"] for st in v["states"] if "gain" in st]
        rows.append(dict(text=v["text"], kind=v["kind"], init=v["init"], split=len(split) > 1, n_labels=len(split),
                         gains=g, gmax=max(g), gmean=float(np.mean(g)), g1=g[0], g2=max(g[:2]), g3=max(g[:3])))
    y = [r["split"] for r in rows]; grp = [r["init"] for r in rows]
    print(f"{len(rows)} (prompt, init) pairs, {sum(y)} split in goal5\n")
    for feat in ("gmax", "gmean", "g1", "g2", "g3"):
        sc = [r[feat] for r in rows]
        lo, hi = boot(sc, y, grp)
        within = [auroc([r[feat] for r in rows if r["init"] == i], [r["split"] for r in rows if r["init"] == i])
                  for i in range(N_INIT)]
        print(f"  {feat:5s}: AUROC {auroc(sc, y):.2f} [{lo:.2f}, {hi:.2f}]   within init: " +
              " ".join(f"{a:.2f}" for a in within))
    print("\n  gain per replan (t 10 .. ) | goal5 labels | prompt")
    for r in sorted(rows, key=lambda r: (r["init"], -r["gmax"])):
        print(f"  init {r['init']} {'SPLIT' if r['split'] else '     '} " + " ".join(f"{x:5.1f}" for x in r["gains"]) +
              f"  | {r['n_labels']} | {r['text']} ({r['kind']})")
    # post hoc (defined after seeing the raw result): the gain relative to the trained prompts that name
    # one of the prompt's admissible goals, at the same replan, max over the window
    from goal5 import PROMPTS, ORIG
    from env5 import GOAL_NAMES
    goal_of = {p: GOAL_NAMES[i] for i, p in enumerate(ORIG)}
    by = {(r["text"], r["init"]): r for r in rows}
    rel = []
    for text, kind, ok in PROMPTS:
        matched = [p for p in ORIG if ok and goal_of[p] in ok] if kind != "trained" else []
        for i in range(N_INIT):
            if matched and (text, i) in by:
                m = np.mean([by[(p, i)]["gains"] for p in matched], axis=0)
                rel.append(dict(score=float(np.max(np.log(by[(text, i)]["gains"]) - np.log(m))), split=by[(text, i)]["split"],
                                init=i))
    sc, y, grp = [r["score"] for r in rel], [r["split"] for r in rel], [r["init"] for r in rel]
    lo, hi = boot(sc, y, grp)
    within = [auroc([r["score"] for r in rel if r["init"] == i], [r["split"] for r in rel if r["init"] == i])
              for i in range(N_INIT)]
    print(f"\n  post hoc, {len(rel)} non-trained pairs with matched trained prompts ({sum(y)} split): max_t log(gain / "
          f"matched gain) AUROC {auroc(sc, y):.2f} [{lo:.2f}, {hi:.2f}]   within init: " + " ".join(f"{a:.2f}" for a in within))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(*[int(x) for x in sys.argv[2:]])
    elif sys.argv[1] == "report":
        report()
