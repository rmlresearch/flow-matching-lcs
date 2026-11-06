"""
Group candidate prompts by what the policy would do with them (no modes known a priori).

For one scene state, the mean predicted eef path of each prompt's first chunk (goal5 screen,
128 seeds) is a point in path space; prompts are linked by the maximum over chunk steps of the
distance between their mean paths and cut at `thr` cm (the seed spread of a single prompt is
0.5-1.5 cm). A group = prompts the policy treats the same for this chunk. Behaviours that only
diverge after the first chunk share a group here and separate when re-grouped at a later state.

    python cluster5.py [init] [steps] [thr_cm]      # default: all inits, 10 steps / 1.5 cm and 50 steps / 4 cm
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from goal5 import path_cm, PROMPTS  # noqa: E402

OUT = os.path.join(HERE, "out")


def groups(M, steps, thr):
    """M [n, H, 3] mean paths (cm) -> list of index groups (average linkage on max-over-steps distance)."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    D = np.linalg.norm(M[:, None, :steps] - M[None, :, :steps], axis=-1).max(-1)
    lab = fcluster(linkage(squareform(D), "average"), thr, "distance")
    g = {}
    for i, l in enumerate(lab):
        g.setdefault(l, []).append(i)
    return sorted(g.values(), key=len, reverse=True)


def main(inits=range(4), settings=((10, 1.5), (50, 4.0))):
    S = np.load(os.path.join(OUT, "goal_screen.npz"), allow_pickle=True)
    texts = [p for p, _, _ in PROMPTS]
    kind = {p: k for p, k, _ in PROMPTS}
    for init in inits:
        M = np.stack([path_cm(S["chunks"][(S["text"] == t) & (S["init"] == init)][0].astype(float)).mean(0) for t in texts])
        for steps, thr in settings:
            G = groups(M, steps, thr)
            print(f"\ninit {init}, first {steps} steps, cut {thr} cm -> {len(G)} groups")
            for g in G:
                print("   {" + "; ".join(texts[i] + ("" if kind[texts[i]] == "trained" else f" [{kind[texts[i]][:5]}]")
                                        for i in g) + "}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a:
        main([int(a[0])], ((int(a[1]), float(a[2])),) if len(a) > 2 else ((10, 1.5), (50, 4.0)))
    else:
        main()
