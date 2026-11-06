"""Ground truth for the t 0 clusters: the same 16 branches (same noise seeds) run to 130 steps,
first goal event per branch, compared with decode3's cluster labels; and the per-branch eef at
t 60 / t 100 to see what separates the clusters."""
import json, sys
import numpy as np
sys.path.insert(0, "/tmp/claude-1000/-home-james-Projects-flow-matching-lcs/fc615ee8-3dcc-4cb5-8739-f0992e1fde17/scratchpad")
sys.path.insert(0, "/home/james/Projects/flow-matching-lcs/robot5")
import pi05flow
from env5 import Scene
from decode import TEXT, INIT, N_BR, branch, log

OUTJ = "/home/james/Projects/flow-matching-lcs/robot5/out/decode3_open_a_drawer_3.json"


def main():
    f = pi05flow.load()
    res = json.load(open(OUTJ))[0]
    cl = np.array(res["cl"]); T0 = np.load(OUTJ.replace(".json", "_T0.npy"))
    s = Scene(0, INIT, 0, suite_name="libero_goal")
    s.reset(); x = s.get_state()
    T, labs = branch(s, x, TEXT, f, N_BR, 130)
    same = np.abs(T[:, :61] - T0).max()
    log(f"reproduces decode3's branches to {same:.2e} m")
    for i in range(N_BR):
        ev = sorted(labs[i].items(), key=lambda kv: kv[1])
        log(f"branch {i:2d} cluster {cl[i]}: eef t60 {np.round(100 * T[i, 60], 1).tolist()} t100 {np.round(100 * T[i, 100], 1).tolist()} "
            f"t130 {np.round(100 * T[i, 130], 1).tolist()}  events {ev}")
    json.dump(dict(cl=cl.tolist(), events=[sorted(l.items(), key=lambda kv: kv[1]) for l in labs]),
              open(OUTJ.replace("decode3", "decode5"), "w"))
    np.save(OUTJ.replace("decode3", "decode5").replace(".json", "_T0.npy"), T)
    s.close()


if __name__ == "__main__":
    main()
