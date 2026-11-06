"""Confirm the decoded suggestions closed-loop: at t 0 (and t 10), re-branch 8x under each suggested
prompt and assign each branch to the nearest of the original clusters' final-eef centres."""
import json, sys
import numpy as np
sys.path.insert(0, "/tmp/claude-1000/-home-james-Projects-flow-matching-lcs/fc615ee8-3dcc-4cb5-8739-f0992e1fde17/scratchpad")
sys.path.insert(0, "/home/james/Projects/flow-matching-lcs/robot5")
import pi05flow
from env5 import Scene
from decode import A, TEXT, INIT, MID, TOP, HORIZON, branch, log

SUG = ["open the top drawer", "open the middle drawer", "open the drawer", "open a top drawer", "open a middle drawer"]
OUTJ = "/home/james/Projects/flow-matching-lcs/robot5/out/decode3_open_a_drawer_3.json"


def main():
    f = pi05flow.load()
    R = json.load(open(OUTJ))
    s = Scene(0, INIT, 0, suite_name="libero_goal")
    obs = s.reset()
    rng_main = np.random.default_rng(0)
    out = []
    for res in R:
        t = res["t"]
        x = s.get_state()
        T = np.load(OUTJ.replace(".json", f"_T{t}.npy")); cl = np.array(res["cl"])
        cent = np.stack([T[cl == c, -1].mean(0) for c in range(2)])
        # which cluster is 'top'? the counterfactual plans are not used here: name by the TOP prompt's branches
        row = dict(t=t, clusters=res["clusters"], counts={})
        for tx in SUG + [TOP, MID]:
            Tc, labs = branch(s, x, tx, f, 8, HORIZON, seed0=300)
            lab = np.argmin(np.linalg.norm(Tc[:, -1][:, None] - cent[None], axis=-1), axis=1)
            row["counts"][tx] = np.bincount(lab, minlength=2).tolist()
            ev = [sorted(l.items(), key=lambda kv: kv[1])[:1] for l in labs]
            log(f"t {t}: '{tx}': branches land in cluster 0 / 1 = {row['counts'][tx]}; events {ev}")
        out.append(row)
        json.dump(out, open(OUTJ.replace("decode3", "decode4"), "w"))
        obs = s.set_state(x)
        o = s.obs_to_lerobot(obs, TEXT)
        a = f.unnorm(f.flow_np(rng_main.standard_normal((1, A)), f.prefix([o], [TEXT]))[0])
        for j in range(10):
            obs, _, _ = s.step(a[j])
    s.close()


if __name__ == "__main__":
    main()
