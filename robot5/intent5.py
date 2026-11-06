"""
Which drawer does the ambiguous prompt *intend* at each replan?  (python intent5.py)
Plan intention along closed-loop rollouts: at every replan of 'open a drawer' (prefix batch 1),
the plan's position lambda between the plans the same state and noise give under the two specific
prompts (0 = middle-drawer plan, 1 = top-drawer plan), at chunk steps 10 / 30 / 50, plus events."""
import sys
import numpy as np
sys.path.insert(0, "/home/james/Projects/flow-matching-lcs/robot5")
import pi05flow
from pi05flow import H, D_ACT
from env5 import Scene, GOAL_NAMES, EVENT_NAMES
from goal5 import path_cm

NAMES = GOAL_NAMES + EVENT_NAMES
MID, TOP = "open the middle drawer of the cabinet", "open the top drawer and put the bowl inside"


def main():
    A = H * D_ACT
    f = pi05flow.load()
    for text, init, seed in [("open a drawer", 3, 0), ("open a drawer", 3, 1), ("open a drawer", 1, 0), (MID, 3, 0), (TOP, 3, 0)]:
        s = Scene(0, init, 0, suite_name="libero_goal")
        obs = s.reset()
        rng = np.random.default_rng(seed)
        t, first, lams = 0, {}, []
        while t < 160:
            o = s.obs_to_lerobot(obs, text)
            z = rng.standard_normal((1, A))
            p = {k: path_cm(f.unnorm(f.flow_np(z, f.prefix([o], [k]))[0])) for k in (text, MID, TOP)}
            v = p[TOP] - p[MID]
            lam = [float(((p[text][i] - p[MID][i]) @ v[i]) / max(v[i] @ v[i], 1e-9)) for i in (9, 29, 49)]
            sep = [float(np.linalg.norm(v[i])) for i in (9, 29, 49)]
            lams.append((t, lam, sep))
            a = f.unnorm(f.flow_np(z, f.prefix([o], [text]))[0])
            for j in range(10):
                obs, _, _ = s.step(a[j]); t += 1
                for i, q in enumerate(s.preds()):
                    if q and NAMES[i] not in first:
                        first[NAMES[i]] = t
        print(f"\n'{text}' init {init} seed {seed}: events {first}")
        print("   t | lambda at step 10 / 30 / 50 | mid-top plan separation (cm) at 10 / 30 / 50")
        for t_, lam, sep in lams:
            print(f"  {t_:3d} | {lam[0]:5.2f} {lam[1]:5.2f} {lam[2]:5.2f} | {sep[0]:5.1f} {sep[1]:5.1f} {sep[2]:5.1f}", flush=True)
        s.close()


if __name__ == "__main__":
    main()
