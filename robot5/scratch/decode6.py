"""Outcome-level confirmation at t 0: 8 branches to 130 steps under the decoded middle suggestion,
the neutral edit and the ambiguous prompt; first goal event per branch."""
import sys, json
import numpy as np
sys.path.insert(0, "/tmp/claude-1000/-home-james-Projects-flow-matching-lcs/fc615ee8-3dcc-4cb5-8739-f0992e1fde17/scratchpad")
sys.path.insert(0, "/home/james/Projects/flow-matching-lcs/robot5")
import pi05flow
from env5 import Scene
from decode import INIT, branch, log

def main():
    f = pi05flow.load()
    s = Scene(0, INIT, 0, suite_name="libero_goal"); s.reset(); x = s.get_state()
    out = {}
    for tx in ["open the middle drawer", "open the top drawer", "open the drawer"]:
        T, labs = branch(s, x, tx, f, 8, 130, seed0=300)
        ev = [sorted(l.items(), key=lambda kv: kv[1])[:1] for l in labs]
        out[tx] = ev
        log(f"t 0: '{tx}' to 130 steps: {ev}")
    json.dump(out, open("/home/james/Projects/flow-matching-lcs/robot5/out/decode6_open_a_drawer_3.json", "w"))
    s.close()

if __name__ == "__main__":
    main()
