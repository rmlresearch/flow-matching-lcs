"""World-space truth for naming a path: replay each vlm_cases episode (deterministic) and at every open
replan record the true object positions (geometry centres of stove, plate, bowl, cheese, bottle, cabinet
and its three drawer handles), then name each side path by the same rule as the VLM assignment but in
3-D (nearest object to the endpoint within R, else first object within R of the ray from the endpoint
along end - start, else nearest to the line); the object nearest the start is the held one and excluded.
-> out/vlm_world.json: per frame, object positions and the chunk-level name of each side."""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/home/james/Projects/flow-matching-lcs/robot5")
import pi05flow
from pi05flow import H, D_ACT
from env5 import Scene
import monitor5 as M

J = json.load(open("/home/james/Projects/flow-matching-lcs/robot5/out/vlm_cases.json"))
D = np.load("/home/james/Projects/flow-matching-lcs/robot5/out/vlm_cases.npz")
A = H * D_ACT
OBJ = {"stove": "flat_stove_1", "plate": "plate_1", "bowl": "akita_black_bowl_1", "cheese": "cream_cheese_1", "bottle": "wine_bottle_1", "cabinet": "wooden_cabinet_1", "rack": "wine_rack_1"}
R = 0.12   # m


def geom_centre(s, body):
    sim, e = s.sim, s.env._env.env; bid = e.obj_body_id[body]
    def under(b):
        while b > 0:
            if b == bid:
                return True
            b = sim.model.body_parentid[b]
        return False
    ids = [g for g in range(sim.model.ngeom) if under(sim.model.geom_bodyid[g])]
    return sim.data.geom_xpos[ids].mean(0)


def drawers(s):
    """Positions of the cabinet's drawer bodies (names containing 'drawer'), labelled top/middle/bottom by height."""
    sim = s.sim; names = [n for n in sim.model.body_names if "cabinet" in n and "drawer" in n]
    pos = {n: np.array(sim.data.body_xpos[sim.model.body_name2id(n)]) for n in names}
    if not pos:
        return {}
    order = sorted(pos, key=lambda n: -pos[n][2])
    tags = ["top drawer", "middle drawer", "bottom drawer"][:len(order)]
    return {t: pos[n] for t, n in zip(tags, order)}


def name_path(path, objs, held):
    s0, e = path[0], path[-1]; d = e - s0; d /= np.linalg.norm(d) + 1e-6
    cand = {k: v for k, v in objs.items() if k != held}
    near = {k: np.linalg.norm(v - e) for k, v in cand.items()}
    k0 = min(near, key=near.get)
    if near[k0] < R:
        return k0, "near"
    hits = {}
    for k, v in cand.items():
        r = v - e; along = r @ d
        if along > 0 and np.linalg.norm(r - along * d) < R:
            hits[k] = along
    if hits:
        return min(hits, key=hits.get), "ray"
    best = min(cand, key=lambda k: np.linalg.norm((cand[k] - e) - max((cand[k] - e) @ d, 0) * d))
    return best, "line"


def main():
    f = pi05flow.load(); out = {}
    for key, v in J.items():
        opens = [r["t"] for r in v["rows"] if r.get("open")]
        if not opens:
            continue
        text, init, seed = v["text"], v["init"], v["seed"]
        s = Scene(0, init, 0, suite_name="libero_goal"); obs = s.reset(); x0 = s.get_state(); obs = s.set_state(x0)
        rng = np.random.default_rng(1000 + seed); t = 0
        while t <= max(opens):
            o = s.obs_to_lerobot(obs, text); z = rng.standard_normal((1, A))
            if t in opens:
                objs = {k: geom_centre(s, b) for k, b in OBJ.items()}; objs.update(drawers(s))
                eef = s.eef(); fk = f"{key}|{t}"
                held = min(objs, key=lambda k: np.linalg.norm(objs[k] - eef)) if min(np.linalg.norm(objs[k] - eef) for k in objs) < 0.08 else None
                names = {side: name_path(D[f"path_{side}|{fk}"], objs, held) for side in ("plus", "minus")}
                out[fk] = dict(objects={k: v_.tolist() for k, v_ in objs.items()}, eef=eef.tolist(), held=held, plus=list(names["plus"]), minus=list(names["minus"]))
                M.log(f"{fk}: held {held}; + {names['plus']}  - {names['minus']}")
            a = f.unnorm(f.flow_np(z, f.prefix([o], [text]))[0])
            for j in range(10):
                obs, _, _ = s.step(a[j]); t += 1
        s.close()
    # the three state-2 seed-0 frames
    json.dump(out, open("/home/james/Projects/flow-matching-lcs/robot5/out/vlm_world.json", "w"), indent=1)


if __name__ == "__main__":
    main()
