"""Assign each ridge-side path to a VLM-detected box, offline from out/vlm_label2_*.json.
Rule: direction = chunk displacement (end - start); boxes containing the start (the held object) are
excluded; the path's box is the one its endpoint is inside, else the first box the ray from the endpoint
hits, else the nearest box centre to the endpoint."""
import json, sys, numpy as np
sys.path.insert(0, "/home/james/Projects/flow-matching-lcs/robot5/scratch")
from vlm_label import frames, SCALE, HERE
J = json.load(open(f"{HERE}/out/vlm_label2_Qwen3-VL-4B-Instruct.json"))


def assign(path, boxes, start=None):
    """Score = perpendicular distance of the box centre to the line (endpoint, chunk direction), boxes
    behind the endpoint (centre projection < -half box size) and the held object (box containing the
    start) excluded; ties to the closer centre. Returns (label, score, held)."""
    p = path * SCALE; s0 = p[0] if start is None else start * SCALE; e = p[-1]
    d = e - s0; d = d / (np.linalg.norm(d) + 1e-6)
    inside = lambda b, q: b[0] <= q[0] <= b[2] and b[1] <= q[1] <= b[3]
    held = [l for l, b in boxes if inside(b, s0)]
    cand = [(l, np.array(b)) for l, b in boxes if not inside(b, s0)]
    ins = [(float((b[2] - b[0]) * (b[3] - b[1])), l) for l, b in cand if inside(b, e)]
    if ins:                                              # 1. endpoint inside a box (the smallest, if several)
        return min(ins)[1], "inside", held
    hits = [(s_, l) for l, b in cand for s_ in np.arange(0, 700, 4) if inside(b, e + s_ * d)]
    if hits:                                             # 2. first box the ray enters
        return min(hits)[1], "ray", held
    best = None
    for l, b in boxes:                                   # 3. nearest box centre to the line, not behind
        if inside(b, s0):
            continue
        c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]); r = c - e
        along, perp = r @ d, abs(r[0] * d[1] - r[1] * d[0])
        half = max(b[2] - b[0], b[3] - b[1]) / 2
        if along < -half:
            continue
        score = perp + 0.1 * max(along, 0)
        if best is None or score < best[0]:
            best = (score, l)
    return (best[1], f"{best[0]:.0f}", held) if best else ("?", "none", held)


ok = 0; n = 0
for t, img, paths, truth in frames():
    for g in ("ground0", "ground1"):
        boxes = J[str(t)][g]["boxes"]
        out = {s: assign(paths[s], boxes, start=paths["plus"][0]) for s in paths}
        hit = {s: out[s][0] == truth[s] for s in paths}
        ok += sum(hit.values()); n += 2
        print(f"t {t} {g}: plus -> {out['plus'][0]:6s} ({out['plus'][1]}, truth {truth['plus']}) {'OK' if hit['plus'] else 'X'}   "
              f"minus -> {out['minus'][0]:6s} ({out['minus'][1]}, truth {truth['minus']}) {'OK' if hit['minus'] else 'X'}   held: {out['plus'][2]}")
print(f"{ok}/{n} correct")
final = {}
for t, img, paths, truth in frames():
    boxes = J[str(t)]["ground1"]["boxes"]
    final[t] = {s: list(assign(paths[s], boxes, start=paths["plus"][0])[:2]) for s in paths}
    final[t]["truth"] = truth
json.dump(final, open(f"{HERE}/out/vlm_assign.json", "w"), indent=1)
