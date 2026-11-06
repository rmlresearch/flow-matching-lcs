"""Re-score the VLM naming on the 2-D cases only (destinations are distinct table objects; the drawer
frames are excluded), offline from the saved Qwen outputs in out/vlm_label3.json, with
  - detection cached across the episode: a box missing in a frame is taken from the nearest other frame
    of the same episode (objects are static apart from the held one, which is excluded anyway);
  - a motion gate: a chunk that moves < GATE cm horizontally is named by the box nearest its endpoint,
    not by a ray along a noise direction;
  - oversized boxes (> BIG of the image) never taken by a ray hit, only when the endpoint is inside.
Reports vs the executed outcome, overall and by chunk phase (transport / reach / lift / place).
-> out/vlm_score2d.json, and prints the table."""
import json, re, sys
import numpy as np
sys.path.insert(0, "/home/james/Projects/flow-matching-lcs/robot5/scratch")
from vlm_label import boxes_from, SCALE, HERE, px
from vlm_label3 import all_frames, grade

R = json.load(open(f"{HERE}/out/vlm_label3.json"))["frames"]
D = np.load(f"{HERE}/out/vlm_cases.npz")
W = 256 * SCALE; GATE, BIG = 8.0, 0.25


def world_path(fk, side):
    if fk.startswith("move the bowl|2|0|"):
        t = int(fk.split("|")[-1])
        src = np.load(f"{HERE}/out/movebowl_fig_r10a1s6i4.npz") if t == 56 else np.load(f"{HERE}/out/movebowl_overlay.npz")
        return src[f"path_{side}_{t}"]
    return D[f"path_{side}|{fk}"]


def phase(p):
    dh = np.linalg.norm(p[-1, :2] - p[0, :2]) * 100; dv = (p[-1, 2] - p[0, 2]) * 100
    if dh >= GATE and dh >= abs(dv):
        return "transport", dh, dv
    if dv <= -GATE:
        return "reach/descend", dh, dv
    if dv >= GATE:
        return "lift", dh, dv
    return "place/hover", dh, dv


def episode_boxes(fk):
    """Boxes of this frame, plus any label missing here taken from the nearest frame of the same episode."""
    ep = "|".join(fk.split("|")[:3]); t = int(fk.split("|")[3])
    own = boxes_from(R[fk]["objects"]["raw"], W); have = {l for l, _ in own}
    others = sorted((abs(int(k.split("|")[3]) - t), k) for k in R if "|".join(k.split("|")[:3]) == ep and k != fk)
    added = []
    for _, k in others:
        for l, b in boxes_from(R[k]["objects"]["raw"], W):
            if l not in have and l in ("stove", "plate", "bowl", "box", "bottle"):
                own.append((l, b)); have.add(l); added.append(l)
    return own, added


def assign(path_px, p_world, boxes, start):
    s0, e = np.asarray(start) * SCALE, np.asarray(path_px[-1]) * SCALE; d = e - s0; d = d / (np.linalg.norm(d) + 1e-6)
    inside = lambda b, q: b[0] <= q[0] <= b[2] and b[1] <= q[1] <= b[3]
    cand = [(l, b) for l, b in boxes if not inside(b, s0)]
    ph, dh, dv = phase(p_world)
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1]) / W ** 2
    ins = [(area(b), l) for l, b in cand if inside(b, e)]
    if ins:
        return min(ins)[1], "inside"
    centre = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    if dh < GATE:                                                   # motion gate: no usable direction
        return min((np.linalg.norm(centre(b) - e), l) for l, b in cand)[1], "nearest(gate)"
    hits = [(s_, l) for l, b in cand if area(b) < BIG for s_ in np.arange(0, 700, 4) if inside(b, e + s_ * d)]
    if hits:
        return min(hits)[1], "ray"
    best = None
    for l, b in cand:
        r = centre(b) - e; along, perp = r @ d, abs(r[0] * d[1] - r[1] * d[0])
        if along < -max(b[2] - b[0], b[3] - b[1]) / 2:
            continue
        sc = perp + 0.1 * max(along, 0)
        if best is None or sc < best[0]:
            best = (sc, l)
    return (best[1], "line") if best else ("?", "none")


def main():
    out, tally = {}, {}
    def add(k, ok):
        tally.setdefault(k, [0, 0]); tally[k][0] += ok; tally[k][1] += 1
    for fk, text, img, paths, flat, truth, hold in all_frames():
        if "drawer" in fk:
            continue
        boxes, added = episode_boxes(fk)
        same = truth["plus"] == truth["minus"]
        row = dict(truth=truth, added=added, same=same)
        line = f"{fk:44s} " + ("(same) " if same else "       ")
        for s in ("plus", "minus"):
            pw = world_path(fk, s); ph, dh, dv = phase(pw)
            name, how = assign(flat[s], pw, boxes, flat["plus"][0] * 1.0) if boxes else ("?", "none")
            ok = grade(name, truth[s])
            row[s] = dict(name=name, how=how, phase=ph, dh=round(dh), dv=round(dv), ok=ok)
            if ok is not None and not same:
                add("all", ok); add(ph, ok)
            line += f"{s}: {name:7s} ({how:13s} {ph:13s}) {'OK' if ok else 'X' if ok is not None else '-'}   "
        print(line + (f" [+{','.join(added)} cached]" if added else ""))
        out[fk] = row
    json.dump(dict(frames=out, tally=tally), open(f"{HERE}/out/vlm_score2d.json", "w"), indent=1)
    print("\n2-D cases, vs executed outcome, sides of frames whose two outcomes differ:")
    for k, v in tally.items():
        print(f"  {k:14s} {v[0]}/{v[1]}")


if __name__ == "__main__":
    main()
