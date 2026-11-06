"""VLM naming on the open frames of out/vlm_cases.{npz,json} (+ the three "move the bowl" state-2 seed-0
frames): Qwen3-VL-4B detection (prompt 2, and a variant that also asks for the cabinet's drawers), the
geometric assignment of vlm_assign.py, graded against the held-side rollouts. Also the direct arrows
question for comparison. -> out/vlm_label3.json, out/vlm3_<case>_<t>.png (boxes + paths)."""
import json, re, sys, time
import numpy as np, torch
from PIL import Image, ImageDraw
sys.path.insert(0, "/home/james/Projects/flow-matching-lcs/robot5/scratch")
from vlm_label import VLM, boxes_from, px, SCALE, HERE, MODEL, frames as frames_state2
from vlm_label2 import draw_arrows, HEAD
from vlm_assign import assign
from robosuite.utils.camera_utils import project_points_from_world_to_camera
W2P = np.load(f"{HERE}/out/cam_W2P.npy"); Z_TABLE = 0.91
WORLD = json.load(open(f"{HERE}/out/vlm_world.json")) if __import__("os").path.exists(f"{HERE}/out/vlm_world.json") else {}


def flat_px(path_world):
    """The path with every point dropped to the table plane, projected: the horizontal motion only, so a
    lift does not read as a move towards the back of the scene."""
    p = np.array(path_world, float).copy(); p[:, 2] = Z_TABLE
    return px(project_points_from_world_to_camera(p, W2P, 256, 256))

GROUND = {"objects": "Outline the position of each object on the table and output all the coordinates in JSON format. Name each object (for example plate, bowl, stove, bottle, cabinet, box).",
          "objects+drawers": "Outline the position of each object on the table, and of each drawer of the cabinet separately (top, middle, bottom), and output all the coordinates in JSON format. "
                             "Name each one (for example plate, bowl, stove, bottle, box, top drawer, middle drawer, bottom drawer)."}
SYN = {"cheese": ("cheese", "box", "cream"), "bowl": ("bowl",), "stove": ("stove", "burner", "cooktop"), "plate": ("plate", "dish"),
       "top drawer": ("top drawer",), "middle drawer": ("middle drawer",), "bottom drawer": ("bottom drawer",), "cabinet": ("cabinet",), "rack": ("rack",)}


def dest(label, text):
    """Destination word of a rollout label. 'x->y' -> y, except pick choices ('put something on the plate': x)."""
    if label == "none":
        return "none"
    if "->" in label:
        a, b = label.split("->")
        return a if "something" in text else b.replace("top drawer", "top drawer")
    return label.replace(" open", "")


def truth_of(hold, text):
    labs = [dest(l, text) for l in hold]
    return max(set(labs), key=labs.count)


def grade(name, truth):
    if name is None or truth == "none":
        return None
    n = name.lower()
    if truth in SYN:
        return any(s in n for s in SYN[truth]) and not (truth.endswith("drawer") and "drawer" in n and truth.split()[0] not in n)
    return truth.split()[0] in n


def all_frames():
    J = json.load(open(f"{HERE}/out/vlm_cases.json")); D = np.load(f"{HERE}/out/vlm_cases.npz")
    for key, v in J.items():
        for r in v["rows"]:
            if r.get("open"):
                fk = f"{key}|{r['t']}"
                paths = {s: px(D[f"pix_{s}|{fk}"]) for s in ("plus", "minus")}
                flat = {s: flat_px(D[f"path_{s}|{fk}"]) for s in ("plus", "minus")}
                truth = {s: truth_of(r["hold"][s], v["text"]) for s in ("plus", "minus")}
                yield fk, v["text"], D[f"img|{fk}"][::-1, ::-1], paths, flat, truth, r["hold"]
    d2 = {56: (np.load(f"{HERE}/out/movebowl_fig_r10a1s6i4.npz"), 56)}
    ov = np.load(f"{HERE}/out/movebowl_overlay.npz")
    for t, img, paths, truth in frames_state2():
        src = d2[56][0] if t == 56 else ov
        flat = {s: flat_px(src[f"path_{s}_{t}"]) for s in ("plus", "minus")}
        yield f"move the bowl|2|0|{t}", "move the bowl", img, paths, flat, truth, None


def main():
    vlm = VLM(); W = 256 * SCALE; res = {}
    tally = {k: [0, 0] for k in [f"{g}/{m}" for g in GROUND for m in ("raw", "flat")] + ["arrows"]}
    tally_w = {k: [0, 0] for k in tally}
    for fk, text, img, paths, flat, truth, hold in all_frames():
        wt = WORLD.get(fk); world = {s: wt[s][0] for s in ("plus", "minus")} if wt else None
        row = dict(text=text, truth=truth, hold=hold, world=world)
        same = truth["plus"] == truth["minus"]
        print(f"\n=== {fk}: outcome + {truth['plus']}  - {truth['minus']}" + ("  (same)" if same else "") +
              (f"   | chunk (3-D) + {world['plus']}  - {world['minus']}" if world else ""))
        plain = Image.fromarray(np.ascontiguousarray(img)).resize((W, W), Image.BICUBIC)
        for gk, g in GROUND.items():
            txt, dt = vlm.ask(plain, g, 500); boxes = boxes_from(txt, W)
            row[gk] = dict(labels=[l for l, _ in boxes], raw=txt, t=dt)
            print(f"  [{gk}] {dt:.1f}s labels {[l for l, _ in boxes]}")
            for mk, pp in (("raw", paths), ("flat", flat)):
                a = {s: assign(pp[s], boxes, start=pp["plus"][0]) if boxes else ("?", "none", []) for s in pp}
                ok = {s: grade(a[s][0], truth[s]) for s in pp}
                okw = {s: grade(a[s][0], world[s]) for s in pp} if world else {s: None for s in pp}
                row[f"{gk}/{mk}"] = dict(plus=[a["plus"][0], a["plus"][1]], minus=[a["minus"][0], a["minus"][1]], ok=ok, ok_world=okw)
                for s in pp:
                    if ok[s] is not None and not same:
                        tally[f"{gk}/{mk}"][0] += ok[s]; tally[f"{gk}/{mk}"][1] += 1
                    if okw[s] is not None:
                        tally_w[f"{gk}/{mk}"][0] += okw[s]; tally_w[f"{gk}/{mk}"][1] += 1
                mark = lambda o: 'OK' if o else 'X' if o is not None else '-'
                print(f"      {mk:4s} + {a['plus'][0]:14s} ({a['plus'][1]:7s}) outcome {mark(ok['plus'])} chunk {mark(okw['plus'])}   "
                      f"- {a['minus'][0]:14s} ({a['minus'][1]:7s}) outcome {mark(ok['minus'])} chunk {mark(okw['minus'])}")
            if gk == "objects":
                dbg = draw_arrows(img, paths); dr = ImageDraw.Draw(dbg)
                for l, b in boxes:
                    dr.rectangle(b.tolist(), outline=(255, 255, 0), width=2); dr.text((b[0] + 3, b[1] + 2), l, fill=(255, 255, 0))
                dbg.save(f"{HERE}/out/vlm3_{fk.replace('|', '_').replace(' ', '-')}.png")
        im = draw_arrows(img, paths)
        txt, dt = vlm.ask(im, HEAD, 60)
        try:
            ans = json.loads(re.search(r"\{.*\}", txt, re.S).group(0)); ans = {"plus": ans.get("blue"), "minus": ans.get("orange")}
        except Exception:
            ans = {"plus": None, "minus": None}
        ok = {s: grade(ans[s], truth[s]) for s in paths}
        okw = {s: grade(ans[s], world[s]) for s in paths} if world else {s: None for s in paths}
        for s in paths:
            if ok[s] is not None and not same:
                tally["arrows"][0] += ok[s]; tally["arrows"][1] += 1
            if okw[s] is not None:
                tally_w["arrows"][0] += okw[s]; tally_w["arrows"][1] += 1
        row["arrows"] = dict(answer=ans, ok=ok, ok_world=okw, t=dt)
        print(f"  [arrows] {dt:.1f}s + {ans['plus']} {'OK' if ok['plus'] else 'X' if ok['plus'] is not None else '-'}   - {ans['minus']} {'OK' if ok['minus'] else 'X' if ok['minus'] is not None else '-'}")
        res[fk] = row
        json.dump(dict(frames=res, tally=tally, tally_world=tally_w), open(f"{HERE}/out/vlm_label3.json", "w"), indent=1)
    print("\nTALLY vs executed outcome (frames whose two sides differ):", {k: f"{v[0]}/{v[1]}" for k, v in tally.items()})
    print("TALLY vs chunk-level 3-D name (all graded paths):", {k: f"{v[0]}/{v[1]}" for k, v in tally_w.items()})


if __name__ == "__main__":
    main()
