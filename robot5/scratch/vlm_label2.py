"""vlm_label v2: (a) Qwen-style grounding prompt -> named boxes; each path assigned by the ray of its last
segment (first box the ray enters, else the box nearest the endpoint); (c) both paths drawn with arrowheads,
'which object is each path heading towards'. Frames and truth as in vlm_label.py."""
import json, re, sys, time
import numpy as np, torch
from PIL import Image, ImageDraw
sys.path.insert(0, "/home/james/Projects/flow-matching-lcs/robot5/scratch")
from vlm_label import frames, VLM, boxes_from, px, SCALE, BLUE, ORANGE, HERE, MODEL

GROUND = ["Locate every object on the table (ignore the robot) and output their bounding boxes in JSON format, as a list of {\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"<object name>\"}.",
          "Outline the position of each object on the table and output all the coordinates in JSON format. Name each object (for example plate, bowl, stove, bottle, cabinet, box)."]
HEAD = "Two candidate paths for the robot gripper are drawn as arrows, blue and orange. They show where the gripper would move next. Which object on the table is each arrow heading towards? Answer as JSON {\"blue\": name, \"orange\": name}."


def draw_arrows(img, paths):
    im = Image.fromarray(np.ascontiguousarray(img)).resize((256 * SCALE, 256 * SCALE), Image.BICUBIC); dr = ImageDraw.Draw(im)
    for s, col in (("plus", BLUE), ("minus", ORANGE)):
        p = paths[s] * SCALE; dr.line([tuple(q) for q in p], fill=col, width=5, joint="curve")
        d = p[-1] - p[-8]; d = d / (np.linalg.norm(d) + 1e-6); n = np.array([-d[1], d[0]])
        tip = p[-1] + 14 * d; dr.polygon([tuple(tip), tuple(p[-1] - 10 * d + 11 * n), tuple(p[-1] - 10 * d - 11 * n)], fill=col, outline=(255, 255, 255))
    return im


def ray_assign(path, boxes, n_tail=8):
    """First box entered by the ray from the endpoint along the path's last segment; else nearest box."""
    p = path * SCALE; e = p[-1]; d = p[-1] - p[-n_tail]; d = d / (np.linalg.norm(d) + 1e-6)
    hits = []
    for lab, b in boxes:
        if b[0] <= e[0] <= b[2] and b[1] <= e[1] <= b[3]:
            hits.append((0.0, lab, "inside")); continue
        # march along the ray
        for s_ in np.arange(0, 600, 4):
            q = e + s_ * d
            if b[0] <= q[0] <= b[2] and b[1] <= q[1] <= b[3]:
                hits.append((s_, lab, "ray")); break
    if hits:
        return min(hits)
    cx = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, lab) for lab, b in boxes]
    return min((np.hypot(x - e[0], y - e[1]), lab, "nearest") for x, y, lab in cx)


def main():
    vlm = VLM(); W = 256 * SCALE; res = {}
    for t, img, paths, truth in frames():
        print(f"\n=== t {t}: truth plus={truth['plus']} minus={truth['minus']}")
        plain = Image.fromarray(np.ascontiguousarray(img)).resize((W, W), Image.BICUBIC)
        row = dict(truth=truth)
        for i, g in enumerate(GROUND):
            txt, dt = vlm.ask(plain, g, 400); boxes = boxes_from(txt, W)
            labs = [l for l, _ in boxes]
            a = {s: ray_assign(paths[s], boxes) if boxes else None for s in paths}
            print(f"(a{i}) detect [{dt:.1f}s] labels {labs}")
            print(f"      ray -> plus {a['plus']}  minus {a['minus']}")
            row[f"ground{i}"] = dict(labels=labs, boxes=[(l, b.round().tolist()) for l, b in boxes], **{s: a[s] and [float(a[s][0]), a[s][1], a[s][2]] for s in a})
            dbg = draw_arrows(img, paths); dr = ImageDraw.Draw(dbg)
            for l, b in boxes:
                dr.rectangle(b.tolist(), outline=(255, 255, 0), width=2); dr.text((b[0] + 3, b[1] + 2), l, fill=(255, 255, 0))
            dbg.save(f"{HERE}/out/vlm_t{t}_boxes{i}.png")
        im = draw_arrows(img, paths); im.save(f"{HERE}/out/vlm_t{t}_arrows.png")
        txt, dt = vlm.ask(im, HEAD, 60); print(f"(c') heading [{dt:.1f}s]: {txt.replace(chr(10), ' ')}"); row["heading"] = txt
        res[t] = row
    json.dump(res, open(f"{HERE}/out/vlm_label2_{MODEL.split('/')[-1]}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
