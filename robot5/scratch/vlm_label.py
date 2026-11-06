"""Name the two paths of the question with a small VLM, no simulator: the policy's camera image at
the stop with the two ridge-side gripper paths drawn on it (and their endpoints), and Qwen3-VL asked
(a) to detect the objects on the table (boxes -> nearest box to each endpoint), (b) what lies under a
marker at each endpoint, (c) where each drawn path ends. Simulator ground truth (held-side rollouts)
is used only to grade. Images: t 56 stop frame (movebowl_fig_r10a1s6i4), t 50 / t 60 (movebowl_overlay)."""
import json, re, sys, time
import numpy as np, torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForImageTextToText

HERE = "/home/james/Projects/flow-matching-lcs/robot5"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-VL-4B-Instruct"
SCALE = 3
BLUE, ORANGE = (42, 120, 214), (235, 104, 52)


def px(p):
    p = np.asarray(p, float); return np.stack([255 - p[..., 1], p[..., 0]], -1)


def frames():
    d = np.load(f"{HERE}/out/movebowl_fig_r10a1s6i4.npz"); J = json.load(open(f"{HERE}/out/movebowl_fig_r10a1s6i4.json"))
    fr = J["frozen"]; t = fr["t"]
    truth = {s: ("stove" if sum("stove" in l for l, _ in fr["hold"][s]) * 2 >= len(fr["hold"][s]) else "plate") for s in ("plus", "minus")}
    yield t, d[f"img_{t}"][::-1, ::-1], {s: px(d[f"pix_{s}_{t}"]) for s in ("plus", "minus")}, truth
    d = np.load(f"{HERE}/out/movebowl_overlay.npz"); r = json.load(open(f"{HERE}/out/movebowl_t50.json"))
    for t in (50, 60):
        truth = {s: ("stove" if sum("stove" in l for l, _ in r[str(t)]["side_rollouts"][s]) * 2 >= 3 else "plate") for s in ("plus", "minus")}
        yield t, d[f"img_{t}"][::-1, ::-1], {s: px(d[f"pix_{s}_{t}"]) for s in ("plus", "minus")}, truth


def draw(img, paths, marker_only=None):
    im = Image.fromarray(np.ascontiguousarray(img)).resize((256 * SCALE, 256 * SCALE), Image.BICUBIC); dr = ImageDraw.Draw(im)
    for s, col in (("plus", BLUE), ("minus", ORANGE)):
        if marker_only is not None and s != marker_only:
            continue
        p = paths[s] * SCALE
        if marker_only is None:
            dr.line([tuple(q) for q in p], fill=col, width=5, joint="curve")
        x, y = p[-1]; dr.ellipse([x - 12, y - 12, x + 12, y + 12], fill=col, outline=(255, 255, 255), width=3)
    return im


class VLM:
    def __init__(self):
        self.proc = AutoProcessor.from_pretrained(MODEL)
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")

    def ask(self, im, text, max_new=200):
        msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": text}]}]
        inputs = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        torch.cuda.synchronize(); T = time.time()
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        torch.cuda.synchronize()
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip(), time.time() - T


def boxes_from(text, W):
    """Parse [{"bbox_2d": [x1,y1,x2,y2], "label": ...}] (Qwen3-VL: absolute pixels of the input image, or 0-1000)."""
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    if isinstance(items, dict):                 # {name: {"x":, "y":}} or {name: [x, y]} -> point boxes
        for lab, v in items.items():
            try:
                q = [v["x"], v["y"]] if isinstance(v, dict) and "x" in v else (list(v.values())[0] if isinstance(v, dict) else list(v))
                q = [float(x) for x in q]
            except Exception:
                continue
            if len(q) == 2:
                out.append((lab, np.array([q[0] - 30, q[1] - 30, q[0] + 30, q[1] + 30], float) / 1000 * W))
            elif len(q) == 4:
                out.append((lab, np.array(q, float) / 1000 * W))
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        b = it.get("bbox_2d") or it.get("bbox") or it.get("point_2d"); lab = it.get("label", "?")
        if b is None:
            continue
        b = np.array(b, float)
        if len(b) == 2:
            b = np.array([b[0] - 30, b[1] - 30, b[0] + 30, b[1] + 30])
        out.append((lab, b / 1000 * W))         # Qwen3-VL: coordinates normalised to 0-1000
    return out


def nearest_box(p, boxes):
    best = None
    for lab, b in boxes:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        inside = b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]
        dist = 0.0 if inside else np.hypot(max(b[0] - p[0], 0, p[0] - b[2]), max(b[1] - p[1], 0, p[1] - b[3]))
        if best is None or dist < best[0]:
            best = (dist, lab)
    return best


def main():
    vlm = VLM(); W = 256 * SCALE; res = {}
    for t, img, paths, truth in frames():
        print(f"\n=== t {t}: truth plus={truth['plus']} minus={truth['minus']}")
        plain = Image.fromarray(np.ascontiguousarray(img)).resize((W, W), Image.BICUBIC)
        plain.save(f"{HERE}/out/vlm_t{t}_plain.png")
        # (a) grounding on the plain image
        txt, dt = vlm.ask(plain, "Detect every object on the table (not the robot). Output a JSON list of {\"bbox_2d\": [x1, y1, x2, y2], \"label\": name}.", 400)
        boxes = boxes_from(txt, W)
        print(f"(a) detect [{dt:.1f}s]: {txt[:400]}")
        ends = {s: paths[s][-1] * SCALE for s in paths}
        a = {s: nearest_box(ends[s], boxes) for s in ends}
        print("    endpoints ->", {s: (a[s][1], round(a[s][0])) if a[s] else None for s in a})
        # (b) marker at each endpoint, one at a time
        b = {}
        for s, name in (("plus", "blue"), ("minus", "orange")):
            im = draw(img, paths, marker_only=s); im.save(f"{HERE}/out/vlm_t{t}_marker_{s}.png")
            txt, dt = vlm.ask(im, f"A {name} circle is drawn on the image. Which object on the table is directly under the {name} circle? Answer with one or two words.", 20)
            b[s] = txt; print(f"(b) marker {s} ({name}) [{dt:.1f}s]: {txt}")
        # (c) both paths drawn, the question as the operator would see it
        im = draw(img, paths); im.save(f"{HERE}/out/vlm_t{t}_paths.png")
        txt, dt = vlm.ask(im, "Two candidate paths for the robot gripper are drawn on the image, a blue one and an orange one, each ending at a circle. "
                              "Which object on the table does each path end at? Answer as JSON {\"blue\": name, \"orange\": name}.", 60)
        print(f"(c) paths [{dt:.1f}s]: {txt}")
        res[t] = dict(truth=truth, detect=txt and boxes and [(l, b_.round().tolist()) for l, b_ in boxes], endpoints={s: a[s][1] if a[s] else None for s in a}, marker=b, paths=txt)
    json.dump(res, open(f"{HERE}/out/vlm_label_{MODEL.split('/')[-1]}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
