"""
Videos of pi0.5 rollouts on LIBERO-spatial: the camera images the policy sees (front camera,
wrist camera inset), real time (20 Hz), several rollouts side by side.

    python video5.py layouts    # scene (7, 0), ambiguous prompt, same noise seed, fixture layout 0
                                #   vs 1 (a ~1 cm stove shift flips the bowl)   -> videos/ambig_7_0_layouts.mp4
    python video5.py task K     # task K's own instruction, layout 0, seed 0   -> videos/task_K.mp4
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from env5 import Scene, AMBIG  # noqa: E402
from rollout5 import rollout  # noqa: E402

VID = os.path.join(HERE, "videos")
FPS = 20
SIDE, INSET, CAP = 512, 168, 78
INK, MUTED, BG, ACCENT = (32, 33, 36), (110, 113, 118), (250, 250, 248), (31, 119, 180)


def font(size):
    from PIL import ImageFont
    from matplotlib import font_manager
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


def upright(img):
    return np.ascontiguousarray(img[::-1, ::-1])                  # LIBERO renders upside down


def panel(fr, title, sub, t, status):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (SIDE, SIDE + CAP), BG)
    im.paste(Image.fromarray(upright(fr["image"])).resize((SIDE, SIDE), Image.BICUBIC), (0, CAP))
    w = Image.fromarray(upright(fr["image2"])).resize((INSET, INSET), Image.BICUBIC)
    x0, y0 = SIDE - INSET - 10, CAP + SIDE - INSET - 10
    d = ImageDraw.Draw(im)
    d.rectangle([x0 - 3, y0 - 3, x0 + INSET + 2, y0 + INSET + 2], fill=(255, 255, 255))
    im.paste(w, (x0, y0))
    d.text((12, 8), title, fill=INK, font=font(17))
    d.text((12, 32), sub, fill=MUTED, font=font(14))
    d.text((12, 54), f"step {t:3d}", fill=MUTED, font=font(14))
    if status:
        d.text((110, 54), status, fill=ACCENT, font=font(14))
    d.text((x0, y0 - 20), "wrist camera", fill=(255, 255, 255), font=font(12))
    return np.asarray(im)


def run_panels(specs):
    """specs: list of dict(task, init, layout, text, seed, title, sub, names). Returns frames per panel."""
    import pi05flow
    f = pi05flow.load()
    out = []
    for sp in specs:
        s = Scene(sp["task"], sp["init"], sp["layout"])
        frames = []
        r = rollout(f, s, sp["text"], sp["seed"], frames=frames)
        s.close()
        print(f"{sp['title']} | {sp['sub']}: lifted {r['lifted']} at step {r['lift_t']}, success {r['success']}, "
              f"{r['steps']} steps", flush=True)
        out.append((sp, r, frames))
    return out


def compose(runs, path):
    import imageio.v2 as imageio
    n = max(len(fr) for _, _, fr in runs)
    os.makedirs(VID, exist_ok=True)
    w = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    for t in range(n + FPS):                                          # hold the last frame for 1 s
        cols = []
        for sp, r, fr in runs:
            k = min(t, len(fr) - 1)
            status = ""
            if r["lift_t"] >= 0 and k >= r["lift_t"]:
                status = f"lifting {sp['names'][r['lifted']]}"
            if k == len(fr) - 1 and r["success"]:
                status = "task success"
            cols.append(panel(fr[k], sp["title"], sp["sub"], k, status))
            cols.append(np.full((SIDE + CAP, 6, 3), 255, np.uint8))
        w.append_data(np.concatenate(cols[:-1], 1))
    w.close()
    print("wrote", path)


if __name__ == "__main__":
    from env5 import suite
    if sys.argv[1] == "layouts":
        names = ("bowl 1 (on the stove)", "bowl 2")
        specs = [dict(task=7, init=0, layout=l, text=AMBIG, seed=0, names=names,
                      title=f'"{AMBIG}"',
                      sub=f"task 7, init 0, fixture layout {l}, noise seed 0") for l in (0, 1)]
        compose(run_panels(specs), os.path.join(VID, "ambig_7_0_layouts.mp4"))
    elif sys.argv[1] == "task":
        k = int(sys.argv[2])
        text = suite().tasks[k].language
        specs = [dict(task=k, init=0, layout=0, text=text, seed=0, names=("bowl 1", "bowl 2"),
                      title=text if len(text) < 52 else text[:text.index(" and place")],
                      sub=f"task {k}, init 0, layout 0, seed 0")]
        compose(run_panels(specs), os.path.join(VID, f"task_{k}.mp4"))
