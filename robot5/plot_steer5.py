"""figs/steer_pi05.png: the operator view at the prompt-space ridge. 'move the bowl', LIBERO-goal state 2,
seed 0, replans t 40 / 50 / 60 (out/movebowl_overlay.npz from scratch/movebowl_overlay.py; outcomes of
holding the prompt embedding on either side from out/movebowl_t50.json). The agentview image the policy
sees (rotated 180 deg to the usual upright view), with the eef path of the chunk on each side of the ridge
drawn on it: at t 50 the two sides are the two destinations, and that is the question.

    python plot_steer5.py
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, FIGS = os.path.join(HERE, "out"), os.path.join(HERE, "figs")
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE = "#2a78d6", "#eb6834"
TS = (40, 50, 60)
SCENE = json.load(open(os.path.join(OUT, "scene_px.json")))


def px(p):
    """robosuite (row, col) on the raw render -> (x, y) on the image the policy sees (raw rotated 180 deg)."""
    p = np.asarray(p, float)
    return 255 - p[..., 1], p[..., 0]


def main():
    plt.rcParams.update({"figure.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "font.size": 9.5,
                         "axes.titlesize": 10.5, "axes.titleweight": "bold"})
    d = np.load(os.path.join(OUT, "movebowl_overlay.npz"))
    r50 = json.load(open(os.path.join(OUT, "movebowl_t50.json")))
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 5.0), gridspec_kw=dict(wspace=0.06))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.76, bottom=0.17)
    for a, t in zip(ax, TS):
        img = d[f"img_{t}"][::-1, ::-1]
        a.imshow(img); a.set_xlim(0, 255); a.set_ylim(255, 0); a.set_axis_off()
        s1 = float(d[f"sigma1_{t}"])
        # which side goes where (held-embedding rollouts): colour by destination, blue = stove, orange = plate
        dest = {}
        for side in ("plus", "minus"):
            labs = [l for l, _ in r50[str(t)]["side_rollouts"][side]]
            n_st = sum("stove" in l for l in labs)
            dest[side] = ("stove", n_st, len(labs)) if n_st * 2 >= len(labs) else ("plate", len(labs) - n_st, len(labs))
        for side in ("plus", "minus"):
            col = BLUE if dest[side][0] == "stove" else ORANGE
            x, y = px(d[f"pix_{side}_{t}"].astype(float))
            a.plot(x, y, color=col, lw=2.4, solid_capstyle="round", path_effects=None)
            a.plot(x[-1], y[-1], "o", color=col, ms=9, mec=SURF, mew=1.2, zorder=5)
        x, y = px(d[f"pix_exec_{t}"].astype(float))
        a.plot(x, y, color=SURF, lw=1.2, ls=(0, (2, 2)), alpha=0.9)
        x0, y0 = px(d[f"objpix_eef_{t}"].astype(float))
        a.plot(x0, y0, "o", color=SURF, ms=5, mec=INK, mew=0.8, zorder=6)
        for k in ("stove", "plate"):
            x, y = px(np.array(SCENE["flat_stove_1" if k == "stove" else "plate_1"]["geom_rc"], float))
            a.text(x, y + 22, k, color=SURF, fontsize=8, ha="center", va="top",
                   bbox=dict(boxstyle="round,pad=0.15", fc="#00000066", ec="none"))
        jump = float(np.linalg.norm(d[f"path_plus_{t}"] - d[f"path_minus_{t}"])) * 100
        head = {40: "before the lift: no question", 50: "bowl lifted: ask", 60: "10 steps later: still open"}[t]
        a.set_title(f"t {t}   σ₁ {s1:.1f}   sides {jump:.0f} cm apart\n{head}", loc="left")
        lines = []
        for side in ("plus", "minus"):
            k, n, m = dest[side]
            lines.append((BLUE if k == "stove" else ORANGE, f"held on this side → {k} {n}/{m}"))
        for i, (col, txt) in enumerate(sorted(lines, key=lambda c: c[0] != BLUE)):
            a.text(0.0, -0.05 - 0.075 * i, "■ ", color=col, transform=a.transAxes, fontsize=10, va="top")
            a.text(0.06, -0.05 - 0.075 * i, txt, color=INK2, transform=a.transAxes, fontsize=8.8, va="top")
    fig.suptitle("The question the ridge poses, drawn on the camera image:  pi0.5, “move the bowl”, LIBERO-goal state 2",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", y=0.985)
    fig.text(0.01, 0.915, "Solid: the chunk's gripper path with the prompt embedding pushed 2.5% of the token norm to either side of the ridge "
             "(colour = where holding that side ends up).", fontsize=9.0, color=INK2)
    fig.text(0.01, 0.880, "Dashed: the chunk that executes. The two paths separate exactly when the plan reaches past the grasp.",
             fontsize=9.0, color=INK2)
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, "steer_pi05.png"), dpi=180)
    print("wrote", os.path.join(FIGS, "steer_pi05.png"))


if __name__ == "__main__":
    main()
