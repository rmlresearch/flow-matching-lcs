"""figs/ask_pi05.png: the asynchronous stop-and-ask loop on one episode, pi0.5 'move the bowl',
LIBERO-goal state 2, seed 0 (out/movebowl_fig_<tag>.{npz,json} from scratch/movebowl_fig.py).
A: what the operator's screen shows at a few replans, the chunk the robot executes (dashed) and the
chunk on either side of the prompt-space ridge (solid): one path while the task is not steerable, two
once it is, and at the stop the two paths are the question. B: the trigger signal against the step
clock, with the background v1 refreshes, the trigger, the stop and the moment the question is on
screen; the stop of the measured serial schedule (r20a3) marked where it lands. C: the two answers,
executed, and the loop.

    python plot_ask5.py [r10a1s6i4]
"""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, FIGS = os.path.join(HERE, "out"), os.path.join(HERE, "figs")
SURF, INK, INK2, MUTED, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e4df"
BLUE, ORANGE, BLUE_L = "#2a78d6", "#eb6834", "#b9d2f1"
SHOW = (0, 20, 40, 50)          # replans shown in the strip, then the stop
W, HFIG = 14.0, 10.6            # inches
PW = 0.182; PH = PW * W / HFIG  # a 256x256 panel, as figure fractions
X0 = 0.045; GAP = (0.985 - X0 - 5 * PW) / 4


def px(p):
    """robosuite (row, col) on the raw render -> (x, y) on the image the policy sees (raw rotated 180 deg)."""
    p = np.asarray(p, float)
    return 255 - p[..., 1], p[..., 0]


def panel(fig, i, top):
    return fig.add_axes([X0 + i * (PW + GAP), top - PH, PW, PH])


def image(ax, img):
    ax.imshow(img[::-1, ::-1]); ax.set_xlim(0, 255); ax.set_ylim(255, 0); ax.set_axis_off()


SCENE = json.load(open(os.path.join(OUT, "scene_px.json")))


def labels(ax, d, t):
    for k in ("stove", "plate"):
        x, y = px(SCENE["flat_stove_1" if k == "stove" else "plate_1"]["geom_rc"])
        ax.text(x, y + 22, k, color=SURF, fontsize=7.5, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.15", fc="#00000066", ec="none"))


def flips(hold):
    labs = {s: [l for l, _ in hold[s]] for s in ("plus", "minus")}
    return sum(("stove" in a) != ("stove" in b) for a, b in zip(labs["plus"], labs["minus"])), len(labs["plus"])


def main():
    plt.rcParams.update({"figure.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "font.size": 9,
                         "axes.titlesize": 8.6, "axes.titleweight": "bold", "axes.edgecolor": LINE, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8})
    tag = sys.argv[1] if len(sys.argv) > 1 else "r10a1s6i4"
    d = np.load(os.path.join(OUT, f"movebowl_fig_{tag}.npz"))
    J = json.load(open(os.path.join(OUT, f"movebowl_fig_{tag}.json")))
    pm = os.path.join(OUT, "movebowl_fig_r20a3.json")
    miss = json.load(open(pm)) if tag != "r20a3" and os.path.exists(pm) else None
    rows = {r["t"]: r for r in J["rows"]}
    stop, fr, trig, R = J["stop"], J["frozen"], J["trig"], J["refresh"]
    t_fire = max(r["t"] for r in J["rows"] if r["fire"])
    test_s = rows[t_fire].get("t_test_s") or 0.31
    full_s = fr["t_full_s"]; t_ask = stop + int(np.ceil(full_s * 20))
    dest = {}
    for side in ("plus", "minus"):
        labs = [l for l, _ in fr["hold"][side]]
        n_st = sum("stove" in l for l in labs)
        dest[side] = ("stove", n_st, len(labs)) if n_st * 2 >= len(labs) else ("plate", len(labs) - n_st, len(labs))
    col = {s: (BLUE if dest[s][0] == "stove" else ORANGE) for s in dest}
    pv = os.path.join(OUT, "vlm_assign.json")
    vlm = json.load(open(pv)).get(str(stop)) if os.path.exists(pv) else None
    name = {s: (vlm[s][0] if vlm else dest[s][0]) for s in dest}          # words from the VLM, not the simulator
    if dest["plus"][0] == dest["minus"][0]:
        col["minus"] = ORANGE

    fig = plt.figure(figsize=(W, HFIG))
    fig.suptitle("Stop and ask, replayed step by step:  pi0.5, “move the bowl”, LIBERO-goal state 2, seed 0",
                 x=X0, ha="left", fontsize=12.5, fontweight="bold", y=0.982)

    # ---- A: the operator's screen ------------------------------------------------------------------------
    fig.text(X0, 0.945, "A   The operator's screen at four replans and at the stop.  Solid: the chunk's gripper path with the prompt embedding "
             "pushed 2.5% to either side of the ridge; dashed: the executed chunk." + ("  Names at the stop: Qwen3-VL-4B boxes + chunk direction." if vlm else ""),
             fontsize=8.8, color=INK2)
    heads = {0: "start: one path, nothing to ask", 20: "reaching: still one path", 40: "grasping: the paths begin to part",
             50: "bowl lifted: two paths. The ridge", stop: f"STOPPED at t {stop}: the question"}
    topA = 0.900
    for i, t in enumerate(SHOW + (stop,)):
        ax = panel(fig, i, topA); image(ax, d[f"img_{t}"])
        jump = float(np.linalg.norm(d[f"path_plus_{t}"] - d[f"path_minus_{t}"])) * 100
        if t == stop:
            for side in ("plus", "minus"):
                x, y = px(d[f"pix_{side}_{t}"])
                ax.plot(x, y, color=col[side], lw=2.6, solid_capstyle="round")
                ax.plot(x[-1], y[-1], "o", color=col[side], ms=9, mec=SURF, mew=1.2, zorder=5)
                st = dest[side][0] == "stove"
                ax.text(x[-1] + (-14 if st else 10), y[-1] - (24 if st else 12), f"{name[side]}?", color=col[side], fontsize=10, fontweight="bold",
                        ha="right" if st else "left", va="bottom", bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=0.9))
            ax.set_axis_on(); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor(INK); sp.set_linewidth(1.6)
        else:
            for side in ("plus", "minus"):
                x, y = px(d[f"pix_{side}_{t}"])
                ax.plot(x, y, color=INK2, lw=2.0, solid_capstyle="round", alpha=0.95)
                ax.plot(x[-1], y[-1], "o", color=INK2, ms=7, mec=SURF, mew=1.0, zorder=5)
            x, y = px(d[f"pix_exec_{t}"]); ax.plot(x, y, color=SURF, lw=1.2, ls=(0, (2, 2)), alpha=0.9)
        x0, y0 = px(d[f"objpix_eef_{t}"]); ax.plot(x0, y0, "o", color=SURF, ms=5, mec=INK, mew=0.8, zorder=6)
        labels(ax, d, t)
        s1 = fr["sigma1"] if t == stop else rows[t]["sigma1"]
        ax.set_title(f"t {t}  ·  σ₁ {s1:.1f}  ·  sides {jump:.0f} cm apart\n{heads[t]}", loc="left", pad=4)

    # ---- B: the trigger signal, the step-by-step window ---------------------------------------------------
    topB, hB, hS, hL = 0.596, 0.178, 0.020, 0.022
    fig.text(X0, topB + 0.028, "B   The trigger: the jump between the two sides of the ridge along the newest direction the background has delivered "
             f"(one every {R} steps = {R / 20:.1f} s), against the step-by-step window.", fontsize=8.8, color=INK2)
    ax = fig.add_axes([X0, topB - hB, 0.985 - X0, hB])
    strip = fig.add_axes([X0, topB - hB - hS - 0.004, 0.985 - X0, hS], sharex=ax)
    lane = fig.add_axes([X0, topB - hB - hS - hL - 0.010, 0.985 - X0, hL], sharex=ax)
    ts = sorted(rows)
    S = json.load(open(os.path.join(OUT, "movebowl_scan.json")))["rows"]
    sc_t = [r["t"] for r in S]
    fresh_t = [t for t in ts if t < sc_t[0]] + sc_t
    fresh = [rows[t]["jumps"][str(t)][0] for t in ts if t < sc_t[0]] + [r["jump"]["fresh4"] for r in S]
    asy_t = [t for t in ts if rows[t]["jump_async"] is not None]
    asy = [rows[t]["jump_async"][0] for t in asy_t]
    ymax = max(fresh + asy) * 1.25
    ax.plot(fresh_t, fresh, color=BLUE_L, lw=1.8, marker="o", ms=3.5, mec=SURF, mew=0.6, zorder=3,
            label="along a fresh v₁ of the state at that step (every step from t 46; a synchronous test would report it 1.3 s later)")
    ax.plot(asy_t, asy, color=BLUE, lw=2, marker="o", ms=6, mec=SURF, mew=0.8, zorder=4,
            label=f"along the newest delivered v₁, at the replans (the trigger test, {test_s:.2f} s)")
    ax.axhline(trig, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(ts[0] + 0.5, trig + 3, f"trigger threshold {trig:.0f} cm", fontsize=8, color=INK2, va="bottom")
    # the window, step by step: holding the embedding on either side of the fresh v1 (2 noises each)
    def steer(r):
        h = r.get("hold")
        if not h:
            return None
        p, m = h["fresh4+"], h["fresh4-"]
        if all(a == b for a in p for b in m):
            return "stove" if "stove" in p[0] else "plate"
        return "split"
    st = {r["t"]: steer(r) for r in S}
    open_t = [t for t, v in st.items() if v == "split"]
    ax.add_patch(Rectangle((46, 0), open_t[-1] + 1 - 46, ymax, fc=BLUE, alpha=0.06, ec="none", zorder=0))
    ax.text(ts[0] + 0.5, ymax * 0.70, f"shaded: the ridge is present from t 46, and the held prompt decides stove vs plate at t {open_t[0]}–{open_t[-1]}\n"
            f"except a two-step flicker (strip below). From t {open_t[-1] + 1} the bowl goes to the stove whatever the prompt.",
            fontsize=8, color=INK2, va="top")
    x_end = max(sc_t[-1], stop) + 24
    ax.add_patch(Rectangle((stop, 0), 400, ymax, fc="#f1f0ec", ec="none", zorder=0))
    ax.text(x_end - 0.5, ymax * 0.60, f"robot holds position. Full test on the frozen state\n(fresh v₁, {fr.get('n_iter', 1)} power iterations; side plans; projection): {full_s:.1f} s",
            fontsize=8, color=INK2, ha="right", va="top")
    ax.text(x_end - 0.5, ymax * 0.985, f"question on screen at t {t_ask} ({full_s:.1f} s later) →", fontsize=8, color=INK, va="top", ha="right")
    for t, txt, c, y, ls, lw in ((t_fire, f"test at t {t_fire}\nfires", BLUE, 0.27, (0, (3, 2)), 1.1),
                                 (stop, f"result arrives,\nt {stop}: STOP", INK, 0.985, "-", 1.8)):
        ax.axvline(t, color=c, lw=lw, ls=ls, zorder=2)
        ax.text(t + 0.6, ymax * y, txt, fontsize=8, color=c, va="top", fontweight="bold" if t == stop else None)
    if miss is not None and miss.get("frozen"):
        tm, fm = miss["stop"], miss["frozen"]; mf, mn = flips(fm["hold"])
        ax.plot([tm], [fm["jump"]], "x", color=INK, ms=8, mew=1.6, zorder=6)
        ax.text(x_end - 0.5, ymax * 0.80, f"×: with one refresh per 1.0 s the first test to fire is t 60 and its result\nlands at t {tm}: σ₁ {fm['sigma1']:.1f}, sides {fm['jump']:.0f} cm apart, holds flip {mf}/{mn}. Too late.",
                fontsize=8, color=INK, va="top", ha="right")
    ax.set_xlim(ts[0] - 0.5, x_end); ax.set_ylim(0, ymax)
    ax.set_ylabel("jump |P(+) − P(−)| at 2.5%, cm"); ax.tick_params(labelbottom=False)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8, handlelength=2.2)
    # steerability strip
    for t, v in st.items():
        if v is None:
            continue
        if v == "split":
            strip.add_patch(Rectangle((t, 0), 0.5, 1, fc=BLUE, ec="none")); strip.add_patch(Rectangle((t + 0.5, 0), 0.5, 1, fc=ORANGE, ec="none"))
        else:
            strip.add_patch(Rectangle((t, 0), 1, 1, fc=BLUE if v == "stove" else ORANGE, ec="none", alpha=0.45))
    strip.set_ylim(0, 1); strip.set_yticks([]); strip.tick_params(labelbottom=False); strip.spines[["top", "right", "left", "bottom"]].set_visible(False)
    strip.text(ts[0] + 0.5, 0.5, "hold test at each step (2 noises per side):  ■■ open, a side each    ■ / ■ committed to stove / plate", fontsize=7, color=INK2, va="center", ha="left")
    # refresh lane
    for k0 in range(0, stop, R):
        lane.add_patch(Rectangle((k0, 0), R, 1, fc=LINE, ec=SURF, lw=1))
        lane.text(k0 + R / 2, 0.5, f"v₁ of t {k0}" if R >= 10 else "", fontsize=7, color=INK2, ha="center", va="center")
    lane.set_ylim(0, 1); lane.set_yticks([]); lane.spines[["top", "right", "left"]].set_visible(False)
    lane.set_xlabel("control step (20 Hz; replan every 10 steps = 0.5 s)", labelpad=2)
    lane.set_ylabel("v₁ refresh", fontsize=7, rotation=0, ha="right", va="center", labelpad=4)

    # ---- C: the answers, executed; the loop ------------------------------------------------------------------
    topC = 0.268
    fig.text(X0, topC + 0.047, "C   The two answers, executed: resume with the prompt embedding held on the chosen side.", fontsize=8.8, color=INK2)
    for i, side in enumerate(sorted(dest, key=lambda s: dest[s][0] != "stove")):
        ax = panel(fig, i, topC); k, n, m = dest[side]
        b = next(j for j in range(m) if (k in fr["hold"][side][j][0]))
        image(ax, d[f"final_{side}_{b}"])
        x, y = px(d[f"finalpix_{side}_{b}"]); ax.plot(x, y, color=col[side], lw=2.2, solid_capstyle="round", alpha=0.95)
        ax.plot(x[0], y[0], "o", color=SURF, ms=5, mec=INK, mew=0.8, zorder=6)
        te = fr["hold"][side][b][1]
        ax.set_title(f"answer “{k}”\n→ on the {k} at t {te}  ({n}/{m} noises)", loc="left", pad=4)
    ax = fig.add_axes([X0 + 2 * (PW + GAP) + 0.01, topC - PH, 0.985 - X0 - 2 * (PW + GAP) - 0.01, PH]); ax.set_axis_off()
    steps = [
        f"Background thread refreshes the ridge direction v₁ of the current state (one power iteration, ~1.0 s). Here delivered every {R} steps.",
        f"Every replan: the chunk (0.18 s) and the jump between the two sides of the newest v₁ ({test_s:.2f} s). Fires when the 2.5% jump ≥ {trig:.0f} cm.",
        f"Stop the robot when the result arrives (t {stop}). The ridge is a function of the state, so freezing the state freezes the question.",
f"On the frozen state: fresh v₁ ({fr.get('n_iter', 1)} power iterations), the two side plans, projected into the camera image ({full_s:.1f} s). Show the two paths, ask.",
        "Push the prompt embedding to the chosen side (or re-prompt) and resume.",
    ]
    ax.text(0.0, 1.0, "The loop", fontsize=9.5, fontweight="bold", va="top", transform=ax.transAxes)
    for i, s in enumerate(steps):
        ax.text(0.0, 0.84 - 0.165 * i, str(i + 1), fontsize=9, fontweight="bold", color=BLUE, va="top", transform=ax.transAxes)
        ax.text(0.035, 0.84 - 0.165 * i, s, fontsize=8.6, color=INK2, va="top", transform=ax.transAxes, wrap=True)
    os.makedirs(FIGS, exist_ok=True)
    name = "ask_pi05.png" if tag == "r10a1s6i4" else f"ask_pi05_{tag}.png"
    fig.savefig(os.path.join(FIGS, name), dpi=170)
    print("wrote", os.path.join(FIGS, name))


if __name__ == "__main__":
    main()
