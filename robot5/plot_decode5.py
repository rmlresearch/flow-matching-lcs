"""figs/decode_pi05.png: the candidate-free pipeline on the example case ("open a drawer", LIBERO-goal
init 3), one panel per step, all from cached outputs of scratch/decode*.py, scratch/wordscores.py and
pridge5.py check:

  1  branch the closed loop and cluster (out/decode5_*_T0.npy, decode5_*.json)
  2  score every dictionary word by the first-order change of the plan along the mode difference
     (out/decode_wordscores_*.json)
  3  verify the decoded prompts by rolling them out (out/decode5/6_*.json)
  4  why it works: the plan is a step function of the prompt embedding (out/log_pridge_check.txt)

    python plot_decode5.py
"""
import json, os, re
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, FIGS = os.path.join(HERE, "out"), os.path.join(HERE, "figs")
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e4df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
BLUES = ["#b9d3f3", "#7fb0e8", "#4a90dd", "#1e5fb3"]
TAG = "open_a_drawer_3"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 1.6})


def outcomes(events, horizon=130):
    """events: list (per branch) of [name, t] pairs -> counts of top / middle / neither."""
    c = np.zeros(3, int)
    for ev in events:
        names = [n for n, _ in ev]
        c[0 if any("top" in n for n in names) else 1 if any("middle" in n for n in names) else 2] += 1
    return c


def main():
    style()
    T = np.load(os.path.join(OUT, f"decode5_{TAG}_T0.npy")) * 100            # [16, 131, 3] cm
    d5 = json.load(open(os.path.join(OUT, f"decode5_{TAG}.json")))
    d6 = json.load(open(os.path.join(OUT, f"decode6_{TAG}.json")))
    W = json.load(open(os.path.join(OUT, f"decode_wordscores_{TAG}.json")))
    cl = np.array(d5["cl"])
    top_cluster = int(np.argmax([sum(any("top" in n for n, _ in d5["events"][i]) for i in np.where(cl == c)[0]) for c in range(2)]))
    lam = {}
    for line in open(os.path.join(OUT, "log_pridge_check.txt")):
        m = re.match(r"\[\s*\d+s\] init (\d)", line)
        if m:
            k = int(m.group(1))
        elif line.strip().startswith("lambda:"):
            lam[k] = np.array([float(x) for x in line.split()[1:]])
        elif "'open a [a] drawer' lambda" in line:
            lam[(k, "a")] = float(line.rsplit("lambda", 1)[1])
    ss = np.linspace(-0.5, 1.5, 21)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.1), gridspec_kw=dict(width_ratios=[1.05, 1.15, 1.05, 1.0], wspace=0.42))
    fig.subplots_adjust(left=0.045, right=0.985, top=0.775, bottom=0.21)

    # 1. branch ----------------------------------------------------------------------------------
    a = ax[0]
    for i in range(len(T)):
        col = BLUE if cl[i] == top_cluster else ORANGE
        a.plot(T[i, :61, 0], T[i, :61, 2], color=col, alpha=0.85, lw=1.4)
        a.plot(T[i, 60:, 0], T[i, 60:, 2], color=col, alpha=0.3, lw=1.0)
        a.plot(T[i, 60, 0], T[i, 60, 2], "o", color=col, ms=4, mec=SURF, mew=0.6)
    a.plot(0, 0, "o", color=INK, ms=5, zorder=5)
    a.annotate("t 0", (0, 0), xytext=(-18, 6), textcoords="offset points", color=INK2)
    a.annotate("t 60: cluster\non gripper position", (float(T[cl == top_cluster, 60, 0].mean()), float(T[cl == top_cluster, 60, 2].mean())),
               xytext=(-70, -70), textcoords="offset points", color=INK2, fontsize=8.5,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    a.text(T[cl == top_cluster, -1, 0].mean() + 4, T[cl == top_cluster, -1, 2].mean() + 1, "top drawer (8)", color=BLUE, ha="left", va="center", fontsize=8.5)
    a.text(T[cl != top_cluster, -1, 0].mean() + 1, T[cl != top_cluster, -1, 2].mean() - 3, "middle\ndrawer (8)", color=ORANGE, ha="left", va="top", fontsize=8.5)
    a.set_xlabel("gripper x (cm, from start)"); a.set_ylabel("gripper z (cm)")
    a.set_title("1  Detect: branch the closed loop\n16 rollouts from one state, fresh noise", loc="left")
    a.set_aspect("equal"); a.grid(color=GRID, lw=0.6)

    # 2. decode ----------------------------------------------------------------------------------
    a = ax[1]
    sc = np.array(W["words"]["a"]["scores"]); pcs = W["words"]["a"]["pieces"]
    bins = np.linspace(sc.min() - 2, sc.max() + 2, 70)
    a.hist(sc, bins=bins, color="#cfd6de", edgecolor=SURF, lw=0.3)
    a.set_yscale("log"); a.set_ylim(0.8, 1e5)
    for w, col, lab, yy in (("top", BLUE, "“open top drawer”", 1.5e4), ("middle", ORANGE, "“open middle drawer”", 1.5e4), ("the", MUTED, "“open the drawer”", 3e2)):
        i = pcs.index(w); x = sc[i]
        a.axvline(x, color=col, lw=1.4, ymax=0.9 if w != "the" else 0.5)
        a.text(x + 1.5, yy, lab, color=col, ha="left", va="center", fontsize=8.5)
    a.text(0.02, 0.60, f"{len(sc):,} dictionary\nwords put in\nthe “a” slot",
           transform=a.transAxes, va="top", fontsize=8.3, color=INK2)
    a.set_xlabel("score = ⟨∂⟨plan, Δ⟩/∂E, emb(word) − emb(“a”)⟩  (cm)\nΔ = mean plan-horizon path, cluster A − cluster B")
    a.set_ylabel("words")
    a.set_title("2  Decode: one gradient, score every word\nthe extremes are the words that select a mode", loc="left")
    a.grid(axis="y", color=GRID, lw=0.6)

    # 3. verify ----------------------------------------------------------------------------------
    a = ax[2]
    rows = [("“open a drawer”  (original)", outcomes(d5["events"]) / len(d5["events"])),
            ("“open the drawer”", outcomes(d6["open the drawer"]) / 8),
            ("“open the top drawer”", outcomes(d6["open the top drawer"]) / 8),
            ("“open the middle drawer”", outcomes(d6["open the middle drawer"]) / 8)]
    ys = np.arange(len(rows))[::-1]
    for y, (name, c) in zip(ys, rows):
        left = 0
        for v, col in zip(c, (BLUE, ORANGE, "#d3d1cc")):
            if v > 0:
                a.barh(y, v, left=left, color=col, height=0.55, edgecolor=SURF, lw=1.5)
                left += v
    a.set_yticks([])
    for y, (name, c) in zip(ys, rows):
        a.text(0.01, y + 0.33, name, fontsize=8.5, color=INK2, va="bottom")
    a.set_xlim(0, 1); a.set_xticks([0, 0.5, 1]); a.set_xticklabels(["0", "½", "all"])
    a.set_xlabel("fraction of rollouts (130 steps)")
    from matplotlib.patches import Patch
    a.legend([Patch(color=BLUE), Patch(color=ORANGE), Patch(color="#d3d1cc")], ["top drawer opened", "middle drawer opened", "neither yet"],
             loc="upper left", bbox_to_anchor=(-0.02, -0.17), ncol=3, fontsize=8, handlelength=1.0, columnspacing=1.0)
    a.set_ylim(-0.5, len(rows) - 0.2)
    a.spines["left"].set_visible(False)
    a.set_title("3  Verify by rollout, then ask\n“the top drawer, or the middle one?”", loc="left")
    a.grid(axis="x", color=GRID, lw=0.6)

    # 4. ridge in prompt space -------------------------------------------------------------------
    a = ax[3]
    for k in range(4):
        a.plot(ss, lam[k], color=BLUES[k], lw=1.8 if k == 3 else 1.3, label=f"state {k}")
        la = lam[(k, "a")]
        j = np.where((lam[k][:-1] - la) * (lam[k][1:] - la) <= 0)[0]
        if len(j):
            j = j[-1]; x = ss[j] + (la - lam[k][j]) / (lam[k][j + 1] - lam[k][j]) * (ss[j + 1] - ss[j])
            a.plot(x, la, "o", color=BLUES[k], ms=6 if k == 3 else 4.5, mec=SURF, mew=0.7, zorder=5)
    a.annotate("“open a drawer”\nsits on the ridge\nonly at state 3\n(the state that splits)", (0.47, lam[(3, "a")]),
               xytext=(0.72, 0.68), textcoords="data", fontsize=8.3, color=INK2, va="center",
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    a.set_xticks([0, 0.5, 1]); a.set_xticklabels(["emb(top)", "½", "emb(middle)"])
    a.set_yticks([0, 0.5, 1]); a.set_yticklabels(["top\nplan", "½", "middle\nplan"])
    a.set_xlabel("embedding placed in the adjective slot"); a.set_ylabel("plan position λ")
    a.legend(loc="lower left", fontsize=8, handlelength=1.4, labelspacing=0.2)
    a.set_title("4  Why: a ridge in prompt space\nthe plan steps, it does not ramp", loc="left")
    a.grid(color=GRID, lw=0.6)

    for i in range(3):
        x1 = ax[i].get_position().x1; x0 = ax[i + 1].get_position().x0
        fig.text((x1 + x0) / 2, 0.52, "→", ha="center", va="center", fontsize=18, color=MUTED)
    fig.suptitle("From a behaviour split to a clarifying question, with no candidate list:  pi0.5, “open a drawer”, LIBERO-goal state 3",
                 x=0.045, ha="left", fontsize=12, fontweight="bold", y=0.975)
    fig.text(0.045, 0.895, "The policy alone supplies the modes (branching), the direction that separates them (one gradient w.r.t. the prompt tokens), "
             "the words that move the plan along it (first-order scoring), and the check (rollouts).",
             fontsize=9.3, color=INK2)
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, "decode_pi05.png"), dpi=180)
    print("wrote", os.path.join(FIGS, "decode_pi05.png"))


if __name__ == "__main__":
    main()
