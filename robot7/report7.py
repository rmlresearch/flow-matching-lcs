"""Outcome table of a live7 run (single or multi-episode): per episode the prompt, held side, seed, replans,
the tcp path of the measured states (cm, base frame: x forward, y left, z up), the gripper range and the
tool height range.    python report7.py runs/<timestamp>"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from so101flow import Kin, state_path_cm  # noqa: E402


def episode_rows(run):
    eps_file = os.path.join(run, "episodes.json")
    if os.path.exists(eps_file):
        eps = json.load(open(eps_file))["episodes"]
        return [(e, os.path.join(run, f"ep{e['ep']}")) for e in eps]
    r = json.load(open(os.path.join(run, "run.json")))
    return [(dict(ep=0, task=r["task"], side=(r.get("hold") or {}).get("side"), seed=r["args"].get("seed")), run)]


def main():
    run = sys.argv[1].rstrip("/")
    kin = Kin()
    ch = None
    try:                                                   # which bowl (two-bowl scene): nearest to the final tcp
        from modes7 import Choice, bowls_xy, BOWLS_PX
        if "--bowls-cm" in sys.argv:                       # x1 y1 x2 y2 in the base frame (e.g. from a modes7 screen's json)
            i = sys.argv.index("--bowls-cm")
            ch = Choice(np.array([float(v) for v in sys.argv[i + 1:i + 5]]).reshape(2, 2))
            print(f"bowls (given): left {np.round(ch.b[0], 1).tolist()}, right {np.round(ch.b[1], 1).tolist()} cm")
        else:                                              # BOWLS_PX unprojected at the first state (only right if that view is the one the pixels came from)
            from arm7 import Arm
            e0, d0 = episode_rows(run)[0]
            st0 = json.load(open(os.path.join(d0, "run.json")))["replans"][0]["state"]
            ch = Choice(bowls_xy(Arm(dry_run=True, bus=False, camera=False, quiet=True), st0, BOWLS_PX))
            print(f"bowls from the first state's view: left {np.round(ch.b[0], 1).tolist()}, right {np.round(ch.b[1], 1).tolist()} cm")
    except Exception as ex:
        print(f"(no bowl labels: {ex})")
    print(f"{'ep':>2} {'prompt':34s} {'side':4s} {'seed':>4} {'replans':>7} | {'start tcp (cm)':>22} -> {'final tcp (cm)':>22} | {'y range':>12} | {'gripper':>10} | {'tool z':>10}")
    rows = []
    for e, d in episode_rows(run):
        r = json.load(open(os.path.join(d, "run.json")))
        st = np.array([x["state"] for x in r["replans"]])
        if len(st) == 0:
            continue
        tcp = np.stack([state_path_cm(kin, s) for s in st])
        g = st[:, 5]
        tz = np.array([x["tool_z_cm"] for x in r["replans"]])
        side = e.get("side") or (r.get("hold") or {}).get("side") or "-"
        hs = r.get("hold") or {}
        rows.append(dict(ep=e["ep"], task=e["task"], side=side, seed=e.get("seed"), replans=len(st), start=tcp[0].tolist(),
                         final=tcp[-1].tolist(), y_min=float(tcp[:, 1].min()), y_max=float(tcp[:, 1].max()),
                         g_min=float(g.min()), g_max=float(g.max()), z_min=float(tz.min()), z_max=float(tz.max()),
                         sides_end_cm=hs.get("sides_end_cm"),
                         bowl=(["left", "right"][int(ch.label(tcp[-1]))] if ch is not None else None),
                         q_final=(float(ch.coord(tcp[-1])) if ch is not None else None)))
        print(f"{e['ep']:>2} {e['task'][:34]:34s} {side:4s} {str(e.get('seed')):>4} {len(st):>7} | {np.round(tcp[0], 1).tolist()!s:>22} -> "
              f"{np.round(tcp[-1], 1).tolist()!s:>22} | {tcp[:, 1].min():5.1f}..{tcp[:, 1].max():5.1f} | {g.min():4.0f}..{g.max():4.0f} | {tz.min():4.1f}..{tz.max():4.1f}"
              + (f" | {rows[-1]['bowl']} bowl (q {rows[-1]['q_final']:+.1f})" if ch is not None else ""))
        if hs.get("sides_end_cm"):
            print(f"   held side plans ended at + {np.round(hs['sides_end_cm'][0], 1).tolist()} / - {np.round(hs['sides_end_cm'][1], 1).tolist()} cm (alpha {hs['alpha']})")
    json.dump(rows, open(os.path.join(run, "report.json"), "w"), indent=1)
    if "--fig" in sys.argv:
        draw_paths(run, rows, ch, kin, sys.argv[sys.argv.index("--fig") + 1])


def draw_paths(run, rows, ch, kin, out):
    """Top view of every episode's measured tcp path (one line per episode, coloured by prompt), the bowls as circles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prompts = list(dict.fromkeys(r["task"] for r in rows))
    cols = ["#1f5fbf", "#e07b00", "#1aa7b5", "#d9b400", "#c0399f"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.set_facecolor("#fafafa"); ax.set_aspect("equal")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#bbbbbb")
    ax.tick_params(colors="#666666", labelsize=8)
    if ch is not None:
        for b, lab in zip(ch.b, ("left bowl", "right bowl")):
            ax.add_patch(plt.Circle((b[1], b[0]), 4.5, fill=False, ec="#999999", lw=1.4))
            ax.text(b[1], b[0] - 5.6, lab, ha="center", fontsize=7, color="#777777")
    for e, d in episode_rows(run):
        r = json.load(open(os.path.join(d, "run.json")))
        st = np.array([x["state"] for x in r["replans"]])
        if len(st) == 0:
            continue
        tcp = np.stack([state_path_cm(kin, s_) for s_ in st])
        c = cols[prompts.index(e["task"]) % len(cols)]
        ax.plot(tcp[:, 1], tcp[:, 0], color=c, lw=1.2, alpha=0.8, marker=".", ms=3)
        ax.plot(tcp[-1, 1], tcp[-1, 0], marker="o", ms=5, color=c)
    for p_, c in zip(prompts, cols):
        ax.plot([], [], color=c, lw=2, label=p_)
    ax.invert_xaxis()
    ax.set_xlabel("y (cm; the robot's left is left)", fontsize=8, color="#444444")
    ax.set_ylabel("x forward (cm)", fontsize=8, color="#444444")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_title(f"{os.path.basename(run)}: measured tcp path of every episode (dot = end)", fontsize=9, color="#444444")
    fig.tight_layout(); fig.savefig(out, dpi=150); print("->", out)


if __name__ == "__main__":
    main()
