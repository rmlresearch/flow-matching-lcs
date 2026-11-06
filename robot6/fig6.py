"""
Example closed-loop run of pi0.5 on PushT through the explicit flow map (robot5/pi05flow), with
seeded noise so the run is reproducible. At every replan the plan is also sampled for N_CLOUD other
noise seeds, to show how little the noise moves it.

    python fig6.py run [reset_seed] [noise_seed ...] # -> out/example_<reset>_<noise>.npz
    python fig6.py plot [reset_seed] [noise_seed]   # -> figs/example_<reset>_<noise>.png

Coordinates: world pixels 0..512, y down (as rendered); render = world * 680 / 512.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval6 import make_env, to_lerobot, TASK, OUT  # noqa: E402

N_EXEC, N_CLOUD, MAX_T, N_FRAMES = 8, 16, 300, 6
FIGS = os.path.join(HERE, "figs")


def polys(env, pose=None):
    """T polygons (2 rectangles) of the block, or of the block placed at pose = (x, y, angle)."""
    u = env.unwrapped
    b = u.block if pose is None else u.get_goal_pose_body(pose)
    return [np.array([b.local_to_world(v) for v in s.get_vertices()]) for s in u.block.shapes]


def run(reset_seed=1003, noise_seed=0, f=None):
    from probe6 import load
    f = f or load()
    env = make_env()
    obs, _ = env.reset(seed=reset_seed)
    rng = np.random.default_rng(noise_seed)
    Zc = np.random.default_rng(10_000 + noise_seed).standard_normal((N_CLOUD, f.H * 32))
    agent, block, cov, frames, plans, clouds, t_plan = [], [], [], [], [], [], []
    info = {}
    t = 0
    while t < MAX_T and not info.get("is_success", False):
        c = f.prefix([to_lerobot(obs)], [TASK])
        a = f.unnorm(f.flow_np(rng.standard_normal((1, f.H * 32)), c)[0])
        clouds.append(f.unnorm(f.flow_np(Zc, c, bs=N_CLOUD)))
        plans.append(a); t_plan.append(t)
        for k in range(N_EXEC):
            agent.append(np.array(env.unwrapped.agent.position)); block.append(polys(env))
            frames.append(env.unwrapped._render(visualize=True))
            obs, _, term, trunc, info = env.step(a[k])
            cov.append(info["coverage"]); t += 1
            if info.get("is_success", False):
                break
    agent.append(np.array(env.unwrapped.agent.position)); block.append(polys(env))
    frames.append(env.unwrapped._render(visualize=True))
    print(f"reset {reset_seed} noise {noise_seed}: success {info.get('is_success')} at step {t}, "
          f"max coverage {max(cov):.3f}")
    idx = np.unique(np.linspace(0, t, N_FRAMES).round().astype(int))
    np.savez_compressed(os.path.join(OUT, f"example_{reset_seed}_{noise_seed}.npz"),
                        agent=np.array(agent), block=np.array(block), cov=np.array(cov),
                        frames=np.array([frames[i] for i in idx]), frame_t=idx, plans=np.array(plans),
                        clouds=np.array(clouds), t_plan=np.array(t_plan), goal=np.array(polys(env, env.unwrapped.goal_pose)),
                        success=bool(info.get("is_success")))


def plot(reset_seed=1003, noise_seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Polygon
    SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e4e3df"
    CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#c4359b"]
    BLUES = ["#eaf2fc", "#c4dbf5", "#8fbcec", "#5a9be0", "#2a78d6", "#1d5aa6", "#123d73"]
    ramp = LinearSegmentedColormap.from_list("b", BLUES[1:])
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
    R = np.load(os.path.join(OUT, f"example_{reset_seed}_{noise_seed}.npz"))
    agent, cov, ft, tp = R["agent"], R["cov"], R["frame_t"], R["t_plan"]
    T = len(agent) - 1
    s = 680 / 512

    fig = plt.figure(figsize=(12, 6.6))
    gs = fig.add_gridspec(2, len(ft), height_ratios=[1, 1.35], hspace=0.32, wspace=0.06)
    for j, (img, t) in enumerate(zip(R["frames"], ft)):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(img)
        ax.plot(*(agent[:t + 1] * s).T, color=INK, lw=1, alpha=0.6)
        p = np.searchsorted(tp, t, side="right") - 1                 # plan being executed at t
        if t < T:
            for cl in R["clouds"][p]:
                ax.plot(*(cl * s).T, color=CAT[1], lw=0.6, alpha=0.35)
            ax.plot(*(R["plans"][p] * s).T, color=CAT[1], lw=2)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_title(f"step {t}" + (f" · coverage {cov[t - 1]:.2f}" if t else ""), fontsize=8.5, color=INK2)

    # workspace: block poses over time on the target, agent path coloured by time
    ax = fig.add_subplot(gs[1, :3])
    for g in R["goal"]:
        ax.add_patch(Polygon(g, closed=True, fc="#dff1e8", ec=CAT[2], lw=1))
    for i, t in enumerate(ft):
        col = ramp(i / max(len(ft) - 1, 1))
        for q in R["block"][t]:
            ax.add_patch(Polygon(q, closed=True, fc="none", ec=col, lw=1.6))
    pts = agent
    for i in range(T):
        ax.plot(*pts[i:i + 2].T, color=ramp(i / T), lw=2, solid_capstyle="round")
    ax.plot(*pts[0], "o", ms=8, mfc=SURF, mec=BLUES[2], mew=2)
    ax.plot(*pts[-1], "o", ms=8, mfc=BLUES[-1], mec=SURF, mew=2)
    ax.set_xlim(0, 512); ax.set_ylim(512, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_title("agent path; block at the frame times\n(light → dark = early → late; green = target)",
                 fontsize=8.5, color=INK2, loc="left")

    # coverage over time
    ax = fig.add_subplot(gs[1, 3:])
    ax.axhline(0.95, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(2, 0.955, "success threshold 0.95", color=MUTED, fontsize=8, va="bottom")
    for t in tp:
        ax.axvline(t, color=GRID, lw=0.6, zorder=0)
    ax.plot(np.arange(1, T + 1), cov, color=CAT[0], lw=2)
    ax.set_xlim(0, max(T, 1)); ax.set_ylim(0, 1.02)
    ax.set_xlabel("env step (10 Hz); grid = replans every 8 steps"); ax.set_ylabel("target coverage")
    ax.spines[["top", "right"]].set_visible(False)
    sp = np.sqrt((R["clouds"].std(1) ** 2).sum(-1)).max(-1)
    ax.set_title(f"target coverage\nplan spread across {N_CLOUD} noise seeds per replan: median {np.median(sp):.1f} px, "
                 f"max {sp.max():.1f} px", fontsize=8.5, color=INK2, loc="left")
    ax.set_ylabel("")

    fig.suptitle(f"pi0.5 on PushT, reset seed {reset_seed}, noise seed {noise_seed}: "
                 f"{'success' if R['success'] else 'no success'} at step {T}, max coverage {cov.max():.3f}\n"
                 f"frames: black = agent path so far, orange = plan being executed, faint orange = the plan under "
                 f"{N_CLOUD} other noise seeds, red cross = current target",
                 x=0.125, ha="left", fontsize=10, color=INK)
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, f"example_{reset_seed}_{noise_seed}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    if sys.argv[1] == "run" and len(sys.argv) > 4:          # run reset noise0 noise1 ...: one model load
        from probe6 import load
        f = load()
        for n in sys.argv[3:]:
            run(int(sys.argv[2]), int(n), f)
    else:
        args = [int(x) for x in sys.argv[2:4]]
        {"run": run, "plot": plot}[sys.argv[1]](*args)
