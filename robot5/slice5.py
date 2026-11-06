"""
pi0.5 version of robot3's noise-slice figure (robot3/figs/slice_clip.png), on LIBERO-goal.

robot3's first chunk commits, so its slices are coloured by the chunk's open-loop target. pi0.5's
first chunk does not (goal5: <= 1.3 cm spread, unimodal), so here each pixel is a full closed-loop
episode: the first chunk's noise is the grid point, the rest of the episode uses a fixed future
stream (30000 + FUT, as branch5 / boundary5), and the colour is the goal5 label.

Plane (shared by all panels): z = z0 + a e1 + b e2 through a typical seed z0 (branch5's seed
Z0_INDEX at t 0), e1 = top right singular vector of the executed-path Jacobian at z0 under
"open a drawer", e2 = a random unit vector orthogonal to e1. Scene: LIBERO-goal task 0's scene,
initial state 3, t 0 (a decision state of "open a drawer", branch5). |z| ~ 40.

    python slice5.py run [panel ...]     # -> out/slice_<panel>.json (resumable)
    python slice5.py plot                # -> figs/slice_pi05.png
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT, FIGS = os.path.join(HERE, "out"), os.path.join(HERE, "figs")
SUITE, TASK0, LAYOUT, INIT, FUT, Z0_INDEX = "libero_goal", 0, 0, 3, 0, 2
W = int(os.environ.get("N_WORKERS", 8))
N, N_EXEC = 17, 10
PANELS = {"middle": ("open the middle drawer of the cabinet", 8.0),
          "drawer": ("open a drawer", 8.0),
          "top": ("open the top drawer and put the bowl inside", 8.0),
          "drawer_zoom": ("open a drawer", 0.08)}
t00 = time.time()


def log(*a):
    print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)


def plane(f, c_drawer):
    """z0, e1, e2 (cached in out/slice_plane.npz)."""
    from pi05flow import H, D_ACT
    from order5 import phys_jac
    p = os.path.join(OUT, "slice_plane.npz")
    if os.path.exists(p):
        P = np.load(p)
        return P["z0"], P["e1"], P["e2"]
    A = H * D_ACT
    z0 = np.random.default_rng(20_000).standard_normal((8, A))[Z0_INDEX]
    _, J = f.jac(z0[None], c_drawer, steps=N_EXEC, bs=1, chunk=70)
    e1 = np.linalg.svd(phys_jac(f, J, N_EXEC)[0], full_matrices=False)[2][0]
    r = np.random.default_rng(50_000).standard_normal(A)
    e2 = r - (r @ e1) * e1; e2 /= np.linalg.norm(e2)
    P = phys_jac(f, J, N_EXEC)[0]
    gain = np.array([np.linalg.norm(P @ e1), np.linalg.norm(P @ e2)])   # cm of executed path per unit noise
    np.savez(p, z0=z0, e1=e1, e2=e2, gain=gain)
    return z0, e1, e2


def run(*panels):
    import pi05flow
    from vec5 import Vec
    from goal5 import label
    from order5 import phys_jac
    panels = panels or tuple(PANELS)
    f = pi05flow.load()
    v = Vec(W)
    idx = list(range(W))
    v.load(idx, [dict(task_id=TASK0, init_id=INIT, layout=LAYOUT, suite_name=SUITE)] * W)
    obs = v.reset(idx)
    v.save(idx, "s")
    img = {k: np.asarray(x.cpu() if hasattr(x, "cpu") else x)[0] for k, x in obs[0].items() if "image" in k}
    np.savez_compressed(os.path.join(OUT, "slice_scene.npz"), **{k.replace(".", "_"): x for k, x in img.items()})
    z0, e1, e2 = plane(f, f.prefix([obs[0]], ["open a drawer"]))
    for name in panels:
        text, half = PANELS[name]
        path = os.path.join(OUT, f"slice_{name}.json")
        R = json.load(open(path)) if os.path.exists(path) else dict(text=text, half=half, n=N, labels={}, ftle=None)
        g = np.linspace(-half, half, N)
        pts = [(i, j) for j in range(N) for i in range(N)]
        Z = np.stack([z0 + g[i] * e1 + g[j] * e2 for i, j in pts])
        c = f.prefix([obs[0]], [text])
        if R["ftle"] is None:
            ft = []
            for b in range(0, len(Z), 16):
                _, J = f.jac(Z[b:b + 16], c, steps=N_EXEC, bs=4, chunk=70)
                ft += np.log(np.linalg.svd(phys_jac(f, J, N_EXEC), compute_uv=False)[:, 0]).tolist()
            R["ftle"] = ft
            json.dump(R, open(path, "w"))
            log(f"{name}: FTLE median {np.median(ft):+.2f} [{min(ft):+.2f}, {max(ft):+.2f}]")
        todo = [k for k, (i, j) in enumerate(pts) if f"{i},{j}" not in R["labels"]]
        A = f.unnorm(f.flow_np(Z, c, bs=64))
        for b in range(0, len(todo), W):
            kb = todo[b:b + W]
            o = v.restore(idx[:len(kb)], "s")
            res = v.episodes(f, idx[:len(kb)], o, [text] * len(kb), [np.random.default_rng(30_000 + FUT) for _ in kb],
                             first=[A[k] for k in kb], stop="goal")
            for k, r in zip(kb, res):
                i, j = pts[k]
                R["labels"][f"{i},{j}"] = label(r)
            json.dump(R, open(path, "w"))
            if (b // W) % 9 == 8 or b + W >= len(todo):
                log(f"{name}: {len(R['labels'])}/{len(pts)}")
    v.close()


ORDER = ["middle drawer", "bowl->top drawer", "bottom drawer open", "top drawer open", "bowl->cabinet top", "none"]


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    from matplotlib.patches import Patch
    SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#c4359b", "#cfcdc8"]       # last: none (neutral)
    BLUES = ["#eaf2fc", "#c4dbf5", "#8fbcec", "#5a9be0", "#2a78d6", "#1d5aa6", "#123d73"]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
    names = [n for n in PANELS if os.path.exists(os.path.join(OUT, f"slice_{n}.json"))]
    Rs = {n: json.load(open(os.path.join(OUT, f"slice_{n}.json"))) for n in names}
    used = [l for l in ORDER if any(l in R["labels"].values() for R in Rs.values())]
    extra = sorted({l for R in Rs.values() for l in R["labels"].values()} - set(ORDER))
    order = used + extra
    cmap = ListedColormap([CAT[ORDER.index(l)] if l in ORDER else CAT[4] for l in order])
    fig, ax = plt.subplots(2, len(names) + 1, figsize=(3.6 * (len(names) + 1), 7.0),
                           gridspec_kw=dict(hspace=0.32, wspace=0.18))
    S = np.load(os.path.join(OUT, "slice_scene.npz"))
    k_img = [k for k in S.files if "image" in k and "wrist" not in k][0]
    im = S[k_img]
    im = np.transpose(im, (1, 2, 0)) if im.shape[0] == 3 else im
    ax[0, 0].imshow(np.clip(im, 0, 1)); ax[0, 0].set_axis_off()
    ax[0, 0].set_title("scene (initial state 3, t = 0)", fontsize=10, color=INK)
    ax[1, 0].set_axis_off()
    g1, g2 = np.load(os.path.join(OUT, "slice_plane.npz"))["gain"]
    ax[1, 0].text(0, 1.0, "policy: pi0.5 (LIBERO fine-tune)\n"
                  "plane: z0 + a·e1 + b·e2 through a seed\n(|z| ≈ 40); e1 = top singular vector of\n"
                  "the executed-path Jacobian under\n\"open a drawer\"; e2 random,\northogonal to e1\n"
                  f"executed eef path moves {g1:.2f} cm\nper unit a, {g2:.3f} cm per unit b\n\n"
                  "top: closed-loop outcome (first chunk\nfrom the grid point, then a fixed\nfuture noise stream)\n"
                  "bottom: single-chunk FTLE of the\nexecuted eef path (cm)",
                  va="top", fontsize=8.5, color=INK2, transform=ax[1, 0].transAxes)
    fts = np.concatenate([R["ftle"] for R in Rs.values()])
    vmin, vmax = np.floor(fts.min() * 2) / 2, max(np.ceil(fts.max() * 2) / 2, 0.5)
    for c, n in enumerate(names, 1):
        R = Rs[n]; half, N_ = R["half"], R["n"]
        L = np.full((N_, N_), np.nan)
        for key, l in R["labels"].items():
            i, j = map(int, key.split(","))
            L[j, i] = order.index(l)
        ext = [-half, half, -half, half]
        a = ax[0, c]
        a.imshow(L, origin="lower", extent=ext, cmap=cmap, vmin=-0.5, vmax=len(order) - 0.5, interpolation="nearest")
        vals = list(R["labels"].values())
        frac = ", ".join(f"{vals.count(l)}" for l in order if l in vals)
        a.set_title(f"\"{R['text']}\"" + ("\n(zoom ×100)" if half < 1 else ""), fontsize=9.5, color=INK)
        a.set_xlabel("a (along e1)"); a.set_ylabel("b (along e2)" if c == 1 else "")
        a.plot(0, 0, "+", color=INK, ms=8, mew=1.2)
        b = ax[1, c]
        F = np.array(R["ftle"]).reshape(N_, N_)
        h = b.imshow(F, origin="lower", extent=ext, cmap=LinearSegmentedColormap.from_list("b", BLUES),
                     vmin=vmin, vmax=vmax, interpolation="nearest")
        b.set_xlabel("a (along e1)"); b.set_ylabel("b (along e2)" if c == 1 else "")
        b.set_title(f"FTLE {np.median(F):+.2f} [{F.min():+.2f}, {F.max():+.2f}]", fontsize=8.5, color=INK2)
    cax = ax[1, -1].inset_axes([1.06, 0.0, 0.06, 1.0])
    fig.colorbar(h, cax=cax, label="FTLE")
    ax[0, -1].legend(handles=[Patch(color=cmap(k), label=l) for k, l in enumerate(order)], loc="upper left",
                     bbox_to_anchor=(1.03, 1.0), frameon=False, fontsize=8.5, title="closed-loop outcome")
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, "slice_pi05.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(*sys.argv[2:])
    elif sys.argv[1] == "plot":
        plot()
