"""
Noise-space modes of the SO-101 generalist in the two-bowl scene: does an ambiguous prompt ("pick up
the bowl") give two plans (one per bowl) over the noise seeds, and a specific one ("the bowl on the
left") a single plan? Then the LCS between two seeds that chose differently (robot2's recipe on this
flow: label change along the segment, the stretching / FTLE peak, the margin along v1).

    python modes7.py screen --obs runs/20261106_234800:0 runs/20261106_233849/ep1:2,4 \
                            --prompts "pick up the bowl" "move to the bowl on the left" --seeds 48 --temps 1 2
    python modes7.py table out/modes_<ts>.json              # the prompt x observation table
    python modes7.py lcs --obs runs/...:n --prompt "pick up the bowl" --pair 3 17   # two seeds of different modes

An observation spec is `<run dir>:<replan numbers>` (frame_NNN.jpg + the measured state in run.json) or
`--frame path.jpg --state p l e f r g`. Bowls are located by unprojecting their pixel centres onto the
plane `rim_cm` above the desk at the camera pose of the *first* observation (`--bowls u v u v`).
Every seed's plan = the 50-step tcp path (cm, base frame); its mode = the bowl nearest to the path end,
and the "choice coordinate" is the path end projected on the bowl-to-bowl axis (0 = midpoint).
"""
import os, sys, json, time, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from so101flow import (H, D_ACT, A, get_model, Kin, observation, path_cm, prompt_slots, ridge_test, log, OUT,  # noqa: E402
                       state_path_cm)

RUNS = os.path.join(HERE, "runs")
AMBIG = ["pick up the bowl", "move the bowl", "pick the bowl", "pick up a bowl", "grab the bowl", "put the bowl aside"]
SPECIFIC = ["move to the bowl on the left", "move to the bowl on the right", "pick up the bowl on the left",
            "pick up the bowl on the right", "pick up the left bowl", "pick up the right bowl"]
BOWLS_PX = [385, 400, 655, 395]        # left / right bowl centres in the 1280x720 vantage frame of runs/20261106_234800


# ------------------------------------------------------------------------------------------ observations
def load_obs(spec):
    """'run_dir:n1,n2' -> [(name, rgb, state_deg)] from the run's frames and recorded states."""
    import cv2
    d, _, ns = spec.partition(":")
    d = d if os.path.isabs(d) else os.path.join(HERE, d) if not os.path.exists(d) else d
    run = json.load(open(os.path.join(d, "run.json")))
    recs = {r["n"]: r for r in run["replans"]}
    out = []
    for n in [int(x) for x in ns.split(",")] if ns else sorted(recs):
        rgb = cv2.cvtColor(cv2.imread(os.path.join(d, f"frame_{n:03d}.jpg")), cv2.COLOR_BGR2RGB)
        name = os.path.relpath(d, RUNS).replace(os.sep, "/") + f":{n}"
        if recs[n].get("pose") is not None:
            POSES[name] = recs[n]["pose"]
        out.append((name, rgb, np.array(recs[n]["state"], np.float32)))
    return out


POSES = {}


def bowls_xy(arm, state_deg, px, size=(1280, 720), rim_cm=3.5):
    """Pixel centres of the bowls -> (x, y) cm in the base frame on the plane rim_cm above the desk."""
    from so101.robot_model import CameraModel, ray_plane
    cam_p, cam_R = arm.arm.camera_pose(np.radians(np.asarray(state_deg)[:5]))
    cm = CameraModel(arm.arm.fovy, *size)
    out = []
    for u, v in np.asarray(px, float).reshape(-1, 2):
        p = ray_plane(cam_p, cam_R @ cm.ray(u, v), arm.table_z + rim_cm / 100.0)
        out.append((p[:2] * 100.0).tolist() if p is not None else [np.nan, np.nan])
    return np.array(out)


class Choice:
    """Mode label of a plan from the two bowl positions: nearest bowl, and the signed coordinate along the
    bowl axis (0 at the midpoint, + towards bowl 0 = the left one)."""

    def __init__(self, bowls_cm):
        self.b = np.asarray(bowls_cm, float)
        self.mid = self.b.mean(0)
        ax = self.b[0] - self.b[1]
        self.ax = ax / np.linalg.norm(ax)
        self.half = np.linalg.norm(ax) / 2

    def coord(self, end_xy):
        return (np.asarray(end_xy)[..., :2] - self.mid) @ self.ax

    def label(self, end_xy):
        d = np.linalg.norm(np.asarray(end_xy)[..., None, :2] - self.b, axis=-1)
        return d.argmin(-1)                                    # 0 = left bowl, 1 = right bowl


def shift_view(rgb, px):
    """Frame content moved `px` pixels to the right (camera turned left), edge pixels replicated: a synthetic
    observation standing in for a vantage pose whose camera is centred elsewhere (the 4:3 centre crop the
    policy sees is x in [160, 1120] of a 1280-wide frame, so |px| < 160 keeps the crop free of filled pixels)."""
    out = np.empty_like(rgb)
    if px >= 0:
        out[:, px:] = rgb[:, :rgb.shape[1] - px]; out[:, :px] = rgb[:, :1]
    else:
        out[:, :px] = rgb[:, -px:]; out[:, px:] = rgb[:, -1:]
    return out


def seeds_z(n, seed, temp=1.0):
    return temp * np.random.default_rng(seed).standard_normal((n, H * D_ACT))


def plans_for(f, kin, c, z, bs=48):
    """Seeds [N, 1600] -> tcp paths [N, H, 3] cm."""
    a = f.flow_np(z, c, bs=bs)
    return path_cm(f, kin, a)


def mode_stats(P, ch, step=-1):
    """Paths [N, H, 3] -> counts per bowl, choice coordinate spread, largest gap, spread of the path ends."""
    end = P[:, step]
    lab = ch.label(end); q = ch.coord(end)
    qs = np.sort(q)
    gaps = np.diff(qs)
    j = int(gaps.argmax()) if len(gaps) else 0
    return dict(n=int(len(P)), n_left=int((lab == 0).sum()), n_right=int((lab == 1).sum()),
                q_mean=float(q.mean()), q_std=float(q.std()), q_min=float(q.min()), q_max=float(q.max()),
                gap=float(gaps.max()) if len(gaps) else 0.0, gap_at=float((qs[j] + qs[j + 1]) / 2) if len(gaps) else 0.0,
                end_spread=float(np.linalg.norm(end - end.mean(0), axis=1).mean()),
                end_mean=end.mean(0).tolist(), label=lab.tolist(), q=q.tolist())


# ------------------------------------------------------------------------------------------ commands
def cmd_screen(args):
    from arm7 import Arm
    arm = Arm(dry_run=True, bus=False, camera=False, quiet=True)
    obs_list = []
    for s in args.obs:
        obs_list += load_obs(s)
    if args.frame:
        import cv2
        obs_list.append((os.path.basename(args.frame), cv2.cvtColor(cv2.imread(args.frame), cv2.COLOR_BGR2RGB), np.array(args.state, np.float32)))
    bowls = bowls_xy(arm, obs_list[0][2], args.bowls, rim_cm=args.rim_cm)
    ch = Choice(bowls)
    log(f"bowls (cm, base frame): left {np.round(bowls[0], 1).tolist()}, right {np.round(bowls[1], 1).tolist()}, "
        f"{2 * ch.half:.1f} cm apart; {len(obs_list)} observations, {len(args.prompts)} prompts, {args.seeds} seeds x temps {args.temps}")
    if args.shift_px:
        obs_list = [(f"{n}+shift{args.shift_px:+d}px", shift_view(rgb, args.shift_px), st + np.array([args.shift_pan_deg, 0, 0, 0, 0, 0], np.float32))
                    for n, rgb, st in obs_list]
        log(f"SYNTHETIC view: frame content shifted {args.shift_px:+d} px, pan joint {args.shift_pan_deg:+.1f} deg")
    f = get_model(); kin = Kin()
    ts = time.strftime("%Y%m%d_%H%M%S")
    res = dict(bowls=bowls.tolist(), bowls_px=list(args.bowls), rim_cm=args.rim_cm, seeds=args.seeds, temps=args.temps,
               prompts=list(args.prompts), obs=[], cells=[])
    paths = {}
    for name, rgb, state in obs_list:
        tcp = state_path_cm(kin, state)
        res["obs"].append(dict(name=name, state=state.tolist(), tcp=tcp.tolist(), tool_z=float(tcp[2] - arm.table_z * 100),
                               pose=POSES.get(name)))
        o = observation(rgb, state)
        for text in args.prompts:
            c = f.prefix([o], [text])
            for temp in args.temps:
                z = seeds_z(args.seeds, args.seed, temp)
                t0 = time.time()
                P = plans_for(f, kin, c, z, bs=args.bs)
                st = mode_stats(P, ch)
                st40 = mode_stats(P, ch, step=39)
                cell = dict(obs=name, prompt=text, temp=temp, t=time.time() - t0, **st,
                            n_left_40=st40["n_left"], n_right_40=st40["n_right"], q_std_40=st40["q_std"], gap_40=st40["gap"])
                if args.ridge and temp == args.temps[0]:
                    zp = seeds_z(2, 1000 + args.seed)
                    r, _ = ridge_test(f, kin, o, text, zp, n_iter=args.iters, alphas=[0.025, 0.05, 0.1], with_random=False)
                    ep, em = np.array(r["path_plus"])[-1], np.array(r["path_minus"])[-1]
                    cell["ridge"] = dict(sigma1=r["sigma1"], d_v=r["d_v"], side=r["side"], end_plus=ep.tolist(), end_minus=em.tolist(),
                                         q_plus=float(ch.coord(ep)), q_minus=float(ch.coord(em)),
                                         lab_plus=int(ch.label(ep)), lab_minus=int(ch.label(em)),
                                         sep_xy=float(np.linalg.norm(ep[:2] - em[:2])))
                res["cells"].append(cell)
                paths[f"{len(res['cells']) - 1}"] = P.astype(np.float32)
                extra = ""
                if "ridge" in cell:
                    rr = cell["ridge"]
                    extra = (f" | prompt ridge: sigma1 {rr['sigma1']:.1f}, sides end q {rr['q_plus']:+.1f} / {rr['q_minus']:+.1f} "
                             f"(bowls {rr['lab_plus']} / {rr['lab_minus']}), {rr['sep_xy']:.1f} cm apart")
                log(f"{name} | {text!r:38s} | T {temp:.1f}: left {st['n_left']:2d} right {st['n_right']:2d} | "
                    f"q {st['q_mean']:+5.1f} +- {st['q_std']:4.1f} [{st['q_min']:+5.1f}, {st['q_max']:+5.1f}] gap {st['gap']:4.1f} | "
                    f"end spread {st['end_spread']:.2f} cm ({cell['t']:.0f}s){extra}")
            del c
        json.dump(res, open(os.path.join(OUT, f"modes_{ts}.json"), "w"), indent=1)
        np.savez_compressed(os.path.join(OUT, f"modes_{ts}.npz"), **paths)
    log(f"-> out/modes_{ts}.json / .npz")
    print_table(res)


def print_table(res):
    obs = [o["name"] for o in res["obs"]]
    print("\nmode counts (left / right bowl) of the path end over the seeds, per prompt x observation; T = noise scale")
    print(f"{'prompt':40s} T   " + "  ".join(f"{o[-12:]:>14s}" for o in obs))
    for text in res["prompts"]:
        for temp in res["temps"]:
            row = []
            for o in obs:
                c = next((c for c in res["cells"] if c["obs"] == o and c["prompt"] == text and c["temp"] == temp), None)
                row.append(f"{c['n_left']:2d}/{c['n_right']:2d} sd{c['q_std']:4.1f} g{c['gap']:3.1f}" if c else " " * 14)
            print(f"{text:40s} {temp:3.1f} " + "  ".join(f"{r:>14s}" for r in row))
    if any("ridge" in c for c in res["cells"]):
        print("\nprompt-space ridge: bowl of the + / - side path end (0 left, 1 right), their separation (cm), sigma1")
        print(f"{'prompt':40s}     " + "  ".join(f"{o[-12:]:>14s}" for o in obs))
        for text in res["prompts"]:
            row = []
            for o in obs:
                c = next((c for c in res["cells"] if c["obs"] == o and c["prompt"] == text and "ridge" in c), None)
                row.append(f"{c['ridge']['lab_plus']}/{c['ridge']['lab_minus']} {c['ridge']['sep_xy']:4.1f}cm s{c['ridge']['sigma1']:3.1f}" if c else " " * 14)
            print(f"{text:40s}     " + "  ".join(f"{r:>14s}" for r in row))


def cmd_table(args):
    print_table(json.load(open(args.file)))


def cmd_lcs(args):
    """Between two seeds of different modes: the label along the straight segment, the plan stretching
    along it (label-free ridge), FTLE at points of the segment, and the margin along v1 from each seed."""
    from arm7 import Arm
    arm = Arm(dry_run=True, bus=False, camera=False, quiet=True)
    (name, rgb, state), = load_obs(args.obs[0])[:1]
    ref_state = load_obs(args.bowl_ref)[0][2] if args.bowl_ref else state      # the view the --bowls pixels belong to
    bowls = bowls_xy(arm, ref_state, args.bowls, rim_cm=args.rim_cm)
    log(f"bowls (cm): left {np.round(bowls[0], 1).tolist()}, right {np.round(bowls[1], 1).tolist()}")
    ch = Choice(bowls)
    f = get_model(); kin = Kin()
    o = observation(rgb, state)
    c = f.prefix([o], [args.prompt])
    z = seeds_z(args.seeds, args.seed, args.temps[0])
    za, zb = z[args.pair[0]], z[args.pair[1]]
    lam = np.linspace(0, 1, args.n_lam)
    zz = (1 - lam[:, None]) * za + lam[:, None] * zb
    P = plans_for(f, kin, c, zz)
    lab = ch.label(P[:, -1]); q = ch.coord(P[:, -1])
    log(f"{name} {args.prompt!r}: labels along the segment {''.join(str(int(l)) for l in lab)}; q from {q[0]:+.1f} to {q[-1]:+.1f}")
    flat = P.reshape(len(lam), -1)
    stretch = np.linalg.norm(np.diff(flat, axis=0), axis=1) / np.diff(lam)        # cm per unit lambda
    k = int(stretch.argmax())
    res = dict(obs=name, prompt=args.prompt, pair=args.pair, lam=lam.tolist(), label=lab.tolist(), q=q.tolist(),
               stretch=stretch.tolist(), lam_ridge=float((lam[k] + lam[k + 1]) / 2), bowls=bowls.tolist())
    # the separatrix by bisection on the label
    flips = np.nonzero(np.diff(lab))[0]
    if len(flips):
        lo, hi = lam[flips[0]], lam[flips[0] + 1]
        for _ in range(args.bisect):
            m = (lo + hi) / 2
            Pm = plans_for(f, kin, c, ((1 - m) * za + m * zb)[None])
            if ch.label(Pm[0, -1]) == lab[0]:
                lo = m
            else:
                hi = m
        res["lam_sep"] = float((lo + hi) / 2)
        log(f"separatrix at lambda {res['lam_sep']:.4f} (bisection), label-free ridge (max stretch {stretch[k]:.0f} cm/lambda) at {res['lam_ridge']:.3f}")
    else:
        log("no label change along the segment")
    # FTLE along the segment
    pts = np.unique(np.clip(np.concatenate([np.linspace(0, 1, args.n_ftle), [res.get("lam_sep", 0.5)]]), 0, 1))
    t0 = time.time()
    _, J = f.jac(((1 - pts[:, None]) * za + pts[:, None] * zb), c, steps=args.steps, bs=1)
    sv = np.linalg.svd(J, compute_uv=False)
    res["lam_ftle"] = pts.tolist(); res["ftle"] = np.log(sv[:, 0]).tolist(); res["sv1"] = sv[:, 0].tolist()
    log(f"FTLE along the segment ({time.time() - t0:.0f}s): " + ", ".join(f"{l:.3f}: {x:+.2f}" for l, x in zip(pts, res['ftle'])))
    # margins along v1 from each seed
    res["from_seed"] = []
    for zi, nm in ((za, "a"), (zb, "b")):
        _, Ji = f.jac(zi[None], c, steps=args.steps, bs=1)
        _, s, Vt = np.linalg.svd(Ji[0], full_matrices=False)
        v1 = Vt[0]
        rd = np.random.default_rng(7).standard_normal(v1.shape); rd /= np.linalg.norm(rd)
        S = np.round(np.arange(-40, 41) * 0.1, 10)
        out = dict(seed=nm, ftle=float(np.log(s[0])))
        for dn, d in (("v1", v1), ("random", rd)):
            Pz = plans_for(f, kin, c, zi[None] + S[:, None] * d[None])
            lz = ch.label(Pz[:, -1]); qz = ch.coord(Pz[:, -1])
            l0 = lz[40]
            cr = [S[i] for i in range(len(S)) if lz[i] != l0]
            out[dn] = dict(labels="".join(str(int(l)) for l in lz), q=qz.tolist(),
                           margin_plus=float(min([s for s in cr if s > 0], default=np.inf)),
                           margin_minus=float(max([s for s in cr if s < 0], default=-np.inf)),
                           max_jump=float(np.linalg.norm(np.diff(Pz.reshape(len(S), -1), axis=0), axis=1).max()))
            log(f"seed {nm} along {dn:6s}: labels {out[dn]['labels']} margins {out[dn]['margin_minus']:+.1f} / {out[dn]['margin_plus']:+.1f} "
                f"noise units, max plan jump per 0.1 {out[dn]['max_jump']:.1f} cm")
        res["from_seed"].append(out)
    slug = args.prompt.replace(" ", "_")[:30]
    json.dump(res, open(os.path.join(OUT, f"lcs_{slug}_{args.pair[0]}_{args.pair[1]}.json"), "w"), indent=1)
    log(f"-> out/lcs_{slug}_{args.pair[0]}_{args.pair[1]}.json")


def cmd_fig(args):
    """Figure of a screen: per observation (columns) and prompt (rows) the choice coordinate of every seed's
    path end (strip), the two bowls marked, and the two sides of the prompt-space ridge (+ / -)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = json.load(open(args.file))
    obs = res["obs"]; prompts = res["prompts"]; temps = res["temps"]
    ch = Choice(res["bowls"])
    fig, axs = plt.subplots(1, len(obs), figsize=(2.6 * len(obs) + 1.5, 0.42 * len(prompts) + 1.2), sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    blues = ["#1f5fbf", "#8fb3e8", "#c9d9f3"]
    rng = np.random.default_rng(0)
    for ax, o in zip(axs, obs):
        ax.set_facecolor("#fafafa")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#bbbbbb")
        ax.tick_params(colors="#666666", labelsize=8)
        ax.axvline(0, color="#dddddd", lw=1, zorder=0)
        for b, lab in zip((ch.half, -ch.half), ("left bowl", "right bowl")):
            ax.axvline(b, color="#999999", lw=1, ls=":", zorder=0)
        ax.axhline(len(AMBIG) - 0.5, color="#cccccc", lw=0.8)
        for i, text in enumerate(prompts):
            for k, temp in enumerate(temps):
                c = next((c for c in res["cells"] if c["obs"] == o["name"] and c["prompt"] == text and c["temp"] == temp), None)
                if c is None:
                    continue
                q = np.array(c["q"])
                ax.scatter(q, i + (k - (len(temps) - 1) / 2) * 0.28 + rng.normal(0, 0.05, len(q)), s=7, color=blues[min(k, 2)],
                           alpha=0.8, lw=0, zorder=3 - k)
                if k == 0 and "ridge" in c:
                    ax.plot([c["ridge"]["q_plus"]], [i - 0.3], marker="^", ms=5, color="#e07b00", lw=0, zorder=4)
                    ax.plot([c["ridge"]["q_minus"]], [i - 0.3], marker="v", ms=5, color="#e07b00", lw=0, zorder=4)
        ttl = o["name"].split("/")[-1]
        if o.get("pose") is not None:
            ttl = f"pan {o['pose'][0]:.0f}"
        ax.set_title(f"{ttl}\ntool {o['tool_z']:.0f} cm up", fontsize=8, color="#444444")
        ax.set_xlim(-max(10, 1.3 * ch.half), max(10, 1.3 * ch.half))
    axs[0].set_yticks(range(len(prompts))); axs[0].set_yticklabels(prompts, fontsize=8)
    axs[0].invert_yaxis()
    fig.supxlabel("path end along the bowl axis (cm; 0 = midpoint, + = left bowl)", fontsize=9, color="#444444")
    fig.suptitle(f"{res['seeds']} seeds per cell, noise scale {' / '.join(str(t) for t in temps)} (dark / light)"
                 + ("; orange: sides of the prompt ridge" if any("ridge" in c for c in res["cells"]) else ""),
                 fontsize=9, color="#444444")
    fig.tight_layout()
    out = args.out or os.path.join(HERE, "figs", os.path.basename(args.file).replace(".json", ".png"))
    fig.savefig(out, dpi=150)
    print("->", out)


def cmd_views(args):
    """Move the real arm through --poses (our deg + gripper, 6 values each, ';'-separated), capture a settled wrist
    frame and the measured state at each, and save them as a run (runs/<ts>_views/frame_NNN.jpg + run.json) that
    `screen --obs <dir>` can use. Returns to the first pose at the end; torque stays on."""
    import cv2
    from arm7 import Arm
    poses = [np.array([float(x) for x in p.split()], np.float32) for p in args.poses]
    assert all(len(p) == 6 for p in poses), "each pose = pan lift elbow flex roll gripper"
    ts = time.strftime("%Y%m%d_%H%M%S")
    d = os.path.join(RUNS, f"{ts}_views"); os.makedirs(d, exist_ok=True)
    arm = Arm(dry_run=args.dry_run, bus=True, camera=True, quiet=False)
    for p in poses:
        if arm.tool_z(p) < 0.06:
            raise SystemExit(f"pose {p.tolist()} puts the tool {arm.tool_z(p) * 100:.1f} cm above the desk (< 6 cm): refusing")
    arm.enable()
    recs = []
    for n, p in enumerate(poses):
        arm.goto(p, seconds=args.seconds)
        time.sleep(0.8)
        for _ in range(10):
            rgb = arm.frame(); time.sleep(0.03)
        st = arm.state_deg() if not args.dry_run else p
        tz = arm.tool_z(st) * 100
        cv2.imwrite(os.path.join(d, f"frame_{n:03d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        recs.append(dict(n=n, t=0.0, state=np.asarray(st, float).tolist(), pose=p.tolist(), tool_z_cm=round(float(tz), 1),
                         brightness=float(np.asarray(rgb).mean())))
        log(f"view {n}: pose {np.round(p, 1).tolist()} -> state {np.round(st, 1).tolist()}, tool {tz:.1f} cm up, brightness {recs[-1]['brightness']:.0f}")
    arm.goto(poses[0], seconds=args.seconds)
    json.dump(dict(task="views", args=vars(args), hold=None, replans=recs), open(os.path.join(d, "run.json"), "w"), indent=1)
    arm.close()
    log(f"-> {d}")
    print(os.path.relpath(d, RUNS))


def cmd_fan(args):
    """Top view of every seed's 50-step tcp path for chosen (observation, prompt) cells of a screen: the fan of
    plans between the two bowls. --cells 'obs_suffix|prompt' ... (obs matched by suffix, temp = first temp)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = json.load(open(args.file)); P = np.load(args.file.replace(".json", ".npz"))
    ch = Choice(res["bowls"])
    sel = []
    for spec in args.cells:
        o, _, pr = spec.partition("|")
        for i, c in enumerate(res["cells"]):
            if c["obs"].endswith(o) and c["prompt"] == pr and c["temp"] == (args.temps[0] if args.temps else res["temps"][0]):
                sel.append((i, c))
    n = len(sel)
    fig, axs = plt.subplots(1, n, figsize=(3.0 * n + 0.5, 3.6), sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (i, c) in zip(axs, sel):
        paths = P[str(i)]                                                 # [N, H, 3] cm
        ax.set_facecolor("#fafafa"); ax.set_aspect("equal")
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"): ax.spines[s_].set_color("#bbbbbb")
        ax.tick_params(colors="#666666", labelsize=7)
        for b, col in zip(ch.b, ("#1f5fbf", "#e07b00")):
            ax.add_patch(plt.Circle((b[1], b[0]), 4.5, fill=False, ec=col, lw=1.6, ls="-"))
        lab = np.array(c["label"])
        for k in range(len(paths)):
            col = "#1f5fbf" if lab[k] == 0 else "#e07b00"
            ax.plot(paths[k, :, 1], paths[k, :, 0], color=col, lw=0.7, alpha=0.45)
            ax.plot(paths[k, -1, 1], paths[k, -1, 0], marker="o", ms=2.5, color=col, lw=0, alpha=0.9)
        o = next(o for o in res["obs"] if o["name"] == c["obs"])
        ax.plot(o["tcp"][1], o["tcp"][0], marker="x", ms=7, color="#444444", mew=1.5)
        ttl = f"pan {o['pose'][0]:.0f}, tool {o['tool_z']:.0f} cm" if o.get("pose") else c["obs"].split("/")[-1]
        ax.set_title(f"{c['prompt']}\n{ttl}: {c['n_left']} left / {c['n_right']} right", fontsize=8, color="#444444")
    axs[0].invert_xaxis()                                                  # +y (robot's left) drawn to the left (shared axis: once)
    axs[0].set_ylabel("x forward (cm)", fontsize=8, color="#444444")
    fig.supxlabel("y (cm, robot's left is left)", fontsize=8, color="#444444")
    fig.suptitle(f"{res['seeds']} noise seeds at scale {args.temps[0] if args.temps else res['temps'][0]}: the plan (50-step tcp path) of every seed, "
                 f"coloured by the bowl nearest its end; circles = the bowls, x = the tcp", fontsize=8.5, color="#444444")
    fig.tight_layout()
    out = args.out or os.path.join(HERE, "figs", os.path.basename(args.file).replace(".json", "_fan.png"))
    fig.savefig(out, dpi=150)
    print("->", out)


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["screen", "table", "lcs", "fig", "views", "fan"])
    ap.add_argument("file", nargs="?", help="table: the modes_*.json")
    ap.add_argument("--obs", nargs="*", default=["20261106_234800:0"], help="run_dir[:n,n,...] (relative to runs/)")
    ap.add_argument("--frame"); ap.add_argument("--state", type=float, nargs=6)
    ap.add_argument("--prompts", nargs="+", default=AMBIG + SPECIFIC)
    ap.add_argument("--prompt", default="pick up the bowl")
    ap.add_argument("--seeds", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temps", type=float, nargs="+", default=None)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--bowls", type=float, nargs=4, default=BOWLS_PX, metavar="PX")
    ap.add_argument("--rim-cm", type=float, default=3.5)
    ap.add_argument("--ridge", action="store_true", help="also the prompt-space ridge test per prompt (2 alphas)")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--pair", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--bowl-ref", help="lcs: observation spec whose frame the --bowls pixels were read from (default: the --obs itself)")
    ap.add_argument("--n-lam", type=int, default=21)
    ap.add_argument("--n-ftle", type=int, default=5)
    ap.add_argument("--bisect", type=int, default=10)
    ap.add_argument("--steps", type=int, default=H)
    ap.add_argument("--out")
    ap.add_argument("--cells", nargs="*", default=[], help="fan: 'obs_suffix|prompt' per panel")
    ap.add_argument("--poses", nargs="+", help="views: 'pan lift elbow flex roll gripper' per pose")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shift-px", type=int, default=0, help="synthetic view: move the frame content this many px right")
    ap.add_argument("--shift-pan-deg", type=float, default=0.0, help="synthetic view: add this to the pan joint of the state")
    return ap


if __name__ == "__main__":
    args = parser().parse_args()
    args.obs = [s if os.path.isabs(s) or s.startswith("runs") else os.path.join("runs", s) for s in args.obs]
    if args.temps is None and args.cmd in ("screen", "lcs"):
        args.temps = [1.0]
    {"screen": cmd_screen, "table": cmd_table, "lcs": cmd_lcs, "fig": cmd_fig, "views": cmd_views, "fan": cmd_fan}[args.cmd](args)
