"""
Latent-space Jacobian of pi0.5: how strongly does the next plan respond to the change in the
policy's own input embedding caused by executing a slightly different chunk?

The closed-loop sensitivity of an outcome to this chunk's noise is a product over replans:

    dpath/dz  x  prod_k [ darm/daction . dE/darm . dpath'/dE ]

dpath/dz is the chunk flow map (exact, ~0.5 cm per noise unit, flat across states: slice5). E is
pi0.5's prefix token embedding (2 x 256 SigLIP patch tokens + prompt/state tokens, 2048-D each),
the latent the policy conditions on. Everything after E (Gemma prefix -> K/V -> action expert ->
chunk) is differentiable, so dpath'/dE is exact (reverse mode, 30 executed-path outputs). The
non-smooth links (controller, MuJoCo, renderer, state discretisation) are pulled back by finite
differences: execute the chunk perturbed along its top singular direction e1 so that the predicted
executed path moves `delta` cm, re-render, re-embed -> Delta E. Then

    gain = |path'(E + Delta E) - path'(E)| / delta     (cm of next plan per cm of this chunk's path)

is the one-replan growth factor of the closed loop, measured where it happens.

Along the main rollout of goal5 / branch5 (noise stream `seed`), at every replan state:
  - J_E for N_PROBE probe seeds of the next chunk (branch5's seeds) and sigma_max(J_E): the gain
    along the most sensitive *arbitrary* latent direction (cm per unit |Delta E|)
  - the response to the previous chunk's perturbations: direct (finite difference through the
    policy), linear (J_E . Delta E), per-token contributions, arm displacement, |Delta E|
  - the chunk's own sigma_1 (order5.phys_jac), e1, and a random direction scaled to the same delta

    python latent5.py run "<prompt>" init seed [n_states]   # -> out/latent_<slug>_<init>_<seed>.json
    python latent5.py report
    python latent5.py plot
"""
import glob, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT, FIGS = os.path.join(HERE, "out"), os.path.join(HERE, "figs")
SUITE, TASK0, LAYOUT = "libero_goal", 0, 0
N_EXEC, N_PROBE = 10, 2
DELTAS = [0.01, 0.1, 1.0]            # cm of predicted executed path along e1
t00 = time.time()


def log(*a):
    print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)


def path_t(f, a):
    """torch: normalised chunk [H, A] -> executed eef path [N_EXEC*3] in cm (order5 calibration)."""
    import torch
    from order5 import OSC, GAIN
    scale = torch.as_tensor(f.a_scale[:3], dtype=a.dtype, device=a.device)
    return (100 * GAIN * OSC * torch.cumsum(a[:N_EXEC, :3] * scale, 0)).flatten()


def paths_from(f, E, pad, att, zp):
    """Executed paths [n, 30] (cm) of probe seeds zp under the conditioning built from E."""
    import torch
    from pi05flow import cexpand
    with torch.no_grad():
        c = cexpand(f.prefix_from(E, pad, att), len(zp))
        a = f.flow(f._z(zp), c)[..., :f.A]
        return torch.stack([path_t(f, a[i]) for i in range(len(zp))]).double().cpu().numpy()


def jac_E(f, E, pad, att, zp):
    """paths [n, 30] and dpath/dE [n, 30, P, W] (reverse mode, one backward pass per output)."""
    import torch
    from pi05flow import cexpand
    E = E.detach().requires_grad_(True)
    with torch.enable_grad():
        c = cexpand(f.prefix_from(E, pad, att), len(zp))
        a = f.flow(f._z(zp), c)[..., :f.A]
        paths = torch.stack([path_t(f, a[i]) for i in range(len(zp))])
        J = torch.zeros(len(zp), paths.shape[1], *E.shape[1:], device=E.device)
        for i in range(len(zp)):
            for r in range(paths.shape[1]):
                g, = torch.autograd.grad(paths[i, r], E, retain_graph=True)
                J[i, r] = g[0]
    return paths.detach().double().cpu().numpy(), J


def blocks(v, layout):
    """Sum a per-token vector over the image slots and the text block."""
    b = np.cumsum([0] + layout["slots"])
    return np.array([v[b[i]:b[i + 1]].sum() for i in range(len(b) - 1)] + [v[b[-1]:].sum()])


def execute(s, x, a, text, f):
    """Restore state x, execute env chunk a for N_EXEC steps; return (E, eef, obs) afterwards."""
    s.set_state(x)
    for j in range(N_EXEC):
        obs, _, _ = s.step(a[j])
    o = s.obs_to_lerobot(obs, text)
    E, _, _ = f.embed([o], [text])
    return E, s.eef(), obs


def run(text, init, seed, n_states=100):
    import torch
    import pi05flow
    from pi05flow import H, D_ACT
    from env5 import Scene, TERMINAL, GOAL_NAMES, EVENT_NAMES
    from order5 import phys_jac
    A = H * D_ACT
    f = pi05flow.load()
    s = Scene(TASK0, init, LAYOUT, suite_name=SUITE)
    obs = s.reset()
    rng = np.random.default_rng(seed)                       # main stream, as branch5 / goal5
    pend, rows, k, t, main = [], [], 0, 0, "none"
    while k < n_states:
        o = s.obs_to_lerobot(obs, text)
        x = s.get_state()
        E, pad, att = f.embed([o], [text])
        P = int(E.shape[1])
        slots = [n * n for n in f.n_img]                    # image token slots (a slot may be an empty camera)
        b = np.cumsum([0] + slots); pn = pad[0].cpu().numpy()
        layout = dict(slots=slots, valid=[int(pn[b[i]:b[i + 1]].sum()) for i in range(len(slots))] + [int(pn[b[-1]:].sum())],
                      P=P)
        zp = np.random.default_rng(20_000 + k).standard_normal((N_PROBE, A))
        t0 = time.time()
        paths0, J = jac_E(f, E, pad, att, zp)
        Jf = J.flatten(2)                                   # [n, 30, P*W]
        sig = [torch.linalg.eigvalsh(Jf[i] @ Jf[i].T).sqrt().flip(0)[:3].tolist() for i in range(N_PROBE)]
        tj = time.time() - t0
        row = dict(k=k, t=t, eef=s.eef().tolist(), sigma_E=sig, layout=layout, responses=[])
        for p in pend:                                       # responses to the previous chunk's perturbations
            dE = (p["E"] - E)[0]
            paths1 = paths_from(f, p["E"], pad, att, zp)
            direct = paths1 - paths0                                              # [n, 30]
            tok = torch.einsum("nrpw,pw->nrp", J, dE)                              # [n, 30, P]
            lin = tok.sum(-1).double().cpu().numpy()
            contrib = tok.norm(dim=1).mean(0).double().cpu().numpy()               # [P] mean over probes
            nd, nl = np.linalg.norm(direct, axis=1), np.linalg.norm(lin, axis=1)
            cos = (direct * lin).sum(1) / np.maximum(nd * nl, 1e-12)
            changed = (dE.abs().sum(-1) > 0).double().cpu().numpy()
            r = dict(kind=p["kind"], delta=p["delta"], arm_mm=float(1000 * np.linalg.norm(p["eef"] - s.eef())),
                     dE=float(dE.norm()), tokens_changed=blocks(changed, layout).tolist(),
                     direct=nd.tolist(), linear=nl.tolist(), cos=cos.tolist(), contrib=contrib.tolist())
            row["responses"].append(r)
        # this chunk: main draw, its top singular direction, perturbed executions
        z_main = rng.standard_normal((1, A))
        with torch.no_grad():
            c = f.prefix_from(E, pad, att)
        a_main, Jz = f.jac(z_main, c, steps=N_EXEC, bs=1, chunk=70)
        Pz = phys_jac(f, Jz, N_EXEC)[0]
        U, S, Vt = np.linalg.svd(Pz, full_matrices=False)
        e1, sigma1 = Vt[0], S[0]
        rd = np.random.default_rng(40_000 + k).standard_normal(A); rd /= np.linalg.norm(rd)
        gain_r = np.linalg.norm(Pz @ rd)
        row.update(sigma1=float(sigma1), ftle=float(np.log(sigma1)), gain_rand=float(gain_r))
        perts = [("base", 0.0, np.zeros(A))] + [("e1", d, d / sigma1 * e1) for d in DELTAS] + \
                [("rand", DELTAS[1], DELTAS[1] / gain_r * rd)]
        pend = []
        for kind, d, dz in perts:
            a = f.unnorm(f.flow_np(z_main + dz, c)[0])
            Ep, eefp, _ = execute(s, x, a, text, f)
            pend.append(dict(kind=kind, delta=d, E=Ep, eef=eefp))
        # main chunk (the last execution is the base chunk, so just repeat it and keep the state)
        a = f.unnorm(a_main[0])
        s.set_state(x)
        done = False
        for j in range(N_EXEC):
            obs, _, _ = s.step(a[j]); t += 1
            g = s.preds()
            if (g & TERMINAL).any():
                main = (GOAL_NAMES + EVENT_NAMES)[int(np.argmax(g & TERMINAL))]; done = True; break
        rows.append(row)
        rs = {r["kind"] + (f"{r['delta']:g}" if r["kind"] == "e1" else ""): r for r in row["responses"]}
        msg = " | ".join(f"{kk}: arm {r['arm_mm']:.2f}mm gain {np.mean(r['direct']) / max(r['delta'], 1e-9):.1f}"
                         for kk, r in rs.items() if r["kind"] != "base")
        log(f"'{text}' t {t - N_EXEC:3d}: sigma1 {sigma1:.3f} sigma_E {np.mean([v[0] for v in sig]):.3f} "
            f"(J_E {tj:.0f}s) {msg}")
        k += 1
        if done or t >= s.max_steps:
            break
    s.close()
    path = os.path.join(OUT, f"latent_{text.replace(' ', '_')}_{init}_{seed}.json")
    json.dump(dict(text=text, init=init, seed=seed, main=main, deltas=DELTAS, n_probe=N_PROBE, states=rows),
              open(path, "w"))
    log(f"main rollout: {main} at t {t}; wrote {path}")
    report(path)


def top_dir(f, z, c, n_iter=4, eps=1e-3):
    """Top right singular vector and value of the executed-path map z -> path (cm) at z, by power
    iteration through the expert only: P v by a central finite difference (two forwards in one
    batch), P^T w by one autograd backward. sigma_2 / sigma_1 ~ 0.3, so 4 iterations suffice."""
    import torch
    from pi05flow import cexpand
    zt = f._z(z[None])
    c2 = cexpand(c, 2)
    g = lambda zz: path_t(f, f.flow(zz, c)[0, :, :f.A])  # noqa: E731

    def Pv(v):
        with torch.no_grad():
            a = f.flow(torch.cat([zt + eps * v, zt - eps * v]), c2)[:, :, :f.A]
            return (path_t(f, a[0]) - path_t(f, a[1])) / (2 * eps)

    v = torch.randn_like(zt); v /= v.norm()
    for _ in range(n_iter):
        w = Pv(v)
        zg = zt.detach().requires_grad_(True)
        with torch.enable_grad():
            u, = torch.autograd.grad(g(zg) @ w, zg)
        v = u / u.norm()
    return v.flatten().double().cpu().numpy(), float(Pv(v).norm())


def gain_rollout(f, s, text, seed, n_states=100, delta=1.0, attrib=True, quiet=False):
    """Per-replan gain in ~3.5 s along the main rollout (noise stream `seed`, as goal5 / branch5)
    from the scene's current reset: e1 by power iteration, +-e1 and +-random executed (each moving
    the predicted path by delta cm), the response by one extra forward pass, token attribution by
    one backward pass (attrib). Returns (rows, main label, t)."""
    import torch
    from pi05flow import H, D_ACT, cexpand
    from env5 import TERMINAL, GOAL_NAMES, EVENT_NAMES
    A = H * D_ACT
    obs = s.reset()
    rng = np.random.default_rng(seed)
    pend, rows, k, t, main = None, [], 0, 0, "none"
    while k < n_states:
        t0 = time.time()
        o = s.obs_to_lerobot(obs, text)
        x = s.get_state()
        E, pad, att = f.embed([o], [text])
        with torch.no_grad():
            c = f.prefix_from(E, pad, att)
        zp = np.random.default_rng(20_000 + k).standard_normal((N_PROBE, A))
        row = dict(k=k, t=t, eef=s.eef().tolist())
        if pend is not None:
            paths0 = paths_from(f, E, pad, att, zp)
            gains, arms, dEs = [], [], []
            for p in pend:                                   # +e1, -e1, +r, -r of the previous chunk
                direct = paths_from(f, p["E"], pad, att, zp) - paths0
                gains.append(float(np.mean(np.linalg.norm(direct, axis=1))) / delta)
                arms.append(float(1000 * np.linalg.norm(p["eef"] - s.eef())))
                dEs.append(float((p["E"] - E).norm()))
                if attrib and p["kind"] == "+e1":
                    # attribution: gradient of (u . path) wrt E, u = response direction, times dE per token
                    dE = (p["E"] - E)[0]
                    u = torch.as_tensor(direct[0] / max(np.linalg.norm(direct[0]), 1e-12), dtype=torch.float32,
                                        device=E.device)
                    Eg = E.detach().requires_grad_(True)
                    with torch.enable_grad():
                        cg = cexpand(f.prefix_from(Eg, pad, att), 1)
                        pth = path_t(f, f.flow(f._z(zp[:1]), cg)[0, :, :f.A])
                        g, = torch.autograd.grad(pth @ u, Eg)
                    contrib = (g[0] * dE).sum(-1).double().cpu().numpy()      # signed, sums to u . (J dE)
            row.update(delta=delta, kinds=[p["kind"] for p in pend], gains=gains, arm_mm=arms, dE=dEs,
                       gain=float(np.mean(gains)), layout=dict(slots=[n * n for n in f.n_img]))
            if attrib:
                row["contrib"] = contrib.tolist()
        z_main = rng.standard_normal((1, A))
        e1, sigma1 = top_dir(f, z_main[0], c)
        rd = np.random.default_rng(40_000 + k).standard_normal(A); rd /= np.linalg.norm(rd)
        with torch.no_grad():                                 # scale r to the same predicted path move
            ar = f.flow(f._z(np.stack([z_main[0] + 1e-2 * rd, z_main[0] - 1e-2 * rd])), cexpand(c, 2))[:, :, :f.A]
            gain_r = float((path_t(f, ar[0]) - path_t(f, ar[1])).norm() / 2e-2)
        row.update(sigma1=sigma1, gain_rand=gain_r)
        a_main = f.unnorm(f.flow_np(z_main, c)[0])
        pend = []
        for kind, dz in (("+e1", delta / sigma1 * e1), ("-e1", -delta / sigma1 * e1),
                         ("+r", delta / gain_r * rd), ("-r", -delta / gain_r * rd)):
            Ep, eefp, _ = execute(s, x, f.unnorm(f.flow_np(z_main + dz, c)[0]), text, f)
            pend.append(dict(kind=kind, E=Ep, eef=eefp))
        s.set_state(x)
        done = False
        for j in range(N_EXEC):
            obs, _, _ = s.step(a_main[j]); t += 1
            gg = s.preds()
            if (gg & TERMINAL).any():
                main = (GOAL_NAMES + EVENT_NAMES)[int(np.argmax(gg & TERMINAL))]; done = True; break
        row["secs"] = time.time() - t0
        rows.append(row)
        if "gains" in row and not quiet:
            log(f"'{text}' t {t - N_EXEC:3d}: sigma1 {sigma1:.3f} arm {np.mean(row['arm_mm']):.2f}mm gain "
                f"{row['gain']:5.1f} [{min(row['gains']):.1f}, {max(row['gains']):.1f}]  ({row['secs']:.1f}s)")
        k += 1
        if done or t >= s.max_steps:
            break
    return rows, main, t


def fast(text, init, seed, n_states=100, delta=1.0):
    """gain_rollout on a fresh scene -> out/fast_<slug>_<init>_<seed>.json"""
    import pi05flow
    from env5 import Scene
    f = pi05flow.load()
    s = Scene(TASK0, init, LAYOUT, suite_name=SUITE)
    rows, main, t = gain_rollout(f, s, text, seed, n_states, delta)
    s.close()
    path = os.path.join(OUT, f"fast_{text.replace(' ', '_')}_{init}_{seed}.json")
    json.dump(dict(text=text, init=init, seed=seed, main=main, delta=delta, n_probe=N_PROBE, states=rows), open(path, "w"))
    log(f"main rollout: {main} at t {t}; wrote {path}")


def report(path=None):
    for p in [path] if path else sorted(glob.glob(os.path.join(OUT, "latent_*.json"))):
        R = json.load(open(p))
        print(f"\n'{R['text']}' init {R['init']} seed {R['seed']} -> {R['main']}   (gain = cm of next plan per cm of "
              f"this chunk's path; sigma_E = cm per unit |dE|, any direction)")
        lay = R["states"][0]["layout"]
        keep = [i for i, n in enumerate(lay["valid"]) if n > 0]
        names = "/".join(("text" if i == len(lay["slots"]) else f"cam{i + 1}") for i in keep)
        print(f"  {'t':>4s} {'sig1':>5s} {'sigE':>6s} | e1 d=0.01: {'arm':>6s} {'gain':>6s} | d=0.1: {'arm':>6s} {'|dE|':>5s} "
              f"{'gain':>6s} {'lin':>6s} {'cos':>5s} {names:>16s} | d=1: {'arm':>6s} {'gain':>6s} | rand: {'arm':>6s} {'gain':>6s}")
        for st in R["states"][1:]:
            rs = {r["kind"] + (f"{r['delta']:g}" if r["kind"] == "e1" else ""): r for r in st["responses"]}
            g = lambda r: np.mean(r["direct"]) / r["delta"]  # noqa: E731
            r = rs["e10.1"]
            share = blocks(np.array(r["contrib"]), st["layout"])[keep]
            share = share / max(share.sum(), 1e-12)
            print(f"  {st['t']:4d} {st['sigma1']:5.3f} {np.mean([v[0] for v in st['sigma_E']]):6.3f} | "
                  f"{rs['e10.01']['arm_mm']:6.3f} {g(rs['e10.01']):6.1f} | {r['arm_mm']:6.2f} {r['dE']:5.1f} "
                  f"{g(r):6.1f} {np.mean(r['linear']) / r['delta']:6.1f} {np.mean(r['cos']):5.2f} "
                  f"{'/'.join(f'{x:.2f}' for x in share):>16s} | {rs['e11']['arm_mm']:6.2f} {g(rs['e11']):6.1f} | "
                  f"{rs['rand']['arm_mm']:6.2f} {g(rs['rand']):6.1f}")
        base = [np.mean(st["responses"][0]["direct"]) for st in R["states"][1:]]
        print(f"  re-executing the base chunk changes the next plan by max {max(base):.2e} cm (restore check)")


def plot(t_map=20):
    """figs/latent_pi05.png: one-replan gain along both rollouts (one panel per delta) and the
    per-token contribution maps of J_E . Delta E at state t_map (delta 0.1 cm)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    CAT = ["#2a78d6", "#eb6834"]
    BLUES = ["#eaf2fc", "#c4dbf5", "#8fbcec", "#5a9be0", "#2a78d6", "#1d5aa6", "#123d73"]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
    Rs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "latent_*.json")))]
    Rs.sort(key=lambda R: R["text"] != "open a drawer")
    deltas = Rs[0]["deltas"]
    fig = plt.figure(figsize=(11.5, 7.2))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.0, 1.0], hspace=0.5, wspace=0.9)
    for di, d in enumerate(deltas):
        a = fig.add_subplot(gs[0, 4 * di:4 * di + 4])
        for R, col in zip(Rs, CAT):
            ts, gs_ = [], []
            for st in R["states"][1:]:
                r = [r for r in st["responses"] if r["kind"] == "e1" and r["delta"] == d][0]
                ts.append(st["t"]); gs_.append(np.mean(r["direct"]) / d)
            a.plot(ts, gs_, "-o", color=col, ms=3.5, lw=1.6, label=f"\"{R['text']}\"")
        a.axhline(1, color=MUTED, lw=0.8, ls="--")
        a.axvspan(5, 35, color="#2a78d6", alpha=0.07, lw=0)
        a.set_yscale("log"); a.set_xlabel("t of the response (steps)")
        a.set_title(f"chunk path moved {d:g} cm along e1", fontsize=9.5, color=INK)
        if di == 0:
            a.set_ylabel("gain: cm of next plan per cm of this chunk's path")
            a.legend(frameon=False, fontsize=8.5, loc="upper left")
        a.text(20, a.get_ylim()[0] * 1.3, "decision\nwindow", ha="center", va="bottom", fontsize=8, color=INK2)
    # token maps
    cmap = LinearSegmentedColormap.from_list("b", BLUES)
    for ri, R in enumerate(Rs):
        st = [s for s in R["states"] if s["t"] == t_map][0]
        r = [r for r in st["responses"] if r["kind"] == "e1" and r["delta"] == 0.1][0]
        c = np.array(r["contrib"]); n = int(np.sqrt(st["layout"]["slots"][0]))
        tot = blocks(c, st["layout"])
        for ci, name in enumerate(["agentview", "wrist"]):
            a = fig.add_subplot(gs[1, 3 * (2 * ri + ci):3 * (2 * ri + ci) + 3])
            m = c[ci * n * n:(ci + 1) * n * n].reshape(n, n)
            a.imshow(m, cmap=cmap, vmin=0, vmax=m.max(), interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"\"{R['text'][:28]}{'…' if len(R['text']) > 28 else ''}\"\n{name} tokens, t = {t_map}: "
                        f"sum {tot[ci]:.3f} cm", fontsize=8.5, color=INK)
    fig.text(0.5, 0.455, "per-token contribution |J_E[:, i] · ΔE_i| to the next plan's response (each map on its own scale); "
             "text/state tokens contribute < 1% at this state", ha="center", fontsize=8.5, color=INK2)
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, "latent_pi05.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), *[int(x) for x in sys.argv[5:]])
    elif sys.argv[1] == "fast":
        fast(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), *[float(x) for x in sys.argv[5:]])
    elif sys.argv[1] == "report":
        report()
    elif sys.argv[1] == "plot":
        plot()
