"""
Prompt-space ridge: can the policy alone, from one observation, tell that a prompt is ambiguous?

The LCS method of robot2 moved to the prompt-embedding block E_p of pi0.5's prefix: the plan
(predicted 50-step eef path, cm) as a function of E_p has basins (one per behaviour the language
could select) separated by a ridge. An ambiguous prompt should sit near the ridge: a small move of
its embedding flips the plan. A specific prompt should sit deep in its basin.

  check   known answer: interpolate the "a" slot of "open a drawer" between emb(top) and
          emb(middle) (s = 0 -> top, 1 -> middle) and print lambda(plan) along s: a step (ridge)
          or a ramp? Also where "open a drawer" itself sits.
  screen  for every goal5 prompt x initial state at t 0, candidate-free: top singular direction
          v1 of dplan/dE_p (power iteration), sigma_1, and the plan displacement d(alpha) along
          +-v1 for alpha = a grid of fractions of the mean prompt-token norm (one batched
          forward). Features: sigma_1, d at fixed alpha, and the margin alpha* = first alpha
          where d exceeds D_JUMP. Ground truth: goal5 split labels.  -> out/pridge_screen.json
  report  AUROC overall / within state, 95% bootstrap clustered by state (screen5 conventions).

    python pridge5.py check | screen | report
"""
import json, os, sys, time
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "out")
SUITE, N_INIT, N_Z = "libero_goal", 4, 2
ALPHAS = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]      # fractions of the mean prompt-token norm
D_JUMP = 20.0                                         # cm (150-D path norm) = "the plan changed mode"
t00 = time.time()


def log(*a):
    print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)


def path50_t(f, a):
    import torch
    from order5 import OSC, GAIN
    sc = torch.as_tensor(f.a_scale[:3], dtype=a.dtype, device=a.device)
    sh = torch.as_tensor(f.a_shift[:3], dtype=a.dtype, device=a.device)
    return (100 * GAIN * OSC * torch.cumsum(a[:, :3] * sc + sh, 0)).flatten()


def plans(f, E, pad, att, zp):
    """E [N, P, W] (pad/att [1, P] broadcast) -> plans [N, n_z, 150] cm."""
    import torch
    from pi05flow import cexpand, cslice
    N, out = E.shape[0], []
    with torch.no_grad():
        for i0 in range(0, N, 8):
            Eb = E[i0:i0 + 8]; n = Eb.shape[0]
            c = f.prefix_from(Eb, pad.expand(n, -1), att.expand(n, -1))
            for i in range(n):
                a = f.flow(f._z(zp), cexpand(cslice(c, i, i + 1), len(zp)))[..., :f.A]
                out.append(torch.stack([path50_t(f, a[j]) for j in range(len(zp))]))
            del c
    return torch.stack(out)


def prompt_slots(f, o, text):
    """Indices in E of the prompt's own tokens (between 'Task:' and ', State:')."""
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
    import pi05flow
    sp = pi05flow.SPTokenizer().sp
    ids = f.batch([o], [text])[OBS_LANGUAGE_TOKENS][0].cpu().numpy()
    n_img = sum(n * n for n in f.n_img)
    pcs = [sp.id_to_piece(int(i)) for i in ids]
    j = pcs.index("▁State")
    return list(range(n_img + 3, n_img + j - 1)), pcs[3:j - 1]


def top_dir_E(f, E, pad, att, slots, zp, n_iter=4):
    """Top right singular direction (over the prompt slots) and value of E -> mean plan."""
    import torch
    from pi05flow import cexpand
    r = E[0, slots].norm(dim=1).mean()
    eps = 1e-3 * r

    def Pv(v):                                          # [len(slots), W] -> [150], central FD
        Ep = E.repeat(2, 1, 1); Ep[0, slots] += eps * v; Ep[1, slots] -= eps * v
        p = plans(f, Ep, pad, att, zp).mean(1)
        return (p[0] - p[1]) / (2 * eps)

    v = torch.randn(len(slots), E.shape[2], device=E.device); v /= v.norm()
    for _ in range(n_iter):
        w = Pv(v)
        Eg = E.detach().requires_grad_(True)
        with torch.enable_grad():
            c = cexpand(f.prefix_from(Eg, pad, att), len(zp))
            a = f.flow(f._z(zp), c)[..., :f.A]
            obj = torch.stack([path50_t(f, a[j]) for j in range(len(zp))]).mean(0) @ w
            g, = torch.autograd.grad(obj, Eg)
        u = g[0, slots]; v = u / u.norm()
    return v, float(Pv(v).norm()), float(r)


def features(f, o, text, zp):
    import torch
    E, pad, att = f.embed([o], [text])
    slots, pcs = prompt_slots(f, o, text)
    v1, s1, r = top_dir_E(f, E, pad, att, slots, zp)
    rd = torch.randn_like(v1); rd /= rd.norm()
    Es = []
    for d in (v1, -v1, rd, -rd):
        for a in ALPHAS:
            Ep = E.clone(); Ep[0, slots] += a * r * d; Es.append(Ep)
    P = plans(f, torch.cat(Es + [E]), pad, att, zp).mean(1)            # [4*len(ALPHAS)+1, 150]
    p0 = P[-1]
    d = (P[:-1] - p0).norm(dim=1).view(4, len(ALPHAS)).double().cpu().numpy()
    dv = d[:2].max(0)                                                   # along +-v1
    jump = [a for a, x in zip(ALPHAS, dv) if x >= D_JUMP]
    return dict(text=text, tokens=pcs, sigma1=s1, r=r, d_v=d[:2].tolist(), d_rand=d[2:].tolist(),
                d_at=dict(zip(map(str, ALPHAS), dv.tolist())), alpha_star=jump[0] if jump else float("inf"))


def check(init=3):
    import torch
    import pi05flow
    from env5 import Scene
    f = pi05flow.load()
    emb = f.m.paligemma_with_expert.paligemma.model.language_model.get_input_embeddings()
    sp = pi05flow.SPTokenizer().sp
    e_top, e_mid = [emb(torch.tensor([sp.piece_to_id("▁" + w)], device=f.dev))[0] for w in ("top", "middle")]
    e_a = emb(torch.tensor([sp.piece_to_id("▁a")], device=f.dev))[0]
    log(f"|emb(top)| {e_top.norm():.2f} |emb(middle)| {e_mid.norm():.2f} |emb(a)| {e_a.norm():.2f} "
        f"|top-middle| {(e_top - e_mid).norm():.2f} |a-top| {(e_a - e_top).norm():.2f} |a-middle| {(e_a - e_mid).norm():.2f}")
    zp = np.random.default_rng(777).standard_normal((N_Z, pi05flow.H * pi05flow.D_ACT))
    for init in range(N_INIT):
        s = Scene(0, init, 0, suite_name=SUITE); obs = s.reset()
        o = s.obs_to_lerobot(obs, "open a top drawer")
        E, pad, att = f.embed([o], ["open a top drawer"])
        slots, _ = prompt_slots(f, o, "open a top drawer")
        k = slots[2]                                             # the "top" slot
        ss = np.linspace(-0.5, 1.5, 41)
        Es = []
        for sv in ss:
            Ep = E.clone(); Ep[0, k] = (1 - sv) * e_top + sv * e_mid; Es.append(Ep)
        Ea = E.clone(); Ea[0, k] = e_a                           # "open a a drawer" ~ the ambiguous prompt's slot
        P = plans(f, torch.cat(Es + [Ea]), pad, att, zp).mean(1)
        pT, pM = P[list(ss).index(0.0)], P[list(ss).index(1.0)]
        lam = (((P - pM) @ (pT - pM)) / (pT - pM).norm() ** 2).double().cpu().numpy()
        dl = np.abs(np.diff(lam[:-1])) / np.diff(ss)
        log(f"init {init}: |plan(top) - plan(middle)| {(pT - pM).norm():.1f} cm; lambda along s (0 = top, 1 = middle):")
        print("   s:      " + " ".join(f"{x:5.2f}" for x in ss[::2]))
        print("   lambda: " + " ".join(f"{x:5.2f}" for x in lam[:-1][::2]))
        i = int(np.argmax(dl))
        print(f"   steepest step {dl[i]:.2f} per unit s at s {ss[i]:.2f}-{ss[i+1]:.2f} (a ramp would be ~1); "
              f"'open a [a] drawer' lambda {lam[-1]:.2f}", flush=True)
        s.close()


def screen():
    import pi05flow
    from env5 import Scene
    from goal5 import PROMPTS
    path = os.path.join(OUT, "pridge_screen.json")
    R = json.load(open(path)) if os.path.exists(path) else {}
    f = pi05flow.load()
    zp = np.random.default_rng(777).standard_normal((N_Z, pi05flow.H * pi05flow.D_ACT))
    for init in range(N_INIT):
        todo = [p for p in PROMPTS if f"{p[0]}|{init}" not in R]
        if not todo:
            continue
        s = Scene(0, init, 0, suite_name=SUITE); obs = s.reset()
        for text, kind, _ in todo:
            o = s.obs_to_lerobot(obs, text)
            t0 = time.time()
            r = features(f, o, text, zp)
            R[f"{text}|{init}"] = dict(r, kind=kind, init=init)
            json.dump(R, open(path, "w"))
            log(f"init {init} '{text}' ({kind}): sigma1 {r['sigma1']:.1f} cm/unit, r {r['r']:.2f}, d(alpha) along v1 "
                + " ".join(f"{x:.0f}" for x in np.max(r["d_v"], 0)) + f" | rand " + " ".join(f"{x:.0f}" for x in np.max(r["d_rand"], 0))
                + f"  alpha* {r['alpha_star']}  ({time.time() - t0:.1f}s)")
        s.close()


def report():
    from screen5 import auroc, boot
    R = json.load(open(os.path.join(OUT, "pridge_screen.json")))
    G = json.load(open(os.path.join(OUT, "goal_run.json")))
    labels = defaultdict(set)
    for r in G:
        labels[(r["text"], r["init"])].add(r["label"])
    rows = []
    for v in R.values():
        lab = labels.get((v["text"], v["init"]))
        if lab is None:
            continue
        feats = dict(sigma1=v["sigma1"], neg_alpha_star=-v["alpha_star"] if np.isfinite(v["alpha_star"]) else -10.0)
        for a in ALPHAS:
            feats[f"d@{a}"] = v["d_at"][str(a)]
        rows.append(dict(text=v["text"], kind=v["kind"], init=v["init"], split=len(lab) > 1, n=len(lab), **feats))
    y = [r["split"] for r in rows]; grp = [r["init"] for r in rows]
    print(f"{len(rows)} (prompt, init) pairs, {sum(y)} split in goal5\n")
    for feat in ["sigma1", "neg_alpha_star"] + [f"d@{a}" for a in ALPHAS]:
        sc = [r[feat] for r in rows]
        lo, hi = boot(sc, y, grp)
        within = [auroc([r[feat] for r in rows if r["init"] == i], [r["split"] for r in rows if r["init"] == i]) for i in range(N_INIT)]
        print(f"  {feat:14s}: AUROC {auroc(sc, y):.2f} [{lo:.2f}, {hi:.2f}]   within init: " + " ".join(f"{a:.2f}" for a in within))
    print("\n  sigma1 | alpha* | d@0.1 d@0.4 | labels | prompt")
    for r in sorted(rows, key=lambda r: (r["init"], r["neg_alpha_star"], -r["sigma1"]), reverse=False):
        print(f"  init {r['init']} {'SPLIT' if r['split'] else '     '} {r['sigma1']:7.1f} | {-r['neg_alpha_star']:6.3f} | "
              f"{r['d@0.1']:5.0f} {r['d@0.4']:5.0f} | {r['n']} | {r['text']} ({r['kind']})")


if __name__ == "__main__":
    {"check": check, "screen": screen, "report": report}[sys.argv[1]]()
