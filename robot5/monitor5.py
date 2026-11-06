"""
Simulator-free monitor: decode the counterfactual prompts from the policy alone at t 0, then watch
the plan's position between them at every replan while the policy executes.

The decode pipeline (README, "Decoding a split into prompt suggestions") found the two modes by
branching the simulator (26 s) and the words that select them from one gradient. Here the modes
come from the policy's own ridge in prompt space instead (pridge5): the two sides of the ridge
along the top singular direction v1 of dplan/dE_prompt give the two plans P+ / P-, Delta = P+ - P-
replaces the branch difference, and the rest is unchanged (one gradient, first-order score of
every dictionary word in every prompt slot, verification by plan projection). No rollout.

Then, during execution (the simulator only plays the role of the robot): at every replan the plan
under the executed prompt and under the two decoded prompts c+ / c-, same state, same noise, give
lambda = position of the plan between them (0 = c-, 1 = c+) and their separation (intent5). The
hypothesis: lambda pinned = the prompt behaves like a specific one; lambda drifting inside the
first replans = the split happening, with a question available before the outcome is decided.

  decode [init] [prompt]  decode one (prompt, init) at t 0 and print the candidates
  run [n_init] [n_seed]   every goal5 prompt x init: decode at t 0, then n_seed closed-loop rollouts
                          (prefix batch 1, fresh noise per seed, stop at the first terminal event)
                          recording lambda / separation at every replan -> out/monitor_run.json
  report                  pair-level: does early lambda drift predict that the pair splits (labels
                          from the same rollouts)? vs the t 0 ridge features. Rollout-level: does
                          late lambda say which outcome? Lead time of the warning vs the outcome.

    python monitor5.py decode 3 "open a drawer" | run | report
"""
import json, os, re, sys, time
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "out")
SUITE, N_Z, SHORT = "libero_goal", 2, 4
MAX_ID = 50000                                        # common-word prior: Gemma piece ids track frequency (the 573, stove 37932, budapest 215375)
ALPHAS = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8]
D_JUMP, ALPHA_DEFAULT = 20.0, 0.4
STEPS = (9, 29, 49)                                   # chunk steps at which lambda is recorded
t00 = time.time()


def log(*a):
    print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)


def dictionary(f):
    """Vocabulary pieces that are plain dictionary words, and the embedding table."""
    import torch
    import pi05flow
    sp = pi05flow.SPTokenizer().sp
    emb = f.m.paligemma_with_expert.paligemma.model.language_model.get_input_embeddings()
    V = sp.get_piece_size()
    pieces = [sp.id_to_piece(i) for i in range(V)]
    dic = {w.strip().lower() for w in open("/usr/share/dict/american-english", errors="ignore") if w.strip().isalpha()}
    ok = np.array([re.fullmatch(r"▁[a-z]{2,}", p) is not None and p[1:] in dic and i < MAX_ID for i, p in enumerate(pieces)])
    with torch.no_grad():
        Evoc = torch.cat([emb(torch.arange(i, min(i + 32768, V), device=f.dev)) for i in range(0, V, 32768)])
    return pieces, ok, Evoc


def grad_E(f, E, pad, att, zp, d):
    """gradient of mean_z <plan50(z; E), d> wrt E  ->  [P, W]"""
    import torch
    from pi05flow import cexpand
    from pridge5 import path50_t
    Eg = E.detach().requires_grad_(True)
    with torch.enable_grad():
        a = f.flow(f._z(zp), cexpand(f.prefix_from(Eg, pad, att), len(zp)))[..., :f.A]
        obj = torch.stack([path50_t(f, a[i]) for i in range(len(zp))]).mean(0) @ d
        g, = torch.autograd.grad(obj, Eg)
    return g[0]


def decode(f, o, text, zp, voc):
    """Candidate-free, simulator-free: the two prompts that select the two sides of the ridge."""
    import torch
    from pridge5 import plans, prompt_slots, top_dir_E
    pieces, ok, Evoc = voc
    E, pad, att = f.embed([o], [text])
    slots, pcs = prompt_slots(f, o, text)
    v1, s1, r = top_dir_E(f, E, pad, att, slots, zp)
    Es = []
    for a in ALPHAS:
        for sgn in (1, -1):
            Ep = E.clone(); Ep[0, slots] += sgn * a * r * v1; Es.append(Ep)
    P = plans(f, torch.cat(Es + [E]), pad, att, zp).mean(1)                 # [2*len(ALPHAS)+1, 150]
    side = (P[0:-1:2] - P[1:-1:2]).norm(dim=1).double().cpu().numpy()       # |P+ - P-| per alpha
    j = next((k for k, x in enumerate(side) if x >= D_JUMP), ALPHAS.index(ALPHA_DEFAULT))
    Pp, Pm = P[2 * j], P[2 * j + 1]
    d = Pp - Pm; sep = float(d.norm())
    g = grad_E(f, E, pad, att, zp, d / d.norm())
    cands = {}
    for i, (k, pc) in enumerate(zip(slots, pcs)):
        sc = ((Evoc - E[0, k]) @ g[k]).double().cpu().numpy()
        sc[~ok] = np.nan; sc[pieces.index(pc)] = np.nan
        order = np.argsort(np.nan_to_num(sc, nan=0.0))
        for sign, vs in (("+", order[-SHORT:][::-1]), ("-", order[:SHORT])):
            for v in vs:
                if np.isnan(sc[v]):
                    continue
                new = pcs.copy(); new[i] = pieces[v]
                tx = "".join(new).replace("▁", " ").strip()
                cands[tx] = dict(slot=pc[1:], word=pieces[v][1:], sign=sign, fo=float(sc[v]), piece_id=int(v), slot_idx=k)
    # verification by plan projection: same token count -> same pad/att, one batch
    Ec = []
    for tx, c in cands.items():
        Ep = E.clone(); Ep[0, c["slot_idx"]] = Evoc[c["piece_id"]]; Ec.append(Ep)
    Pc = plans(f, torch.cat(Ec), pad, att, zp).mean(1)
    tw = ((Pc - Pm).norm(dim=1) - (Pc - Pp).norm(dim=1)).double().cpu().numpy() / sep
    for (tx, c), t_ in zip(cands.items(), tw):
        c["toward"] = float(t_)
    rows = sorted(cands.values(), key=lambda c: -c["toward"])
    texts = list(cands.keys())
    ip, im = int(np.argmax(tw)), int(np.argmin(tw))
    lam0 = float(((P[-1] - Pm) @ d) / (d @ d))
    return dict(text=text, tokens=[p[1:] for p in pcs], sigma1=s1, r=r, alpha=ALPHAS[j], side=side.tolist(), sep=sep,
                lam0=lam0, cplus=texts[ip], cminus=texts[im], toward_plus=float(tw[ip]), toward_minus=float(tw[im]),
                fo_plus=max(cands.values(), key=lambda c: c["fo"])["word"], fo_minus=min(cands.values(), key=lambda c: c["fo"])["word"],
                cands=[dict(text=tx, **c) for tx, c in cands.items()])


def rollout(f, s, x0, text, cplus, cminus, seed, max_t):
    """Execute `text` closed loop from state x0 (batch-1 prefix, replan every 10 steps); at every
    replan record lambda between the c- and c+ plans and their separation, at chunk steps STEPS."""
    from pi05flow import H, D_ACT
    from env5 import GOAL_NAMES, EVENT_NAMES, TERMINAL
    from goal5 import path_cm
    names = GOAL_NAMES + EVENT_NAMES
    obs = s.set_state(x0)
    rng = np.random.default_rng(seed)
    t, first, trace, label = 0, {}, [], None
    while t < max_t and label is None:
        o = s.obs_to_lerobot(obs, text)
        z = rng.standard_normal((1, H * D_ACT))
        p = {k: path_cm(f.unnorm(f.flow_np(z, f.prefix([o], [k]))[0])) for k in (text, cplus, cminus)}
        v = p[cplus] - p[cminus]
        lam = [float(((p[text][i] - p[cminus][i]) @ v[i]) / max(v[i] @ v[i], 1e-9)) for i in STEPS]
        sep = [float(np.linalg.norm(v[i])) for i in STEPS]
        trace.append(dict(t=t, lam=lam, sep=sep))
        a = f.unnorm(f.flow_np(z, f.prefix([o], [text]))[0])
        for j in range(10):
            obs, _, _ = s.step(a[j]); t += 1
            for i, q in enumerate(s.preds()):
                if q and names[i] not in first:
                    first[names[i]] = t
                    if TERMINAL[i]:
                        label = names[i]
            if label is not None or t >= max_t:
                break
    if label is None:
        opened = [(tt, n) for n, tt in first.items()]
        label = max(opened)[1] if opened else "none"
    return dict(seed=seed, label=label, t=t, events=sorted(first.items(), key=lambda kv: kv[1]), trace=trace)


def cmd_decode(init=3, text="open a drawer"):
    import pi05flow
    from env5 import Scene
    f = pi05flow.load(); voc = dictionary(f)
    zp = np.random.default_rng(777).standard_normal((N_Z, pi05flow.H * pi05flow.D_ACT))
    s = Scene(0, int(init), 0, suite_name=SUITE); obs = s.reset()
    o = s.obs_to_lerobot(obs, text)
    t0 = time.time(); r = decode(f, o, text, zp, voc)
    log(f"init {init} '{text}': sigma1 {r['sigma1']:.1f}, |P+ - P-| {r['sep']:.1f} cm at alpha {r['alpha']} (side " +
        " ".join(f"{x:.0f}" for x in r["side"]) + f"), lambda0 {r['lam0']:.2f}  ({time.time() - t0:.1f}s)")
    log(f"  c+ '{r['cplus']}' (toward {r['toward_plus']:+.2f})   c- '{r['cminus']}' (toward {r['toward_minus']:+.2f})"
        f"   first-order only: + {r['fo_plus']} / - {r['fo_minus']}")
    for c in sorted(r["cands"], key=lambda c: -c["toward"]):
        print(f"    {c['toward']:+.2f} {c['fo']:7.1f} [{c['slot']}{c['sign']}] {c['text']}")
    s.close()


def run(n_init=4, n_seed=4):
    import pi05flow
    from env5 import Scene
    from goal5 import PROMPTS
    path = os.path.join(OUT, "monitor_run.json")
    R = json.load(open(path)) if os.path.exists(path) else {}
    f = pi05flow.load(); voc = dictionary(f)
    zp = np.random.default_rng(777).standard_normal((N_Z, pi05flow.H * pi05flow.D_ACT))
    for init in range(int(n_init)):
        todo = [p for p in PROMPTS if len(R.get(f"{p[0]}|{init}", {}).get("rollouts", [])) < int(n_seed)]
        if not todo:
            continue
        s = Scene(0, init, 0, suite_name=SUITE); obs = s.reset(); x0 = s.get_state()
        for text, kind, _ in todo:
            key = f"{text}|{init}"
            if key not in R:
                o = s.obs_to_lerobot(obs, text)
                t0 = time.time(); d = decode(f, o, text, zp, voc); d["decode_s"] = time.time() - t0
                d["cands"] = sorted(d["cands"], key=lambda c: -c["toward"])[:6] + sorted(d["cands"], key=lambda c: c["toward"])[:6]
                R[key] = dict(d, kind=kind, init=init, rollouts=[])
                log(f"init {init} '{text}' ({kind}): sigma1 {d['sigma1']:.1f}, sep {d['sep']:.0f} cm, lam0 {d['lam0']:.2f}; "
                    f"c+ '{d['cplus']}' ({d['toward_plus']:+.2f}) c- '{d['cminus']}' ({d['toward_minus']:+.2f})  ({d['decode_s']:.0f}s)")
            d = R[key]
            for seed in range(len(d["rollouts"]), int(n_seed)):
                t0 = time.time()
                ro = rollout(f, s, x0, text, d["cplus"], d["cminus"], 1000 + seed, s.max_steps)
                d["rollouts"].append(ro)
                lam = [tr["lam"][2] for tr in ro["trace"]]
                log(f"    seed {seed}: {ro['label']}@{ro['t']}  lambda(50) " + " ".join(f"{x:.2f}" for x in lam[:8]) +
                    (" ..." if len(lam) > 8 else "") + f"  ({time.time() - t0:.0f}s)")
            json.dump(R, open(path, "w"))
        s.close()


def feats(ro, early=60, sep_gate=8.0):
    lam = np.array([tr["lam"][2] for tr in ro["trace"]]); ts = np.array([tr["t"] for tr in ro["trace"]])
    sep = np.array([tr["sep"][2] for tr in ro["trace"]])
    e = ts <= early
    off = np.minimum(np.abs(lam), np.abs(lam - 1))
    g = e & (sep >= sep_gate)                                  # replans where the counterfactuals still differ
    lam_g = lam[g] if g.any() else lam[:1]
    return dict(lam0=float(lam[0]), drift=float(lam[e].max() - lam[e].min()), drift_all=float(lam.max() - lam.min()),
                off=float(off[e].max()), sep_min=float(sep[e].min()), lam_late=float(lam[-3:].mean()),
                drift_g=float(lam_g.max() - lam_g.min()), off_g=float(off[g].max()) if g.any() else 0.0,
                off_cm=float((sep * off)[e].max()), drift_cm=float((sep * np.abs(lam - lam[0]))[e].max()),
                t_warn=next((int(t) for t, l in zip(ts, lam) if abs(l - lam[0]) >= 0.3), -1))


def report():
    from screen5 import auroc, boot
    R = json.load(open(os.path.join(OUT, "monitor_run.json")))
    S = json.load(open(os.path.join(OUT, "pridge_screen.json"))) if os.path.exists(os.path.join(OUT, "pridge_screen.json")) else {}
    pairs = []
    for key, d in R.items():
        if not d["rollouts"]:
            continue
        labs = [ro["label"] for ro in d["rollouts"]]
        F = [feats(ro) for ro in d["rollouts"]]
        pairs.append(dict(text=d["text"], init=d["init"], kind=d["kind"], split=len(set(labs)) > 1, labs=labs, F=F,
                          ts=[ro["t"] for ro in d["rollouts"]],
                          cplus=d["cplus"], cminus=d["cminus"], lam0=d["lam0"], sigma1=d["sigma1"], sep=d["sep"],
                          sigma1_ridge=S.get(key, {}).get("sigma1", np.nan),
                          drift=max(f["drift"] for f in F), off=max(f["off"] for f in F), drift_all=max(f["drift_all"] for f in F),
                          drift_g=max(f["drift_g"] for f in F), off_g=max(f["off_g"] for f in F),
                          off_cm=max(f["off_cm"] for f in F), drift_cm=max(f["drift_cm"] for f in F),
                          sep_min=min(f["sep_min"] for f in F), n_seed=len(labs)))
    y = [p["split"] for p in pairs]; grp = [p["init"] for p in pairs]
    print(f"{len(pairs)} (prompt, init) pairs, {sum(y)} split ({pairs[0]['n_seed']} seeds each)\n")
    print("pair-level: does the monitor's early (t <= 60) lambda drift predict that the pair splits?")
    for name, sc in (("drift (max over seeds)", [p["drift"] for p in pairs]), ("off-basin", [p["off"] for p in pairs]),
                     ("drift, sep >= 8 cm only", [p["drift_g"] for p in pairs]), ("off-basin, sep >= 8 cm", [p["off_g"] for p in pairs]),
                     ("off-basin in cm", [p["off_cm"] for p in pairs]), ("drift in cm", [p["drift_cm"] for p in pairs]),
                     ("drift, whole rollout", [p["drift_all"] for p in pairs]), ("-sep_min", [-p["sep_min"] for p in pairs]),
                     ("t0 sigma1 (ridge)", [p["sigma1"] for p in pairs]), ("t0 |lam0 - 1/2|^-1", [-abs(p["lam0"] - 0.5) for p in pairs])):
        lo, hi = boot(sc, y, grp)
        within = [auroc([s_ for s_, p in zip(sc, pairs) if p["init"] == i], [p["split"] for p in pairs if p["init"] == i]) for i in range(4)]
        sub = [(s_, p["split"]) for s_, p in zip(sc, pairs) if p["kind"] in ("underspecified", "novel")]
        print(f"  {name:24s}: AUROC {auroc(sc, y):.2f} [{lo:.2f}, {hi:.2f}]  within init " + " ".join(f"{a:.2f}" for a in within)
              + f"  within underspecified/novel {auroc([a for a, _ in sub], [b for _, b in sub]):.2f}")
    print("\nrollout-level, split pairs only: does late lambda (mean of the last 3 replans) say which outcome?")
    per = []
    for p in pairs:
        if not p["split"]:
            continue
        maj = max(set(p["labs"]), key=p["labs"].count)
        yy = [l == maj for l in p["labs"]]; sc = [f["lam_late"] for f in p["F"]]
        if 0 < sum(yy) < len(yy):
            a = auroc(sc, yy); per.append(max(a, 1 - a))
            print(f"  init {p['init']} '{p['text']}': labels {p['labs']}  lambda_late " + " ".join(f"{x:.2f}" for x in sc) + f"  |AUROC - 1/2| + 1/2 = {per[-1]:.2f}")
    print(f"  mean over pairs {np.mean(per):.2f}" if per else "  none")
    print("\nlead time, split pairs: first replan where lambda moved >= 0.3 from its t 0 value vs the outcome time")
    for p in pairs:
        if p["split"]:
            print(f"  init {p['init']} '{p['text']}': c+ '{p['cplus']}' c- '{p['cminus']}'  " +
                  "  ".join(f"{l}@{t_} warn@{f['t_warn']}" for l, t_, f in zip(p["labs"], p["ts"], p["F"])))
    print("\ndecoded prompts (underspecified / novel):")
    for p in sorted(pairs, key=lambda p: (p["kind"], p["text"], p["init"])):
        if p["kind"] in ("underspecified", "novel"):
            print(f"  init {p['init']} {'SPLIT' if p['split'] else '     '} '{p['text']}': + '{p['cplus']}' / - '{p['cminus']}'  drift {p['drift']:.2f}  labels {sorted(set(p['labs']))}")


if __name__ == "__main__" and sys.argv[1] in ("decode", "run", "report"):
    {"decode": cmd_decode, "run": run, "report": report}[sys.argv[1]](*sys.argv[2:])


# ----------------------------------------------------------------------------------------------
# v2: wider decode (insertions, second greedy round) and the ridge test online, in cm.
# ----------------------------------------------------------------------------------------------
ALPHAS_ON = [0.025, 0.05, 0.1]          # online ridge test: jump along +-v1 within these fractions of the token norm
SHORT_SUB, SHORT_INS = 8, 4
D_INFO = 15.0                           # cm: a suggestion must move the plan this far from "the slot deleted" (junk words act as deletion)
T_SCAN, N_SCAN = 60, 2                  # ridge test at replans t <= T_SCAN, for the first N_SCAN seeds


def is_step(jump):
    """A ridge, not a ramp: the plan jumps >= D_JUMP within 2.5% of the token norm and is saturated by 10%."""
    return jump[0] >= D_JUMP and jump[0] >= 0.6 * jump[-1]


def plan_texts(f, o, texts, zp, bs=16):
    """Mean-over-z plans [n, 150] (cm) for prompts `texts` at observation o."""
    import torch
    from pi05flow import cexpand, cslice
    from pridge5 import path50_t
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            tx = texts[i:i + bs]
            c = f.prefix_from(*f.embed([o] * len(tx), tx))
            for k in range(len(tx)):
                a = f.flow(f._z(zp), cexpand(cslice(c, k, k + 1), len(zp)))[..., :f.A]
                out.append(torch.stack([path50_t(f, a[j]) for j in range(len(zp))]).mean(0))
            del c
    return torch.stack(out)


def ridge_sides(f, E, pad, att, slots, zp, alphas, n_iter=4):
    """v1, sigma1, and the plans on both sides of the ridge along v1 at each alpha: [len(alphas), 2, 150]."""
    import torch
    from pridge5 import plans, top_dir_E
    v1, s1, r = top_dir_E(f, E, pad, att, slots, zp, n_iter=n_iter)
    Es = []
    for a in alphas:
        for sgn in (1, -1):
            Ep = E.clone(); Ep[0, slots] += sgn * a * r * v1; Es.append(Ep)
    P = plans(f, torch.cat(Es), pad, att, zp).mean(1).view(len(alphas), 2, -1)
    return v1, s1, r, P


def propose(f, o, text, zp, voc, dhat, signs=("+", "-")):
    """Single-token edits of `text` ranked by the first-order change of <plan, dhat>: substitutions
    at every prompt slot and one-token insertions before every slot and after the last."""
    from pridge5 import prompt_slots
    pieces, ok, Evoc = voc
    E, pad, att = f.embed([o], [text])
    slots, pcs = prompt_slots(f, o, text)
    g = grad_E(f, E, pad, att, zp, dhat)
    cands = {}
    for i, (k, pc) in enumerate(zip(slots, pcs)):
        sc = ((Evoc - E[0, k]) @ g[k]).double().cpu().numpy()          # substitution
        sc[~ok] = np.nan
        if pc in pieces:
            sc[pieces.index(pc)] = np.nan
        si = (Evoc @ g[k]).double().cpu().numpy(); si[~ok] = np.nan      # insertion before slot i (gradient of the neighbour)
        order, order_i = np.argsort(np.nan_to_num(sc, nan=0.0)), np.argsort(np.nan_to_num(si, nan=0.0))
        for sign in signs:
            vs = order[-SHORT_SUB:][::-1] if sign == "+" else order[:SHORT_SUB]
            for v in vs:
                if not np.isnan(sc[v]):
                    new = pcs.copy(); new[i] = pieces[v]
                    cands.setdefault("".join(new).replace("▁", " ").strip(), dict(kind="sub", slot=pc[1:], word=pieces[v][1:], sign=sign, fo=float(sc[v])))
            vs = order_i[-SHORT_INS:][::-1] if sign == "+" else order_i[:SHORT_INS]
            for v in vs:
                if not np.isnan(si[v]):
                    new = pcs[:i] + [pieces[v]] + pcs[i:]
                    cands.setdefault("".join(new).replace("▁", " ").strip(), dict(kind="ins", slot=pc[1:], word=pieces[v][1:], sign=sign, fo=float(si[v])))
                    if i == len(slots) - 1:
                        new = pcs + [pieces[v]]
                        cands.setdefault("".join(new).replace("▁", " ").strip(), dict(kind="ins_end", slot=pc[1:], word=pieces[v][1:], sign=sign, fo=float(si[v])))
        if len(slots) > 1:                                                 # the slot deleted: the "no word here" reference
            new = pcs[:i] + pcs[i + 1:]
            cands.setdefault("".join(new).replace("▁", " ").strip(), dict(kind="del", slot=pc[1:], word="", sign="0", fo=0.0))
    cands.pop(text, None)
    return cands


def decode2(f, o, text, zp, voc, sides=None, rounds=2, improve=0.8, ridge_only=False):
    """Candidate-free decode with a wider proposal set and a second greedy round. `sides`: (P+, P-)
    plans to decode towards (default: the two sides of the ridge at this observation)."""
    import torch
    from pridge5 import prompt_slots
    t0 = time.time()
    E, pad, att = f.embed([o], [text])
    slots, pcs = prompt_slots(f, o, text)
    v1 = r = None
    if sides is None:
        v1, s1, r, P = ridge_sides(f, E, pad, att, slots, zp, ALPHAS)
        side = (P[:, 0] - P[:, 1]).norm(dim=1).double().cpu().numpy()
        j = next((k for k, x in enumerate(side) if x >= D_JUMP), ALPHAS.index(ALPHA_DEFAULT))
        Pp, Pm = P[j, 0], P[j, 1]
        info = dict(sigma1=s1, alpha=ALPHAS[j], side=side.tolist(), step=bool(is_step(side[:3])))
        if ridge_only and not info["step"]:
            return dict(text=text, tokens=[p[1:] for p in pcs], sep=float((Pp - Pm).norm()), lam0=float("nan"), cplus=None, cminus=None,
                        toward_plus=0.0, toward_minus=0.0, n_cands=0, decode_s=time.time() - t0, top=[], bottom=[], **info), (v1, r)
    else:
        Pp, Pm = sides; info = {}
    d = Pp - Pm; sep = float(d.norm()); dhat = d / d.norm()

    def toward(texts):
        Pc = plan_texts(f, o, texts, zp)
        return (((Pc - Pm).norm(dim=1) - (Pc - Pp).norm(dim=1)) / sep).double().cpu().numpy(), Pc

    P0 = plan_texts(f, o, [text], zp)[0]
    best = {"+": (text, 0.0), "-": (text, 0.0)}
    log_rows = []
    for rd in range(rounds):
        for sign in ("+", "-"):
            cur, tw0 = best[sign]
            if rd > 0 and abs(tw0) >= improve:
                continue
            cands = propose(f, o, cur, zp, voc, dhat if sign == "+" else -dhat, signs=("+",))
            if not cands:
                continue
            texts = list(cands)
            tw, Pc = toward(texts)
            tw = tw * (1 if sign == "+" else -1)
            ref = {c["slot"]: Pc[k] for k, (tx, c) in enumerate(cands.items()) if c["kind"] == "del"}
            Pcur = P0 if cur == text else plan_texts(f, o, [cur], zp)[0]
            inf = np.array([float((Pc[k] - (ref.get(c["slot"], Pcur) if c["kind"] == "sub" else Pcur)).norm()) for k, c in enumerate(cands.values())])
            valid = np.array([c["kind"] != "del" and inf[k] >= D_INFO for k, c in enumerate(cands.values())])
            log_rows += [dict(round=rd, side=sign, text=tx, toward=float(t_ * (1 if sign == "+" else -1)), info=float(x), **cands[tx])
                         for tx, t_, x in zip(texts, tw, inf)]
            if valid.any():
                k = int(np.argmax(np.where(valid, tw, -np.inf)))
                if tw[k] > tw0:
                    best[sign] = (texts[k], float(tw[k]))
    lam0 = float(((plan_texts(f, o, [text], zp)[0] - Pm) @ d) / (d @ d))
    rows = sorted(log_rows, key=lambda c: -c["toward"])
    out = dict(text=text, tokens=[p[1:] for p in pcs], sep=sep, lam0=lam0,
               cplus=best["+"][0] if best["+"][0] != text else None, cminus=best["-"][0] if best["-"][0] != text else None,
               toward_plus=best["+"][1], toward_minus=-best["-"][1], n_cands=len(log_rows), decode_s=time.time() - t0,
               top=rows[:8], bottom=rows[-8:], **info)
    return (out, (v1, r)) if ridge_only else out


def cmd_decode2(init=3, text="open a drawer"):
    import pi05flow
    from env5 import Scene
    f = pi05flow.load(); voc = dictionary(f)
    zp = np.random.default_rng(777).standard_normal((N_Z, pi05flow.H * pi05flow.D_ACT))
    s = Scene(0, int(init), 0, suite_name=SUITE); obs = s.reset()
    o = s.obs_to_lerobot(obs, text)
    r, _ = decode2(f, o, text, zp, voc, ridge_only=True)
    log(f"init {init} '{text}': sigma1 {r['sigma1']:.1f}, |P+ - P-| {r['sep']:.0f} cm at alpha {r['alpha']} (side " +
        " ".join(f"{x:.0f}" for x in r["side"]) + f"), lambda0 {r['lam0']:.2f}; {r['n_cands']} candidates, {r['decode_s']:.0f}s")
    log(f"  c+ '{r['cplus']}' ({r['toward_plus']:+.2f})   c- '{r['cminus']}' ({r['toward_minus']:+.2f})")
    for c in r["top"] + [None] + r["bottom"]:
        print("    ..." if c is None else f"    {c['toward']:+.2f} info {c['info']:5.1f} r{c['round']}{c['side']} {c['kind']:7s} [{c['slot']}] {c['text']}")
    s.close()


def rollout2(f, s, x0, text, words0, seed, max_t, zp, voc, scan, v1r=None):
    """Execute `text` closed loop; at every replan record cm distances between the executed plan and
    the plans under the t 0 words (and, once found, the online words); if `scan`, at replans
    t <= T_SCAN run the ridge test (sigma1, jump along +-v1 in cm) and decode at the first ridge."""
    from pi05flow import H, D_ACT
    from env5 import GOAL_NAMES, EVENT_NAMES, TERMINAL
    from goal5 import path_cm
    from pridge5 import prompt_slots
    names = GOAL_NAMES + EVENT_NAMES
    obs = s.set_state(x0)
    rng = np.random.default_rng(seed)
    t, first, trace, label, words_on = 0, {}, [], None, None
    while t < max_t and label is None:
        o = s.obs_to_lerobot(obs, text)
        z = rng.standard_normal((1, H * D_ACT))
        rec = dict(t=t)
        for tag, w in (("t0", words0), ("on", words_on)):
            if w is None:
                continue
            p = {k: path_cm(f.unnorm(f.flow_np(z, f.prefix([o], [k]))[0]))[:50].flatten() for k in (text, w[0], w[1])}
            rec[tag] = dict(dp=float(np.linalg.norm(p[text] - p[w[0]])), dm=float(np.linalg.norm(p[text] - p[w[1]])),
                            sep=float(np.linalg.norm(p[w[0]] - p[w[1]])))
        if scan and t <= T_SCAN and v1r is not None and v1r[0] is not None:
            import torch
            from pridge5 import plans
            v1, r = v1r
            E, pad, att = f.embed([o], [text])
            slots, _ = prompt_slots(f, o, text)
            Es = []
            for a_ in ALPHAS_ON:
                for sgn in (1, -1):
                    Ep = E.clone(); Ep[0, slots] += sgn * a_ * r * v1; Es.append(Ep)
            P = plans(f, torch.cat(Es), pad, att, zp).mean(1).view(len(ALPHAS_ON), 2, -1)
            jump = (P[:, 0] - P[:, 1]).norm(dim=1).double().cpu().numpy()
            rec["ridge"] = dict(jump=jump.tolist(), step=bool(is_step(jump)))
            if words_on is None and t > 0 and is_step(jump):
                d2 = decode2(f, o, text, zp, voc, sides=(P[0, 0], P[0, 1]))
                rec["decode"] = dict(cplus=d2["cplus"], cminus=d2["cminus"], toward_plus=d2["toward_plus"],
                                     toward_minus=d2["toward_minus"], sep=d2["sep"], decode_s=d2["decode_s"])
                if d2["cplus"] and d2["cminus"]:
                    words_on = (d2["cplus"], d2["cminus"])
        trace.append(rec)
        a = f.unnorm(f.flow_np(z, f.prefix([o], [text]))[0])
        for j in range(10):
            obs, _, _ = s.step(a[j]); t += 1
            for i, q in enumerate(s.preds()):
                if q and names[i] not in first:
                    first[names[i]] = t
                    if TERMINAL[i]:
                        label = names[i]
            if label is not None or t >= max_t:
                break
    if label is None:
        opened = [(tt, n) for n, tt in first.items()]
        label = max(opened)[1] if opened else "none"
    return dict(seed=seed, label=label, t=t, events=sorted(first.items(), key=lambda kv: kv[1]), words_on=words_on, trace=trace)


def run2(n_init=4, n_seed=4, only=None, inits=None):
    """only: substring of the prompt; inits: comma-separated initial states -> out/monitor2_<only>.json"""
    import pi05flow
    from env5 import Scene
    from goal5 import PROMPTS
    PROMPTS = [p for p in PROMPTS if only is None or only in p[0]]
    init_list = [int(x) for x in inits.split(",")] if inits else list(range(int(n_init)))
    path = os.path.join(OUT, "monitor2_run.json")
    R = json.load(open(path)) if os.path.exists(path) else {}
    f = pi05flow.load(); voc = dictionary(f)
    zp = np.random.default_rng(777).standard_normal((N_Z, pi05flow.H * pi05flow.D_ACT))
    V1 = {}
    for init in init_list:
        todo = [p for p in PROMPTS if len(R.get(f"{p[0]}|{init}", {}).get("rollouts", [])) < int(n_seed)]
        if not todo:
            continue
        s = Scene(0, init, 0, suite_name=SUITE); obs = s.reset(); x0 = s.get_state()
        for text, kind, _ in todo:
            key = f"{text}|{init}"
            o = s.obs_to_lerobot(obs, text)
            d, V1[key] = decode2(f, o, text, zp, voc, ridge_only=True)
            if key not in R:
                R[key] = dict(d, kind=kind, init=init, rollouts=[])
                log(f"init {init} '{text}' ({kind}): sigma1 {d['sigma1']:.1f}, jump " + " ".join(f"{x:.0f}" for x in d["side"][:3]) +
                    (f" cm STEP; c+ '{d['cplus']}' ({d['toward_plus']:+.2f}) c- '{d['cminus']}' ({d['toward_minus']:+.2f})" if d["step"] else " cm, no ridge") +
                    f"  ({d['decode_s']:.0f}s)")
            d = R[key]
            for seed in range(len(d["rollouts"]), int(n_seed)):
                t0 = time.time()
                w0 = (d["cplus"], d["cminus"]) if d["cplus"] and d["cminus"] else None
                ro = rollout2(f, s, x0, text, w0, 1000 + seed, s.max_steps, zp, voc, scan=seed < N_SCAN, v1r=V1[key])
                d["rollouts"].append(ro)
                rid = [tr for tr in ro["trace"] if "ridge" in tr]
                dec = next((tr for tr in ro["trace"] if "decode" in tr), None)
                log(f"    seed {seed}: {ro['label']}@{ro['t']}  jump(cm) " + " ".join(f"{tr['ridge']['jump'][0]:.0f}{'*' if tr['ridge']['step'] else ''}" for tr in rid) +
                    (f"  decoded@{dec['t']}: '{dec['decode']['cplus']}' / '{dec['decode']['cminus']}' ({dec['decode']['toward_plus']:+.2f}/{dec['decode']['toward_minus']:+.2f})" if dec else "") +
                    f"  ({time.time() - t0:.0f}s)")
            json.dump(R, open(path, "w"))
        s.close()


if __name__ == "__main__" and sys.argv[1] in ("decode2", "run2"):
    {"decode2": cmd_decode2, "run2": run2}[sys.argv[1]](*sys.argv[2:])


def feats2(ro, early=T_SCAN):
    """Per-rollout online features (cm) from a rollout2 trace."""
    tr = ro["trace"]
    rid = [(x["t"], x["ridge"]) for x in tr if "ridge" in x and x["t"] <= early]
    jmax = max((r["jump"][0] for _, r in rid), default=float("nan"))
    steps = [t for t, r in rid if r["step"] and t > 0]
    out = dict(jump_max=jmax, step=bool(steps), t_step=steps[0] if steps else -1)
    for tag in ("t0", "on"):
        rows = [(x["t"], x[tag]) for x in tr if tag in x and x["t"] <= early]
        if rows:
            diff = np.array([c["dm"] - c["dp"] for _, c in rows]); off = np.array([min(c["dp"], c["dm"]) for _, c in rows])
            out[f"drift_{tag}"] = float(diff.max() - diff.min()); out[f"off_{tag}"] = float(off.max())
            out[f"sep_{tag}"] = float(np.mean([c["sep"] for _, c in rows]))
    return out


def report2():
    from screen5 import auroc, boot
    R = json.load(open(os.path.join(OUT, "monitor2_run.json")))
    G = json.load(open(os.path.join(OUT, "goal_run.json")))
    gl = defaultdict(set)
    for r in G:
        gl[(r["text"], r["init"])].add(r["label"])
    pairs = []
    for key, d in R.items():
        if not d["rollouts"]:
            continue
        labs = [ro["label"] for ro in d["rollouts"]]
        F = [feats2(ro) for ro in d["rollouts"] if any("ridge" in x for x in ro["trace"])]
        real = {l for l in set(labs) | gl.get((d["text"], d["init"]), set()) if l != "none"}
        on = [x["decode"] for ro in d["rollouts"] for x in ro["trace"] if "decode" in x]
        pairs.append(dict(text=d["text"], init=d["init"], kind=d["kind"], labs=labs, ts=[ro["t"] for ro in d["rollouts"]],
                          split4=len(set(labs)) > 1, mode12=len(real) > 1, step0=d["step"], jump0=d["side"][0], sigma1=d["sigma1"],
                          cplus=d["cplus"], cminus=d["cminus"], on=on, F=F,
                          jump_max=max(f["jump_max"] for f in F), step=any(f["step"] for f in F),
                          t_step=min([f["t_step"] for f in F if f["t_step"] >= 0], default=-1),
                          drift=max([f.get("drift_on", f.get("drift_t0", 0.0)) for f in F], default=0.0),
                          off=max([f.get("off_on", f.get("off_t0", 0.0)) for f in F], default=0.0)))
    print(f"{len(pairs)} (prompt, init) pairs; split in this run's 4 seeds: {sum(p['split4'] for p in pairs)}; "
          f"two real modes among these + goal5's 8 seeds: {sum(p['mode12'] for p in pairs)}")
    print(f"ridge (step) at t 0: {sum(p['step0'] for p in pairs)} pairs; a step at some later replan <= {T_SCAN}: {sum(p['step'] for p in pairs)} pairs\n")
    for lab in ("split4", "mode12"):
        y = [p[lab] for p in pairs]; grp = [p["init"] for p in pairs]
        print(f"pair-level AUROC vs {lab}:")
        for name, sc in (("t0 jump along v1 (cm)", [p["jump0"] for p in pairs]), ("t0 sigma1", [p["sigma1"] for p in pairs]),
                         ("t0 step (0/1)", [float(p["step0"]) for p in pairs]),
                         ("online max jump (cm)", [p["jump_max"] for p in pairs]), ("online step any (0/1)", [float(p["step"]) for p in pairs]),
                         ("online drift (cm, words)", [p["drift"] for p in pairs]), ("online off-basin (cm, words)", [p["off"] for p in pairs])):
            lo, hi = boot(sc, y, grp)
            sub = [(s_, p[lab]) for s_, p in zip(sc, pairs) if p["kind"] in ("underspecified", "novel")]
            within = [auroc([s_ for s_, p in zip(sc, pairs) if p["init"] == i], [p[lab] for p in pairs if p["init"] == i]) for i in range(4)]
            print(f"  {name:28s} AUROC {auroc(sc, y):.2f} [{lo:.2f}, {hi:.2f}]  within init " + " ".join(f"{a:.2f}" for a in within)
                  + f"  within underspecified/novel {auroc([a for a, _ in sub], [b for _, b in sub]):.2f}")
        tp = sum(p["step"] and p[lab] for p in pairs); fp = sum(p["step"] and not p[lab] for p in pairs); fn = sum((not p["step"]) and p[lab] for p in pairs)
        print(f"  online step as a detector: precision {tp / max(tp + fp, 1):.2f} recall {tp / max(tp + fn, 1):.2f} (TP {tp}, FP {fp}, FN {fn})\n")
    print("pairs with a step online (first step time, outcome times, decoded words):")
    for p in sorted(pairs, key=lambda p: (p["init"], p["kind"], p["text"])):
        if p["step"] or p["step0"]:
            w = p["on"][0] if p["on"] else None
            print(f"  init {p['init']} {'SPLIT' if p['mode12'] else '     '} '{p['text']}' ({p['kind']}): t0 {'STEP' if p['step0'] else 'ramp'} "
                  f"{p['cplus']!r}/{p['cminus']!r}; online step@{p['t_step']} " + (f"{w['cplus']!r}/{w['cminus']!r} ({w['toward_plus']:+.2f}/{w['toward_minus']:+.2f})" if w else "-")
                  + f"; outcomes " + ", ".join(f"{l}@{t}" for l, t in zip(p["labs"], p["ts"])))
    print("\nsplit pairs without any step (missed):")
    for p in pairs:
        if p["mode12"] and not p["step"] and not p["step0"]:
            print(f"  init {p['init']} '{p['text']}' ({p['kind']}): max online jump {p['jump_max']:.0f} cm; outcomes {sorted(set(p['labs']))}")


if __name__ == "__main__" and sys.argv[1] == "report2":
    report2()
