"""Decode the mode-separating direction of 'open a drawer' (init 3) back into prompt edits.

At a state s (t 0, and t 10 of the seed-0 rollout where lambda ~ 0.5):
  1. modes: N closed-loop branches (fresh noise, batch-1 prefix) to horizon T; cluster the eef
     trajectories (label-free, 2 clusters) -> Delta_branch = mean eef path A - B over the first
     50 steps (the plan horizon).  Counterfactual reference: Delta_cf = plan(TOP) - plan(MID).
  2. direction in prompt space: g = d <plan50, Delta> / dE for the prompt tokens (one backward,
     averaged over n_z seeds).
  3. decode: first-order score of every vocabulary word at each prompt position,
     <g_i, emb(w) - E_i>; top words per sign; verify by running the policy on the edited text
     (projection of the plan shift on Delta, cm, and lambda between the MID/TOP plans).
"""
import json, os, re, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/james/Projects/flow-matching-lcs/robot5")
import pi05flow
from pi05flow import H, D_ACT, cexpand
from env5 import Scene, GOAL_NAMES, EVENT_NAMES
from order5 import OSC, GAIN

NAMES = GOAL_NAMES + EVENT_NAMES
TEXT, INIT = "open a drawer", 3
MID, TOP = "open the middle drawer of the cabinet", "open the top drawer and put the bowl inside"
N_BR, HORIZON, N_Z, TOPK = 16, 60, 4, 12
A = H * D_ACT
OUTJ = "/home/james/Projects/flow-matching-lcs/robot5/out/decode_open_a_drawer_3.json"
t00 = time.time()


def log(*a):
    print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)


def path50_t(f, a):
    """normalised chunk [H, A] (torch) -> predicted eef path over the whole chunk [H*3] cm."""
    sc = torch.as_tensor(f.a_scale[:3], dtype=a.dtype, device=a.device)
    sh = torch.as_tensor(f.a_shift[:3], dtype=a.dtype, device=a.device)
    return (100 * GAIN * OSC * torch.cumsum(a[:, :3] * sc + sh, 0)).flatten()


def plans(f, E, pad, att, zp):
    with torch.no_grad():
        a = f.flow(f._z(zp), cexpand(f.prefix_from(E, pad, att), len(zp)))[..., :f.A]
        return torch.stack([path50_t(f, a[i]) for i in range(len(zp))])


def grad_E(f, E, pad, att, zp, d):
    """gradient of mean_z <plan50(z; E), d> wrt E  ->  [P, W]"""
    Eg = E.detach().requires_grad_(True)
    with torch.enable_grad():
        a = f.flow(f._z(zp), cexpand(f.prefix_from(Eg, pad, att), len(zp)))[..., :f.A]
        obj = torch.stack([path50_t(f, a[i]) for i in range(len(zp))]).mean(0) @ d
        g, = torch.autograd.grad(obj, Eg)
    return g[0]


def branch(s, x, text, f, n, horizon, seed0=100):
    """n closed-loop branches from state x (fresh noise each), eef trajectories [n, horizon+1, 3]
    relative to the start, and first-event labels (reporting only)."""
    trajs, labs = [], []
    for b in range(n):
        obs = s.set_state(x)
        rng = np.random.default_rng(seed0 + b)
        e0 = s.eef().copy(); tr = [np.zeros(3)]; first = {}
        t = 0
        while t < horizon:
            o = s.obs_to_lerobot(obs, text)
            a = f.unnorm(f.flow_np(rng.standard_normal((1, A)), f.prefix([o], [text]))[0])
            for j in range(10):
                obs, _, _ = s.step(a[j]); t += 1
                tr.append(s.eef() - e0)
                for i, q in enumerate(s.preds()):
                    if q and NAMES[i] not in first:
                        first[NAMES[i]] = t
                if t >= horizon:
                    break
        trajs.append(np.array(tr)); labs.append(first)
    return np.stack(trajs), labs


def two_clusters(T):
    """average-linkage, 2 clusters on the eef trajectories [n, T, 3]"""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    X = T.reshape(len(T), -1)
    lab = fcluster(linkage(pdist(X), "average"), 2, "maxclust") - 1
    return lab


def main():
    f = pi05flow.load()
    sp = pi05flow.SPTokenizer().sp
    emb = f.m.paligemma_with_expert.paligemma.model.language_model.get_input_embeddings()
    V = sp.get_piece_size()
    pieces = [sp.id_to_piece(i) for i in range(V)]
    word_ok = np.array([re.fullmatch(r"▁[a-z]{2,}", p) is not None for p in pieces])
    log(f"vocab {V}, readable words {word_ok.sum()}")
    with torch.no_grad():
        Evoc = torch.cat([emb(torch.arange(i, min(i + 32768, V), device=f.dev)) for i in range(0, V, 32768)])
    log(f"embedding table {tuple(Evoc.shape)} scale {float(getattr(emb, 'embed_scale', 1.0)):.2f}")

    s = Scene(0, INIT, 0, suite_name="libero_goal")
    obs = s.reset()
    rng_main = np.random.default_rng(0)
    results = []
    for state_k in range(2):
        t = 10 * state_k
        x = s.get_state()
        o = s.obs_to_lerobot(obs, TEXT)
        E, pad, att = f.embed([o], [TEXT])
        P = E.shape[1]
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
        ids = f.batch([o], [TEXT])[OBS_LANGUAGE_TOKENS][0].cpu().numpy()
        n_img = sum(n * n for n in f.n_img)
        pcs = [pieces[i] for i in ids[: int(pad[0, n_img:].sum())]]
        words = TEXT.split()
        pos = []
        for w in words:
            for j, p in enumerate(pcs):
                if p.lstrip("▁") == w and (n_img + j) not in pos and j > 1:
                    pos.append(n_img + j); break
        log(f"state t {t}: P {P}, text tokens {len(pcs)}: {pcs[:12]} ..., prompt positions {pos}")
        assert len(pos) == len(words)
        zp = np.random.default_rng(777).standard_normal((N_Z, A))

        # modes by branching
        T, labs = branch(s, x, TEXT, f, N_BR, HORIZON)
        cl = two_clusters(T)
        log(f"branches: cluster sizes {np.bincount(cl).tolist()}; labels per cluster: " +
            " | ".join(f"{c}: {[sorted(l.items(), key=lambda kv: kv[1])[:1] for l, cc in zip(labs, cl) if cc == c]}" for c in range(2)))
        # cluster 0 = the one the counterfactual TOP plan is closer to (naming only)
        mA, mB = T[cl == 0, 1:H + 1].mean(0).flatten(), T[cl == 1, 1:H + 1].mean(0).flatten()
        d_br = torch.as_tensor(mA - mB, dtype=torch.float32, device=f.dev)
        s.set_state(x)
        # counterfactual reference
        with torch.no_grad():
            Em, pm, am = f.embed([o], [MID]); Et, pt, at = f.embed([o], [TOP])
        p0, pM, pT = plans(f, E, pad, att, zp), plans(f, Em, pm, am, zp), plans(f, Et, pt, at, zp)
        d_cf = (pT - pM).mean(0)
        lam = lambda p: float((((p - pM) * (pT - pM)).sum(1) / (pT - pM).pow(2).sum(1)).mean())  # noqa: E731
        cos = float(d_br @ d_cf / (d_br.norm() * d_cf.norm()))
        log(f"|Delta_branch| {d_br.norm():.1f} cm, |Delta_cf| {d_cf.norm():.1f} cm, cos {cos:.2f}; lambda(plan) {lam(p0):.2f}")
        if cos < 0:
            d_br = -d_br; mA, mB = mB, mA; cl = 1 - cl
        res = dict(t=t, cluster_sizes=np.bincount(cl).tolist(), cos_branch_cf=cos, lam0=lam(p0),
                   labels=[sorted(l.items(), key=lambda kv: kv[1]) for l in labs], cl=cl.tolist(), per_delta={})
        for dname, d in (("branch", d_br), ("cf", d_cf)):
            dh = d / d.norm()
            g = grad_E(f, E, pad, att, zp, dh)                                   # [P, W]
            gn = g.norm(dim=1).double().cpu().numpy()
            blk = dict(images=float(gn[:n_img].sum()), prompt=float(gn[pos].sum()),
                       other_text=float(gn[n_img:].sum() - gn[pos].sum()))
            log(f"  Delta_{dname}: gradient norm per block {blk}; per prompt token " +
                " ".join(f"{w}:{gn[p]:.3f}" for w, p in zip(words, pos)))
            cands = []
            for w, p in zip(words, pos):
                sc = ((Evoc - E[0, p]) @ g[p]).double().cpu().numpy()
                sc[~word_ok] = np.nan; sc[ids[p - n_img]] = np.nan
                order = np.argsort(np.nan_to_num(sc, nan=0))
                for v in list(order[-TOPK:][::-1]):
                    cands.append(dict(pos=w, word=pieces[v][1:], fo=float(sc[v]), kind="sub"))
                for v in list(order[:TOPK]):
                    cands.append(dict(pos=w, word=pieces[v][1:], fo=float(sc[v]), kind="sub"))
            # verify every candidate (substitution, plus insertion before 'drawer' for the words found)
            texts = {}
            for c in cands:
                ws = words.copy(); ws[words.index(c["pos"])] = c["word"]
                texts[" ".join(ws)] = c
                if c["pos"] != "open":
                    texts[f"open a {c['word']} drawer"] = dict(c, kind="ins")
                    texts[f"open the {c['word']} drawer"] = dict(c, kind="ins-the")
            texts[MID] = dict(pos="-", word="-", fo=float("nan"), kind="ref"); texts[TOP] = dict(pos="-", word="-", fo=float("nan"), kind="ref")
            rows = []
            for txt, c in texts.items():
                with torch.no_grad():
                    Ex, px, ax = f.embed([o], [txt])
                px_ = plans(f, Ex, px, ax, zp)
                shift = float(((px_ - p0) @ dh).mean()); move = float((px_ - p0).norm(dim=1).mean())
                rows.append(dict(text=txt, kind=c["kind"], pos=c["pos"], fo=c["fo"], shift=shift, move=move, lam=lam(px_)))
            rows.sort(key=lambda r: -r["shift"])
            res["per_delta"][dname] = dict(grad_blocks=blk, rows=rows)
            fo = np.array([r["fo"] for r in rows if r["kind"] == "sub"]); sh = np.array([r["shift"] for r in rows if r["kind"] == "sub"])
            log(f"  Delta_{dname}: {len(rows)} verified; first-order vs actual shift (substitutions): "
                f"Spearman {float(np.corrcoef(np.argsort(np.argsort(fo)), np.argsort(np.argsort(sh)))[0, 1]):.2f}")
            print(f"    {'shift cm':>9} {'move':>6} {'lambda':>6} {'f.o.':>8}  text")
            for r in rows[:14] + [dict(text="...", kind="", shift=float("nan"), move=float("nan"), lam=float("nan"), fo=float("nan"))] + rows[-14:]:
                print(f"    {r['shift']:9.2f} {r['move']:6.2f} {r['lam']:6.2f} {r['fo']:8.2f}  {r['text']} [{r['kind']}]")
            print(flush=True)
        results.append(res)
        json.dump(results, open(OUTJ, "w"))
        # advance the main rollout (seed 0) by one chunk
        s.set_state(x)
        a = f.unnorm(f.flow_np(rng_main.standard_normal((1, A)), f.prefix([o], [TEXT]))[0])
        for j in range(10):
            obs, _, _ = s.step(a[j])
    s.close()


if __name__ == "__main__":
    main()
