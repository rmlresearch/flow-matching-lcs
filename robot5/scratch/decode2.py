"""Follow-up to decode.py: (1) where do the drawer words rank first-order? (2) a wider pool
(top POOL words per sign per position) verified in batches, ranked by shift with an alignment
filter (plan moves along Delta, not off-distribution). States t 0 and t 10 as before; direction
= branch clusters (candidate-free), sign fixed by the counterfactual only for naming."""
import json, re, sys, time
import numpy as np, torch
sys.path.insert(0, "/tmp/claude-1000/-home-james-Projects-flow-matching-lcs/fc615ee8-3dcc-4cb5-8739-f0992e1fde17/scratchpad")
sys.path.insert(0, "/home/james/Projects/flow-matching-lcs/robot5")
import pi05flow
from pi05flow import cexpand
from env5 import Scene
from decode import (A, TEXT, INIT, MID, TOP, N_BR, HORIZON, N_Z, plans, grad_E, branch, two_clusters, log)

POOL, ALIGN = 150, 0.5
KEY = ["top", "middle", "bottom", "lower", "upper", "second", "cabinet", "bowl", "open", "close"]
OUTJ = "/home/james/Projects/flow-matching-lcs/robot5/out/decode2_open_a_drawer_3.json"


def main():
    f = pi05flow.load()
    sp = pi05flow.SPTokenizer().sp
    emb = f.m.paligemma_with_expert.paligemma.model.language_model.get_input_embeddings()
    V = sp.get_piece_size()
    pieces = [sp.id_to_piece(i) for i in range(V)]
    word_ok = np.array([re.fullmatch(r"▁[a-z]{2,}", p) is not None for p in pieces])
    wid = {pieces[i][1:]: i for i in np.where(word_ok)[0]}
    with torch.no_grad():
        Evoc = torch.cat([emb(torch.arange(i, min(i + 32768, V), device=f.dev)) for i in range(0, V, 32768)])
    s = Scene(0, INIT, 0, suite_name="libero_goal")
    obs = s.reset()
    rng_main = np.random.default_rng(0)
    results = []
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
    for state_k in range(2):
        t = 10 * state_k
        x = s.get_state()
        o = s.obs_to_lerobot(obs, TEXT)
        E, pad, att = f.embed([o], [TEXT])
        ids = f.batch([o], [TEXT])[OBS_LANGUAGE_TOKENS][0].cpu().numpy()
        n_img = sum(n * n for n in f.n_img)
        words = TEXT.split(); pos = [n_img + 3, n_img + 4, n_img + 5]
        zp = np.random.default_rng(777).standard_normal((N_Z, A))
        T, labs = branch(s, x, TEXT, f, N_BR, HORIZON)
        cl = two_clusters(T[:, -1:])                      # where the branches are at the horizon (phase-robust)
        log(f"  clusters by final eef: sizes {np.bincount(cl).tolist()}, centres (cm) "
            f"{np.round(100 * T[cl == 0, -1].mean(0), 1).tolist()} vs {np.round(100 * T[cl == 1, -1].mean(0), 1).tolist()}; "
            f"events {[sorted(l.items(), key=lambda kv: kv[1])[:1] for l in labs]}")
        np.save(OUTJ.replace(".json", f"_T{t}.npy"), T)
        d = torch.as_tensor(100 * (T[cl == 0, 1:51].mean(0) - T[cl == 1, 1:51].mean(0)).flatten(), dtype=torch.float32, device=f.dev)
        obs = s.set_state(x)
        with torch.no_grad():
            Em, pm, am = f.embed([o], [MID]); Et, pt, at = f.embed([o], [TOP])
        p0, pM, pT = plans(f, E, pad, att, zp), plans(f, Em, pm, am, zp), plans(f, Et, pt, at, zp)
        d_cf = (pT - pM).mean(0)
        cos = float(d @ d_cf / (d.norm() * d_cf.norm()))
        if cos < 0:
            d = -d
        dh = d / d.norm()
        lam = lambda p: float((((p - pM) * (pT - pM)).sum(1) / (pT - pM).pow(2).sum(1)).mean())  # noqa: E731
        log(f"t {t}: clusters {np.bincount(cl).tolist()}, |Delta_branch| {d.norm():.1f} cm, cos(cf) {cos:.2f}, lambda0 {lam(p0):.2f}")
        g = grad_E(f, E, pad, att, zp, dh)
        res = dict(t=t, cos=cos, lam0=lam(p0), ranks={}, rows=[])
        texts = {}
        for w, p in zip(words, pos):
            sc = ((Evoc - E[0, p]) @ g[p]).double().cpu().numpy()
            sc[~word_ok] = -np.inf; sc[ids[p - n_img]] = -np.inf
            order = np.argsort(-sc); rank = np.empty(V, int); rank[order] = np.arange(V)
            n_ok = int(word_ok.sum())
            res["ranks"][w] = {k: (int(rank[wid[k]]), float(sc[wid[k]])) for k in KEY if k in wid}
            log(f"  first-order ranks at '{w}' (of {n_ok}; + side rank, - side rank = {n_ok}-1-rank): " +
                " ".join(f"{k}:{rank[wid[k]]}/{sc[wid[k]]:+.1f}" for k in KEY if k in wid))
            for v in list(order[:POOL]) + list(order[n_ok - POOL:n_ok]):
                ww = pieces[v][1:]
                ws = words.copy(); ws[words.index(w)] = ww
                texts[" ".join(ws)] = (w, ww, float(sc[v]), "sub")
                if w != "open":
                    texts[f"open the {ww} drawer"] = (w, ww, float(sc[v]), "ins")
        for k in KEY:
            for tx in (f"open the {k} drawer", f"open a {k} drawer"):
                texts.setdefault(tx, ("-", k, float("nan"), "key"))
        texts[MID] = ("-", "-", float("nan"), "ref"); texts[TOP] = ("-", "-", float("nan"), "ref")
        tl = list(texts)
        log(f"  verifying {len(tl)} prompts")
        rows = []
        for i in range(0, len(tl), 8):
            chunk = tl[i:i + 8]
            with torch.no_grad():
                c = f.prefix([o] * len(chunk), chunk)
                for j, tx in enumerate(chunk):
                    cj = pi05flow.cslice(c, j, j + 1)
                    a = f.flow(f._z(zp), cexpand(cj, N_Z))[..., :f.A]
                    from decode import path50_t
                    px = torch.stack([path50_t(f, a[q]) for q in range(N_Z)])
                    dv = px - p0
                    shift = float((dv @ dh).mean()); move = float(dv.norm(dim=1).mean())
                    w, ww, fo, kind = texts[tx]
                    rows.append(dict(text=tx, pos=w, word=ww, fo=fo, kind=kind, shift=shift, move=move,
                                     align=shift / max(move, 1e-6), lam=lam(px)))
        res["rows"] = rows
        results.append(res)
        json.dump(results, open(OUTJ, "w"))
        ok = [r for r in rows if abs(r["align"]) >= ALIGN]
        ok.sort(key=lambda r: -r["shift"])
        print(f"\n  t {t}: {len(rows)} verified, {len(ok)} with |align| >= {ALIGN}. Top of each side (shift cm, move, align, lambda):")
        for r in ok[:12]:
            print(f"    {r['shift']:8.2f} {r['move']:7.2f} {r['align']:5.2f} {r['lam']:6.2f}  {r['text']} [{r['kind']}]")
        print("    ...")
        for r in ok[-12:]:
            print(f"    {r['shift']:8.2f} {r['move']:7.2f} {r['align']:5.2f} {r['lam']:6.2f}  {r['text']} [{r['kind']}]")
        print("  key prompts:")
        for r in sorted([r for r in rows if r["kind"] in ("key", "ref")], key=lambda r: -r["shift"]):
            print(f"    {r['shift']:8.2f} {r['move']:7.2f} {r['align']:5.2f} {r['lam']:6.2f}  {r['text']}")
        print(flush=True)
        s.set_state(x)
        a = f.unnorm(f.flow_np(rng_main.standard_normal((1, A)), f.prefix([o], [TEXT]))[0])
        for j in range(10):
            obs, _, _ = s.step(a[j])
    s.close()


if __name__ == "__main__":
    main()
