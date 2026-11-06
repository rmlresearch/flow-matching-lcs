"""End-to-end, candidate-free: detect the split by branching, decode the separating direction into
prompt edits (first-order over dictionary words), verify by closeness of the edited prompt's plan
to the branch modes, and confirm by re-branching under the two suggestions."""
import json, os, re, sys, time
import numpy as np, torch
sys.path.insert(0, "/tmp/claude-1000/-home-james-Projects-flow-matching-lcs/fc615ee8-3dcc-4cb5-8739-f0992e1fde17/scratchpad")
sys.path.insert(0, "/home/james/Projects/flow-matching-lcs/robot5")
import pi05flow
from pi05flow import cexpand
from env5 import Scene
from decode import A, TEXT, INIT, MID, TOP, N_BR, HORIZON, N_Z, plans, grad_E, branch, two_clusters, path50_t, log

SHORT = 6
DIC = {w.strip().lower() for w in open("/usr/share/dict/american-english", errors="ignore") if w.strip().isalpha()}
OUTJ = "/home/james/Projects/flow-matching-lcs/robot5/out/decode3_open_a_drawer_3.json"


def main():
    f = pi05flow.load()
    sp = pi05flow.SPTokenizer().sp
    emb = f.m.paligemma_with_expert.paligemma.model.language_model.get_input_embeddings()
    V = sp.get_piece_size()
    pieces = [sp.id_to_piece(i) for i in range(V)]
    word_ok = np.array([re.fullmatch(r"▁[a-z]{2,}", p) is not None and p[1:] in DIC for p in pieces])
    log(f"dictionary pieces {word_ok.sum()}")
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
        # 1. detect: branch, cluster on the final eef
        T, labs = branch(s, x, TEXT, f, N_BR, HORIZON)
        cl = two_clusters(T[:, -1:])
        cent = np.stack([T[cl == c, -1].mean(0) for c in range(2)])
        m = [100 * T[cl == c, 1:51].mean(0).flatten() for c in range(2)]        # mode paths over the plan horizon, cm
        sep = float(np.linalg.norm(m[0] - m[1]))
        spread = float(np.mean([np.linalg.norm(100 * T[i, 1:51].flatten() - m[cl[i]]) for i in range(len(T))]))
        log(f"t {t}: clusters {np.bincount(cl).tolist()}, mode separation over 50 steps {sep:.1f} cm, within-mode spread {spread:.1f} cm")
        obs = s.set_state(x)
        # 2. direction and gradient
        d = torch.as_tensor(m[0] - m[1], dtype=torch.float32, device=f.dev); dh = d / d.norm()
        g = grad_E(f, E, pad, att, zp, dh)
        p0 = plans(f, E, pad, att, zp)
        dist = lambda p: [float(np.linalg.norm(p.double().cpu().numpy() - mm, axis=1).mean()) for mm in m]  # noqa: E731
        d0 = dist(p0)
        log(f"  plan under '{TEXT}': distance to mode 0 / 1: {d0[0]:.1f} / {d0[1]:.1f} cm")
        # 3. decode: first-order over dictionary words, shortlist per sign and position
        cands = {}
        for w, p in zip(words, pos):
            sc = ((Evoc - E[0, p]) @ g[p]).double().cpu().numpy()
            sc[~word_ok] = np.nan; sc[ids[p - n_img]] = np.nan
            order = np.argsort(np.nan_to_num(sc, nan=0))
            for sign, vs in (("+", order[-SHORT:][::-1]), ("-", order[:SHORT])):
                for v in vs:
                    ww = pieces[v][1:]
                    ws = words.copy(); ws[words.index(w)] = ww
                    cands[" ".join(ws)] = dict(pos=w, word=ww, sign=sign, fo=float(sc[v]), kind="sub")
                    if w != "open":
                        cands[f"open the {ww} drawer"] = dict(pos=w, word=ww, sign=sign, fo=float(sc[v]), kind="ins")
            log(f"  '{w}': + {[pieces[v][1:] for v in order[-SHORT:][::-1]]}  - {[pieces[v][1:] for v in order[:SHORT]]}")
        for tx in (MID, TOP):
            cands[tx] = dict(pos="-", word="-", sign="ref", fo=float("nan"), kind="ref")
        # 4. verify by closeness of the plan to the modes
        rows = []
        for tx, c in cands.items():
            with torch.no_grad():
                Ex, px, ax = f.embed([o], [tx])
            pxp = plans(f, Ex, px, ax, zp)
            dd = dist(pxp)
            rows.append(dict(text=tx, **c, d0=dd[0], d1=dd[1], move=float((pxp - p0).norm(dim=1).mean()),
                             toward=(dd[1] - dd[0]) / sep, near=min(dd)))
        rows.sort(key=lambda r: -r["toward"])
        print(f"\n  t {t}: {len(rows)} candidates. toward = (dist to mode 1 - dist to mode 0) / separation; near = dist to the nearer mode")
        print(f"    {'toward':>7} {'near':>6} {'move':>6} {'f.o.':>7}  text")
        for r in rows[:10] + [None] + rows[-10:]:
            if r is None:
                print("    ..."); continue
            print(f"    {r['toward']:7.2f} {r['near']:6.1f} {r['move']:6.1f} {r['fo']:7.1f}  {r['text']} [{r['sign']}{r['kind']}]")
        # 5. suggestions: nearest plan to each mode among the non-reference candidates; confirm by re-branching
        sug = [min([r for r in rows if r["kind"] != "ref"], key=lambda r: r[f"d{c}"]) for c in range(2)]
        conf = []
        for c, r in enumerate(sug):
            Tc, _ = branch(s, x, r["text"], f, 8, HORIZON, seed0=300)
            lab = np.argmin(np.linalg.norm(Tc[:, -1][:, None] - cent[None], axis=-1), axis=1)
            conf.append(np.bincount(lab, minlength=2).tolist())
            log(f"  suggestion for mode {c}: '{r['text']}' (plan {r[f'd{c}']:.1f} cm from mode {c}, {r[f'd{1-c}']:.1f} from the other); "
                f"re-branched 8x: lands in mode 0 / 1 = {conf[-1]}")
        obs = s.set_state(x)
        results.append(dict(t=t, clusters=np.bincount(cl).tolist(), sep=sep, spread=spread, d0=d0, rows=rows,
                            suggestions=[r["text"] for r in sug], confirm=conf,
                            events=[sorted(l.items(), key=lambda kv: kv[1]) for l in labs], cl=cl.tolist()))
        json.dump(results, open(OUTJ, "w"))
        np.save(OUTJ.replace(".json", f"_T{t}.npy"), T)
        a = f.unnorm(f.flow_np(rng_main.standard_normal((1, A)), f.prefix([o], [TEXT]))[0])
        for j in range(10):
            obs, _, _ = s.step(a[j])
    s.close()


if __name__ == "__main__":
    main()
