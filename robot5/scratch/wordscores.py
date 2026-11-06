"""Full first-order word scores for the figure: for the cached t 0 split of "open a drawer" (init 3),
score every dictionary word at each prompt slot, and record the gradient norm per prefix block."""
import json, re, sys
import numpy as np, torch
sys.path.insert(0, "/tmp/claude-1000/-home-james-Projects-flow-matching-lcs/fc615ee8-3dcc-4cb5-8739-f0992e1fde17/scratchpad")
sys.path.insert(0, "/home/james/Projects/flow-matching-lcs/robot5")
import pi05flow
from env5 import Scene
from decode import A, TEXT, INIT, N_Z, grad_E, log

DIC = {w.strip().lower() for w in open("/usr/share/dict/american-english", errors="ignore") if w.strip().isalpha()}
OUTJ = "/home/james/Projects/flow-matching-lcs/robot5/out/decode3_open_a_drawer_3.json"


def main():
    f = pi05flow.load()
    sp = pi05flow.SPTokenizer().sp
    emb = f.m.paligemma_with_expert.paligemma.model.language_model.get_input_embeddings()
    V = sp.get_piece_size()
    pieces = [sp.id_to_piece(i) for i in range(V)]
    word_ok = np.array([re.fullmatch(r"▁[a-z]{2,}", p) is not None and p[1:] in DIC for p in pieces])
    with torch.no_grad():
        Evoc = torch.cat([emb(torch.arange(i, min(i + 32768, V), device=f.dev)) for i in range(0, V, 32768)])
    res = json.load(open(OUTJ))[0]
    T = np.load(OUTJ.replace(".json", "_T0.npy")); cl = np.array(res["cl"])
    m = [100 * T[cl == c, 1:51].mean(0).flatten() for c in range(2)]
    s = Scene(0, INIT, 0, suite_name="libero_goal"); obs = s.reset()
    o = s.obs_to_lerobot(obs, TEXT)
    E, pad, att = f.embed([o], [TEXT])
    n_img = sum(n * n for n in f.n_img)
    zp = np.random.default_rng(777).standard_normal((N_Z, A))
    d = torch.as_tensor(m[0] - m[1], dtype=torch.float32, device=f.dev); dh = d / d.norm()
    g = grad_E(f, E, pad, att, zp, dh)
    gn = g.norm(dim=1).double().cpu().numpy()
    P = E.shape[1]
    L = int(att[0].sum()) if att.dtype != torch.bool else int(att[0].sum())
    blocks = dict(images=float(np.linalg.norm(gn[:n_img])), bos_task=float(np.linalg.norm(gn[n_img:n_img + 3])),
                  prompt=float(np.linalg.norm(gn[n_img + 3:n_img + 6])), rest=float(np.linalg.norm(gn[n_img + 6:])))
    log("block gradient norms", blocks, "per prompt token", gn[n_img + 3:n_img + 6].tolist())
    out = dict(blocks=blocks, g_tok=gn[n_img + 3:n_img + 6].tolist(), words={})
    idx = np.where(word_ok)[0]
    for w, p in zip(TEXT.split(), [n_img + 3, n_img + 4, n_img + 5]):
        sc = ((Evoc - E[0, p]) @ g[p]).double().cpu().numpy()[idx]
        out["words"][w] = dict(scores=sc.tolist(), pieces=[pieces[i][1:] for i in idx])
        o_ = np.argsort(sc)
        log(f"'{w}': + {[pieces[idx[i]][1:] for i in o_[-5:][::-1]]}  - {[pieces[idx[i]][1:] for i in o_[:5]]}")
    json.dump(out, open("/home/james/Projects/flow-matching-lcs/robot5/out/decode_wordscores_open_a_drawer_3.json", "w"))
    s.close()


if __name__ == "__main__":
    main()
