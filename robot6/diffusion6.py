"""
Control for the PushT mode screen: lerobot/diffusion_pusht (Diffusion Policy, DDPM 100 steps,
2 observation frames, horizon 16, 8 executed; official eval: 65.4% success). Loaded directly; its
old-style normalisation buffers (lerobot's migration script fails on this config) are applied by
hand: image MEAN_STD, state and action MIN_MAX to [-1, 1].

    python diffusion6.py success [n_ep]   # closed loop, checks the hand-built pipeline
    python diffusion6.py screen           # same states as probe6 screen (100 initial states +
                                          #   replan states of 12 own rollouts): plan spread across
                                          #   128 noise samples, bimodality of the main axis

Bimodality: Sarle's coefficient BC = (g^2 + 1) / (k + 3 (n-1)^2 / ((n-2)(n-3))) of the samples'
projection on their first principal component (g skewness, k excess kurtosis); BC > 5/9 suggests
two modes (uniform = 5/9, Gaussian = 1/3).
"""
import glob, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval6 import make_env, OUT  # noqa: E402

N_SAMP = 128


class Diffusion:
    def __init__(self):
        from safetensors import safe_open
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        p = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--lerobot--diffusion_pusht/snapshots/*/"))[0]
        self.pol = DiffusionPolicy.from_pretrained(p).to("cuda").eval()
        with safe_open(p + "model.safetensors", "pt") as f:
            g = lambda k: f.get_tensor(k).to("cuda")  # noqa: E731
            self.im_mean, self.im_std = g("normalize_inputs.buffer_observation_image.mean"), g(
                "normalize_inputs.buffer_observation_image.std")
            self.s_min, self.s_max = g("normalize_inputs.buffer_observation_state.min"), g(
                "normalize_inputs.buffer_observation_state.max")
            self.a_min, self.a_max = (g("unnormalize_outputs.buffer_action.min").cpu().numpy(),
                                      g("unnormalize_outputs.buffer_action.max").cpu().numpy())
        self.H = self.pol.config.horizon

    @torch.no_grad()
    def cond(self, obs2):
        """obs2: the two latest raw env observations (oldest first) -> global conditioning [1, D]."""
        img = torch.stack([torch.as_tensor(o["pixels"], device="cuda").permute(2, 0, 1).float() / 255 for o in obs2])
        img = (img - self.im_mean) / self.im_std
        st = torch.stack([torch.as_tensor(o["agent_pos"], device="cuda").float() for o in obs2])
        st = 2 * (st - self.s_min) / (self.s_max - self.s_min) - 1
        return self.pol.diffusion._prepare_global_conditioning(
            {"observation.state": st[None], "observation.images": img[None, :, None]})

    @torch.no_grad()
    def sample(self, gc, n, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = self.pol.diffusion.conditional_sample(n, gc.expand(n, -1), generator=g).cpu().numpy()
        return (x + 1) / 2 * (self.a_max - self.a_min) + self.a_min          # [n, horizon, 2] px

    def plan(self, obs2, seed):
        """Executed part of the plan, as lerobot: steps n_obs-1 .. n_obs-1+8."""
        a = self.sample(self.cond(obs2), 1, seed)[0]
        s = self.pol.config.n_obs_steps - 1
        return a[s:s + self.pol.config.n_action_steps]


def bimodality(a):
    x = a.reshape(len(a), -1); x = x - x.mean(0)
    p = x @ np.linalg.svd(x, full_matrices=False)[2][0]
    n = len(p); z = (p - p.mean()) / p.std()
    g, k = (z ** 3).mean(), (z ** 4).mean() - 3
    return (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def spread(a):
    return float(np.sqrt((a.std(0) ** 2).sum(-1)).max())


def success(n_ep=10):
    d = Diffusion(); env = make_env()
    ok, res = [], []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=1000 + ep); prev = obs; done = False; cov = 0.0
        for t in range(0, 300, 8):
            a = d.plan([prev, obs], seed=100 * ep + t)
            for k in range(8):
                prev = obs
                obs, _, term, trunc, info = env.step(a[k])
                cov = max(cov, float(info.get("coverage", 0)))
                if info.get("is_success", False) or term or trunc:
                    done = True; break
            if done:
                break
        done = bool(info.get("is_success", False))
        ok.append(done)
        res.append(dict(seed=1000 + ep, success=done, coverage=cov, steps=t + k + 1))
        print(f"ep {ep}: success {done} max coverage {cov:.2f} steps {t + k + 1}", flush=True)
    c = np.array([r["coverage"] for r in res])
    print(f"diffusion_pusht success {np.mean(ok):.2f}, mean max coverage {c.mean():.3f}, "
          f"DP-paper score (mean min(cov/0.95, 1)) {np.minimum(c / 0.95, 1).mean():.3f}")
    import json
    json.dump(res, open(os.path.join(OUT, "success_diffusion_exec8.json"), "w"))


def screen(n_init=100, n_roll=12):
    d = Diffusion(); env = make_env()
    rows = []
    t0 = time.time()

    def probe(obs2, kind, seed, t):
        a = d.sample(d.cond(obs2), N_SAMP, seed=7)
        rows.append(dict(kind=kind, seed=seed, t=t, spread=spread(a), bc=bimodality(a), chunks=a.astype(np.float32),
                         agent=obs2[-1]["agent_pos"].copy()))

    for s in range(1000, 1000 + n_init):
        obs, _ = env.reset(seed=s)
        probe([obs, obs], "init", s, 0)
    print(f"[{time.time() - t0:.0f}s] initial states done", flush=True)
    for r in range(n_roll):
        obs, _ = env.reset(seed=2000 + r); prev = obs; done = False
        for t in range(0, 300, 8):
            probe([prev, obs], "roll", 2000 + r, t)
            a = d.plan([prev, obs], seed=10_000 * r + t)
            for k in range(8):
                prev = obs
                obs, _, term, trunc, info = env.step(a[k])
                if info.get("is_success", False) or term or trunc:
                    done = True; break
            if done:
                break
        print(f"[{time.time() - t0:.0f}s] rollout {r}: success {bool(info.get('is_success', False))}", flush=True)
    sp = np.array([r["spread"] for r in rows]); bc = np.array([r["bc"] for r in rows])
    print(f"diffusion: {len(rows)} states | plan spread (px) median {np.median(sp):.1f}, 90% {np.percentile(sp, 90):.1f}, "
          f"max {sp.max():.1f} | bimodality coefficient median {np.median(bc):.2f}, share > 5/9: {np.mean(bc > 5 / 9):.2f}")
    np.savez_compressed(os.path.join(OUT, "screen_diffusion.npz"), **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    # pi0.5 on the same initial states, same statistic
    P = np.load(os.path.join(OUT, "screen.npz"), allow_pickle=True)
    bcp = np.array([bimodality(a) for a in P["chunks"]])
    print(f"pi0.5    : {len(bcp)} states | plan spread (px) median {np.median(P['spread']):.1f}, 90% "
          f"{np.percentile(P['spread'], 90):.1f}, max {P['spread'].max():.1f} | bimodality coefficient median "
          f"{np.median(bcp):.2f}, share > 5/9: {np.mean(bcp > 5 / 9):.2f}")
    ini = np.array([r["kind"] == "init" for r in rows])
    print(f"initial states only: spread median diffusion {np.median(sp[ini]):.1f} px vs pi0.5 "
          f"{np.median(P['spread'][P['kind'] == 'init']):.1f} px")


if __name__ == "__main__":
    if sys.argv[1] == "success":
        success(*[int(x) for x in sys.argv[2:3]])
    elif sys.argv[1] == "screen":
        screen()
