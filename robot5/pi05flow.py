"""
pi0.5 (lerobot/pi05_libero_finetuned_v044) as a differentiable flow map, for FTLE / LCS analysis
on LIBERO.

pi0.5 = PaliGemma 3B (SigLIP + Gemma 2B) + a 300M Gemma action expert. Differences from pi0
(robot4/pi0flow.py) that matter here:

  - no state token: the 8-D proprio state is normalised, discretised into 256 bins and written
    into the prompt ("Task: ..., State: 12 200 ...;\\nAction: "). So the prefix K/V cache is the
    *whole* conditioning c (images + language + state), and the action tokens only attend to
    the cache and to each other.
  - flow time enters through adaRMSNorm: every expert norm maps cond(t) to (scale, shift, gate);
    residuals are gated. cond(t) = silu(W2 silu(W1 sincos(t))) does not depend on the actions,
    so it is precomputed once per Euler step.
  - chunk 50 x 32 (7 real dims: OSC delta pos 3, delta axis-angle 3, gripper), so the noise space
    is 1600-D and the task chunk 350-D. 10 forward-Euler steps from t = 1 (noise) to 0.

Phi(z; c) = chunk after 10 steps; FTLE = log sigma_max(dPhi_task / dz). Outputs are in the
policy's normalised action space (MEAN_STD); `unnorm` maps to env actions.

    python pi05flow.py check     # explicit flow vs lerobot sample_actions; dPhi/dz vs FD (float64)

Precision: VLM prefix in float32 (bfloat16 makes the K/V depend on the batch, see robot4);
expert float32, float64 for derivative checks. TF32 stays off.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jacrev, vmap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REPO = "lerobot/pi05_libero_finetuned_v044"
H = 50                           # chunk length (LIBERO checkpoint; per instance: self.H)
D_ACT = 32                       # padded action width
A_TASK = 7                       # real action dims (LIBERO checkpoint; per instance: self.A)
N_STEPS = 10
MASK_NEG = -2.3819763e38         # openpi's additive mask value
DEVICE = torch.device(os.environ.get("TORCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
DTYPE = getattr(torch, os.environ.get("TORCH_DTYPE", "float32"))


def rope(x, cos, sin):
    x1, x2 = x.chunk(2, -1)
    return x * cos + torch.cat([-x2, x1], -1) * sin


def cmap(c, fn):
    return {k: ([fn(u) for u in v] if isinstance(v, list) else tuple(fn(u) for u in v) if isinstance(v, tuple)
                else fn(v)) for k, v in c.items()}


def cbatch(c):
    return c["mask"].shape[0]


def cslice(c, i, j):
    return cmap(c, lambda u: u[i:j])


def cexpand(c, n):
    return cmap(c, lambda u: u.expand(n, *u.shape[1:]))


def ccat(cs):
    return {k: ([torch.cat([c[k][l] for c in cs]) for l in range(len(cs[0][k]))] if isinstance(cs[0][k], list)
                else tuple(torch.cat([c[k][l] for c in cs]) for l in range(len(cs[0][k])))
                if isinstance(cs[0][k], tuple) else torch.cat([c[k] for c in cs])) for k in cs[0]}


class Pi05Flow:
    def __init__(self, policy, pre, post, dtype=DTYPE, vlm_dtype=torch.float32, device=DEVICE):
        self.pol, self.pre, self.post = policy, pre, post
        self.m = policy.model                                       # PI05Pytorch
        self.dev = torch.device(device)
        pwe = self.m.paligemma_with_expert
        pwe.gemma_expert.lm_head = None                             # unused 257k-vocab head (1 GB fp32)
        pwe.paligemma.to(device=self.dev, dtype=vlm_dtype)
        for mod in (pwe.paligemma.model.vision_tower, pwe.paligemma.model.multi_modal_projector):
            mod.to(torch.float32)
        self.layers = pwe.gemma_expert.model.layers
        self.final_norm = pwe.gemma_expert.model.norm
        cfg = pwe.gemma_expert.config
        self.n_heads, self.hd = cfg.num_attention_heads, cfg.head_dim
        self.m.eval()
        self.m.requires_grad_(False)
        self.set_dtype(dtype)
        st = {k: v for k, v in post.steps[0].state_dict().items()}
        self.H = policy.config.chunk_size
        self.A = policy.config.output_features["action"].shape[0]
        # the postprocessor's unnormalisation as an affine map: env action = a * a_scale + a_shift
        mode = str(getattr(policy.config.normalization_mapping["ACTION"], "value",
                           policy.config.normalization_mapping["ACTION"]))
        g = lambda k: st[f"action.{k}"].double().cpu().numpy()[:self.A]  # noqa: E731
        if mode == "MEAN_STD":                                   # LIBERO checkpoints
            self.a_scale, self.a_shift = g("std"), g("mean")
        elif mode == "QUANTILES":                                # PushT checkpoint: [-1, 1] <-> [q01, q99]
            self.a_scale = (g("q99") - g("q01")) / 2; self.a_shift = g("q01") + self.a_scale
        else:
            raise NotImplementedError(f"action normalisation {mode}")

    def set_dtype(self, dtype):
        self.dt = dtype
        m = self.m
        for mod in (m.paligemma_with_expert.gemma_expert, m.action_in_proj, m.action_out_proj,
                    m.time_mlp_in, m.time_mlp_out):
            mod.to(device=self.dev, dtype=dtype)
        theta = self.m.paligemma_with_expert.gemma_expert.config.rope_parameters["rope_theta"]
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.hd, 2, dtype=torch.float64, device=self.dev) / self.hd))
        # adaRMS conditioning for each Euler step (independent of the actions)
        from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding
        self.ts = [1.0 + i * (-1.0 / N_STEPS) for i in range(N_STEPS)]
        with torch.no_grad():
            t = torch.tensor(self.ts, dtype=torch.float32, device=self.dev)
            temb = create_sinusoidal_pos_embedding(t, m.action_in_proj.out_features, m.config.min_period,
                                                   m.config.max_period, device=self.dev).to(torch.float32).to(dtype)
            self.tcond = F.silu(m.time_mlp_out(F.silu(m.time_mlp_in(temb))))      # [N_STEPS, W]
        return self

    # ---------------------------------------------------------------- prefix (conditioning)
    def batch(self, obs, texts):
        """obs: list of lerobot observations (after LiberoProcessorStep, batch 1 each);
        texts: list of instructions. Returns the preprocessed policy batch."""
        bs = []
        for o, t in zip(obs, texts):
            o = dict(o); o["task"] = [t]
            bs.append(self.pre(o))
        keys = [k for k, v in bs[0].items() if torch.is_tensor(v)]
        return {k: torch.cat([b[k] for b in bs]) for k in keys}

    def _rope_tables(self, pos):
        f = pos.to(torch.float64)[..., None] * self.inv_freq
        emb = torch.cat([f, f], -1)[:, None]
        return emb.cos().to(self.dt), emb.sin().to(self.dt)

    @torch.no_grad()
    def embed(self, obs, texts):
        """Prefix token embeddings E [N, P, W] (image patches of each camera, then the prompt +
        state tokens), their pad mask and attention mask. E is the latent the policy conditions
        on: everything after it (Gemma prefix -> K/V -> expert) is differentiable."""
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        b = self.batch(obs, texts)
        images, img_masks = self.pol._preprocess_images(b)
        E, pad, att = self.m.embed_prefix(images, img_masks, b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK])
        self.n_img = [im.shape[-1] // self.m.paligemma_with_expert.paligemma.config.vision_config.patch_size for im in images]
        return E.to(self.m.paligemma_with_expert.paligemma.dtype), pad, att

    def prefix_from(self, E, pad, att):
        """Conditioning c from prefix embeddings (differentiable in E when grad is enabled)."""
        from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d
        pwe = self.m.paligemma_with_expert
        att2 = make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, 1) - 1
        pwe.paligemma.model.language_model.config._attn_implementation = "eager"
        _, kv = pwe.forward(attention_mask=prepare_attention_masks_4d(att2), position_ids=pos,
                            past_key_values=None, inputs_embeds=[E, None], use_cache=True)
        K = [k[:, 0].to(self.dt) for k, v, *_ in kv]               # 1 kv head: [N, P, hd]
        V = [v[:, 0].to(self.dt) for k, v, *_ in kv]
        offset = pad.sum(1)
        return dict(K=K, V=V, mask=pad,
                    rope=self._rope_tables(offset[:, None] + torch.arange(self.H, device=self.dev)))

    @torch.no_grad()
    def prefix(self, obs, texts, bs=8):
        """Conditioning c for each (observation, instruction): per-layer prefix K/V, pad mask,
        RoPE tables of the H action tokens."""
        if len(obs) > bs:
            return ccat([self.prefix(obs[i:i + bs], texts[i:i + bs], bs) for i in range(0, len(obs), bs)])
        return self.prefix_from(*self.embed(obs, texts))

    def recast(self, c):
        """Conditioning computed under another expert dtype -> this one."""
        return {k: ([u.to(self.dt) for u in v] if k in ("K", "V") else
                    tuple(u.to(self.dt) for u in v) if k == "rope" else v) for k, v in c.items()}

    # ---------------------------------------------------------------- expert
    @staticmethod
    def _ada(norm, x, mod):
        """PiGemmaRMSNorm with precomputed modulation mod = dense(cond) [..., 3W] (broadcast
        over tokens). Returns normed x and the residual gate."""
        xf = x.to(torch.promote_types(x.dtype, torch.float32))
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm.eps)
        scale, shift, gate = mod.chunk(3, -1)
        return (normed * (1 + scale) + shift).to(x.dtype), gate

    def _layer(self, layer, h, cond, cs, Kc, Vc, cmask):
        at = layer.self_attn
        x, g1 = self._ada(layer.input_layernorm, h, layer.input_layernorm.dense(cond))
        q = at.q_proj(x).unflatten(-1, (self.n_heads, self.hd)).transpose(-3, -2)    # [..., nh, T, hd]
        k = at.k_proj(x); v = at.v_proj(x)                                           # [..., T, hd]
        cos, sin = cs
        q = rope(q, cos, sin); k = rope(k, cos[..., 0, :, :], sin[..., 0, :, :])
        s_c = torch.einsum("...htd,...pd->...htp", q, Kc) * at.scaling
        s_c = s_c + torch.where(cmask, 0.0, MASK_NEG).to(s_c.dtype)[..., None, None, :]
        s_s = torch.einsum("...htd,...sd->...hts", q, k) * at.scaling
        w = torch.softmax(torch.cat([s_c, s_s], -1).to(torch.promote_types(q.dtype, torch.float32)), -1).to(q.dtype)
        P = Kc.shape[-2]
        o = (torch.einsum("...htp,...pd->...thd", w[..., :P], Vc) +
             torch.einsum("...hts,...sd->...thd", w[..., P:], v)).flatten(-2)
        h = h + at.o_proj(o) * g1
        x, g2 = self._ada(layer.post_attention_layernorm, h, layer.post_attention_layernorm.dense(cond))
        return h + layer.mlp(x) * g2

    def velocity(self, x, i, c):
        """x [N, H, 32] at Euler step i (t = 1 - i/10) -> v [N, H, 32]. Pure torch."""
        cond = self.tcond[i][None, None]                             # [1, 1, W] broadcasts over N, T
        h = self.m.action_in_proj(x)
        for li, layer in enumerate(self.layers):
            h = self._layer(layer, h, cond, c["rope"], c["K"][li], c["V"][li], c["mask"])
        out, _ = self._ada(self.final_norm, h, self.final_norm.dense(cond))
        return self.m.action_out_proj(out)

    def flow(self, z, c, n_steps=N_STEPS):
        assert n_steps == N_STEPS
        dt = -1.0 / n_steps
        x = z
        for i in range(n_steps):
            x = x + dt * self.velocity(x, i, c)
        return x

    # ---------------------------------------------------------------- numpy in/out
    def _z(self, z):
        return torch.as_tensor(np.asarray(z).reshape(len(z), self.H, D_ACT), dtype=self.dt, device=self.dev)

    @torch.no_grad()
    def flow_np(self, z, c, bs=64):
        """Normalised task chunks [N, H, A] for seeds z [N, H*32] under c of batch N or 1."""
        out = []
        for i in range(0, len(z), bs):
            zz = self._z(z[i:i + bs])
            ci = cexpand(c, len(zz)) if cbatch(c) == 1 else cslice(c, i, i + bs)
            out.append(self.flow(zz, ci)[..., :self.A])
        return torch.cat(out).double().cpu().numpy()

    def unnorm(self, a):
        """Normalised chunk [..., 7] -> env actions (the checkpoint's postprocessor)."""
        return a * self.a_scale + self.a_shift

    def jac(self, z, c, steps=None, bs=4, chunk=None):
        """Chunks [N, H, A] and dPhi/dz [N, steps*A, H*32] (rows: the first `steps` chunk steps,
        step-major), for c of batch N or 1 (shared). Reverse mode."""
        shared = cbatch(c) == 1
        c = cmap(c, lambda u: u.detach())
        steps = steps or self.H

        def f(z1, c1):
            return self.flow(z1[None], cmap(c1, lambda u: u[None]))[0, :steps, :self.A].flatten()

        out = []
        for i in range(0, len(z), bs):
            zz = self._z(z[i:i + bs])
            ci = cmap(c, lambda u: u[0]) if shared else cslice(c, i, i + bs)
            J = vmap(jacrev(f, chunk_size=chunk), in_dims=(0, None if shared else 0))(zz, ci)
            out.append(J.flatten(-2).double().cpu())
        return self.flow_np(z, c), torch.cat(out).numpy()


TOKENIZER = os.path.join(HERE, "..", "robot4", "assets", "paligemma_tokenizer.model")   # big_vision's copy


class SPTokenizer:
    """PaliGemma's sentencepiece tokenizer behind the HF call signature lerobot uses
    (google/paligemma-3b-pt-224 is gated): BOS + text, no EOS, padded with 0."""

    def __init__(self, path=TOKENIZER):
        import sentencepiece
        self.sp = sentencepiece.SentencePieceProcessor(model_file=path)

    def __call__(self, text, max_length, truncation=True, padding="max_length", padding_side="right",
                 return_tensors="pt"):
        texts = [text] if isinstance(text, str) else list(text)
        assert padding == "max_length" and padding_side == "right"
        ids = torch.zeros(len(texts), max_length, dtype=torch.long)
        mask = torch.zeros(len(texts), max_length, dtype=torch.long)
        for i, t in enumerate(texts):
            tok = ([self.sp.bos_id()] + self.sp.encode(t))[:max_length]
            ids[i, :len(tok)] = torch.tensor(tok); mask[i, :len(tok)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def load(dtype=DTYPE, repo=REPO, **kw):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = PreTrainedConfig.from_pretrained(repo)
    cfg.dtype, cfg.compile_model, cfg.gradient_checkpointing, cfg.device = torch.float32, False, False, "cpu"
    pol = PI05Policy.from_pretrained(repo, config=cfg)
    pre, post = make_pre_post_processors(pol.config, repo,
                                         preprocessor_overrides={"device_processor": {"device": str(DEVICE)},
                                                                 "tokenizer_processor": {"tokenizer": SPTokenizer()}})
    return Pi05Flow(pol, pre, post, dtype=dtype, **kw)


def ftle(J):
    return np.log(np.linalg.svd(J, compute_uv=False)[:, 0])


# ----------------------------------------------------------------------------- checks
def check(n_scene=3, n_dir=8, eps=1e-4, steps=10):
    """(1) the explicit expert flow equals lerobot's sample_actions on the same noise (float32);
    (2) dPhi/dz (reverse mode, first `steps` chunk steps) equals central finite differences along
    random directions, with the expert in float64."""
    from env5 import Scene
    rng = np.random.default_rng(0)
    f = load(torch.float32)
    obs, texts = [], []
    for k in range(n_scene):
        s = Scene(task_id=k, init_id=k)
        obs.append(s.obs_to_lerobot(s.reset(), s.task)); texts.append(s.task)
        s.close()
    z = rng.standard_normal((n_scene, H * D_ACT))
    with torch.no_grad():
        b = f.batch(obs, texts)
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        images, img_masks = f.pol._preprocess_images(b)
        ref = f.m.sample_actions(images, img_masks, b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK],
                                 noise=f._z(z), num_steps=N_STEPS)[..., :A_TASK].double().cpu().numpy()
        c32 = f.prefix(obs, texts)
        mine = f.flow_np(z, c32)
    print(f"explicit flow vs lerobot sample_actions (fp32): max abs diff {np.abs(mine - ref).max():.2e} "
          f"(|a| max {np.abs(ref).max():.2f})", flush=True)
    print("first env action of each chunk:", np.round(f.unnorm(mine[:, 0]), 3).tolist(), flush=True)

    f.set_dtype(torch.float64)
    c = f.recast(c32)
    t0 = time.time()
    a, J = f.jac(z, c, steps=steps, bs=1)
    print(f"jacobian [{J.shape[1]} x {J.shape[2]}] x {n_scene}: {time.time() - t0:.1f}s (float64)", flush=True)
    U = rng.standard_normal((n_dir, H * D_ACT)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    err = []
    for i in range(n_scene):
        ci = cslice(c, i, i + 1)
        fd = (f.flow_np(z[i] + eps * U, ci)[:, :steps] - f.flow_np(z[i] - eps * U, ci)[:, :steps]) / (2 * eps)
        fd = fd.reshape(n_dir, -1)
        err.append(np.abs(U @ J[i].T - fd).max() / np.abs(fd).max())
    print(f"dPhi/dz (float64) vs central FD, {n_dir} random directions x {n_scene} scenes: "
          f"max rel err {max(err):.2e} | FTLE {np.round(ftle(J), 3)}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
