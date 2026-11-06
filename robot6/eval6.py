"""
PushT (gym-pusht) with flow-matching policies fine-tuned on lerobot/pusht:
  pi05   minhwantttt/pi05-pusht-chunk16 (pi0.5 from pi05_base, full fine-tune, 50k x 32,
         chunk 16, 8 executed)
  smolvla n3puiol/smolvla_pusht (migrated to policies/smolvla_pusht). Unusable: trained with
         load_vlm_weights=False and a frozen VLM, i.e. on a random frozen backbone (0/20 successes).

    POLICY=pi05 python eval6.py success [n_ep] [n_exec] [max_steps]   # lerobot's standard
                                          #   select_action loop: success rate (max_steps: past the 300-step limit)

PushT: a 2-D agent (disk) pushes a T-shaped block onto a fixed T target. Action = absolute target
position of the agent (pixels, 0..512), 10 Hz, 300 steps; success = coverage of the target > 95%.
Initial agent / block poses come from the reset seed.
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
POLICIES = {"pi05": "minhwantttt/pi05-pusht-chunk16", "smolvla": os.path.join(HERE, "policies", "smolvla_pusht")}
NAME = os.environ.get("POLICY", "pi05")
POLICY = POLICIES[NAME]
TASK = "Push the T-shaped block onto the T-shaped target."
OUT = os.path.join(HERE, "out")


def make_env():
    import gymnasium as gym
    import gym_pusht  # noqa: F401
    return gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")


def load_policy(n_exec=None):
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    over = {"device_processor": {"device": "cuda"}}
    if NAME == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        pol = SmolVLAPolicy.from_pretrained(POLICY)
    else:
        sys.path.insert(0, os.path.join(HERE, "..", "robot5"))
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from pi05flow import SPTokenizer                     # google/paligemma tokenizer is gated
        cfg = PreTrainedConfig.from_pretrained(POLICY)
        cfg.dtype, cfg.compile_model, cfg.gradient_checkpointing, cfg.device = torch.float32, False, False, "cpu"
        pol = PI05Policy.from_pretrained(POLICY, config=cfg)
        over["tokenizer_processor"] = {"tokenizer": SPTokenizer()}
    pol = pol.to("cuda").eval()
    if n_exec:
        pol.config.n_action_steps = n_exec
    pre, post = make_pre_post_processors(pol.config, POLICY, preprocessor_overrides=over)
    return pol, pre, post


def to_lerobot(obs):
    from lerobot.envs.utils import preprocess_observation
    o = preprocess_observation({k: v[None] for k, v in obs.items()})
    o["task"] = [TASK]
    return o


def success(n_ep=20, n_exec=None, max_steps=None):
    import torch
    pol, pre, post = load_policy(n_exec)
    env = make_env()
    res = []
    t0 = time.time()
    for ep in range(n_ep):
        obs, _ = env.reset(seed=1000 + ep)
        pol.reset()
        cov, ok = 0.0, False
        for t in range(max_steps or env.spec.max_episode_steps):
            with torch.inference_mode():
                a = post(pol.select_action(pre(to_lerobot(obs))))
            obs, r, term, trunc, info = env.step(a[0].cpu().numpy())
            cov = max(cov, float(info.get("coverage", r)))
            if info.get("is_success", False):
                ok = True
                break
        res.append(dict(seed=1000 + ep, success=ok, coverage=cov, steps=t + 1))
        print(f"[{time.time() - t0:5.0f}s] ep {ep} seed {1000 + ep}: success {ok} max coverage {cov:.2f} "
              f"steps {t + 1}", flush=True)
    n = pol.config.n_action_steps
    print(f"n_exec {n}: success {np.mean([r['success'] for r in res]):.2f}, "
          f"mean max coverage {np.mean([r['coverage'] for r in res]):.2f}")
    os.makedirs(OUT, exist_ok=True)
    tag = f"_steps{max_steps}" if max_steps else ""
    json.dump(res, open(os.path.join(OUT, f"success_{NAME}_exec{n}{tag}.json"), "w"))


if __name__ == "__main__":
    if sys.argv[1] == "success":
        success(*[int(x) for x in sys.argv[2:5]])
