# PushT: looking for noise-space modes in pretrained policies

PushT (`gym-pusht`, pymunk): a 2-D disk agent pushes a T-shaped block onto a fixed target.
Action = absolute target position of the agent (pixels, 0–512), 10 Hz, 300 steps; success =
target coverage > 95%. Human demos (`lerobot/pusht`, 206 episodes) are the classic example of
multimodal behaviour (going around the block on either side, Diffusion Policy paper). The aim
here: find a pretrained policy whose *noise* decides between modes, as a testbed for the
FTLE / LCS pipeline before fine-tuning our own.

## Setup

```bash
uv pip install --python .venv/bin/python gym-pusht num2words diffusers
```

(`gym-pusht` downgrades `opencv-python` 5.0 → 4.14.)

## Checkpoints

| checkpoint | type | closed-loop success | notes |
|---|---|---|---|
| `n3puiol/smolvla_pusht` | SmolVLA (flow matching) | 0/20 (mean best coverage 0.22 at 8 executed steps, 0.07 at 50) | Unusable: trained with `load_vlm_weights=False` and a frozen VLM, i.e. on a random frozen backbone. Migrated to `policies/smolvla_pusht` with lerobot's `migrate_policy_normalization.py`. |
| `minhwantttt/pi05-pusht-chunk16` | pi0.5 (flow matching), full fine-tune from `pi05_base`, 50k × 32, chunk 16, 8 executed | 8/20 and 14/20 in two runs on the same seeds (lerobot's loop, unseeded noise); 7/12 through our flow map (seeded) | The analysed flow-matching policy. Same tokenizer workaround as robot5. Never evaluated in sim by its uploader. |
| `lerobot/diffusion_pusht` | Diffusion Policy (DDPM, 100 steps, 2 frames, horizon 16, 8 executed) | 12/20 on the same seeds (official eval: 65.4%) | Control. Migration fails on its config; loaded directly with the old normalisation buffers applied by hand (`diffusion6.py`). |

**pi0.5 is about as good as Diffusion Policy; the strict threshold makes both look worse.** Same 20
reset seeds (1000–1019), 300 steps:

| | pi0.5 run 1 | pi0.5 run 2 (600-step limit) | Diffusion Policy |
|---|---|---|---|
| success (coverage > 0.95) | 8/20 | 16/20 (14 within 300 steps) | 12/20 |
| Diffusion Policy paper's score, mean min(max coverage / 0.95, 1) | 0.962 | 0.998 | 0.981 |

Nearly every episode of both policies ends at 93–97% coverage, so success is mostly which side of
0.95 the final nudge lands on; the gap between the two pi0.5 runs is the (unseeded) noise, not the
time limit (only 2 episodes needed > 300 steps). The prompt matches the dataset's task string.

## pi0.5 as a flow map (`probe6.py check`)

`robot5/pi05flow.py` now takes chunk length, action size and the action normalisation (MEAN_STD for
LIBERO, QUANTILES for this checkpoint) from the checkpoint. PushT: chunk 16 × 2, noise space
16 × 32 = 512-D.

| check | result |
|---|---|
| explicit flow vs lerobot `sample_actions` (fp32) | 1.8e-6 |
| `unnorm` vs lerobot postprocessor | 3e-5 px |
| ∂Φ/∂z vs central FD, 8 directions × 3 states (fp64) | 7.1e-9 relative |

(A first version unnormalised with mean/std, which made flow-map rollouts fail; those screen outputs
are in `out/badunnorm/` and are invalid. The lerobot-loop success rates above were unaffected.)

## Noise-space modes: pi0.5 vs Diffusion Policy (`probe6.py screen`, `diffusion6.py screen`)

States: 100 initial states (reset seeds 1000–1099) plus every replan state along 12 closed-loop
rollouts of each policy (its own state distribution). Per state, 128 noise samples of the plan:
spread = max over chunk steps of the RMS std across samples; bimodality = Sarle's coefficient of
the samples' projection on their first principal component (> 5/9 suggests two modes).

| | pi0.5 (424 states) | Diffusion Policy (394 states) |
|---|---|---|
| plan spread, median / 90% / max (px) | 2.4 / 7.7 / 38.3 | 4.3 / 19.5 / 99.6 |
| initial states, median spread | 3.7 px | 12.3 px |
| states with bimodality coefficient > 5/9 | **1 / 424** (initial 0%, rollouts 0.3%) | **31 / 394** (initial 20%, rollouts 4%) |
| FTLE of the flow map (initial states) | −1.4 to −3.8 (contracting 4–45×) | (DDPM is stochastic; no flow map yet) |

- **pi0.5 is essentially unimodal.** Its most variable states spread along one direction with a
  single-humped distribution. The one exception (rollout 2001, step 80) is a 116-sample cluster plus
  a handful of stragglers (2-means separation 8), i.e. at most a ~5% minority branch.
- **Diffusion Policy has genuine noise-space modes**: at its most bimodal states the samples
  split into a majority and a minority cluster with an empty gap between them (e.g. 122 vs 6,
  113 vs 14, 111 + 6 vs 10 samples); in some, the two clusters' plans end tens of pixels apart
  (e.g. x = 194 vs 258).
- So the multimodality of the PushT demos survives in the small Diffusion Policy but not in
  fine-tuned pi0.5, consistent with LIBERO (robot5), where pi0.5's mode was set by the scene,
  never by the noise. A plausible reason: a 3B model fine-tuned for many steps on 206
  episodes fits each observation's own future, so ambiguity is resolved by the image.

Diffusion's DDPM sampler is stochastic; for FTLE / LCS the same trained noise predictor can be
integrated deterministically (DDIM / probability-flow ODE), giving a flow map z → chunk.

## Example run (`fig6.py`)

`python fig6.py run 1003 0 5 && python fig6.py plot 1003 5` -> `figs/example_1003_5.png`: one seeded
closed-loop run through the flow map (frames, agent path, block poses, coverage), with the plan under
16 other noise seeds drawn at each replan. From reset seed 1003, noise seeds 2–5 succeed (134–208
steps) and 0–1 stall at 0.94–0.95: the block reaches ~0.85 coverage by step 40, then the agent
circles and nudges it. The plans under other seeds mostly overlap within ~2 px.
