"""
pi0.5 on the real SO-101: the SO-100/101 generalist (hqfang/pi05-so100_101, pi0.5 base fine-tuned on
1,209 community datasets) as a differentiable flow map on real observations (wrist camera + joint
readings + prompt), with robot5's machinery (pi05flow.Pi05Flow) and the SO-101 bridge of simlab.

Differences from robot5/robot6 that matter here:

  - observation: one 1280x720 wrist camera -> centre crop to 4:3 (community wrist cams), letterbox
    to a square, 224x224 -> slot `observation.images.camera_0`; slots 1-3 are missing and the policy
    pads them (-1 images, masked out of the prefix). State = 6 joints in the *old SO-100 degrees*
    the mixture was recorded in (so101/vla_common.py: shoulder_lift 90 - ours, elbow 90 + ours,
    wrist_roll inverted about the camera-on-top roll; `--units so100` in simlab's client). The state
    is normalised by the checkpoint's own quantile statistics and discretised into the prompt.
  - actions: absolute joint targets in the same old degrees (QUANTILES unnormalisation), chunk
    50 x 6, so the noise space is 1600-D and the task chunk 300-D.
  - plan: the chunk pushed through the SO-101 forward kinematics (torch, extracted from simlab's
    MJCF) = the gripper (tcp) path in cm, the analogue of robot5's eef path.
  - 12 GB GPU: the Gemma prefix (language model) runs in bfloat16, SigLIP and the expert in
    float32 (float64 for derivative checks). Consequences: the prefix is computed at batch 1
    always (bf16 K/V depend on the batch, robot4), and derivatives with respect to the prompt
    embedding use autograd (reverse mode, and forward mode by double backward), never finite
    differences in E (a 1e-3 relative step is below bf16 resolution).

    python so101flow.py check [--live]          # flow vs lerobot sample_actions; dPhi/dz vs FD; FK vs MuJoCo
    python so101flow.py ftle  [--live] [--seeds 4] [--steps 15]
    python so101flow.py ridge [--live] [--task "..."]

Observation sources: --live reads the wrist camera (and the bus if it answers; else the rest pose),
--frame path.jpg [--state p l e f r g] uses a saved frame. Outputs go to out/.
"""
import os, sys, json, time, math, argparse
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "robot5"))
SIMLAB = os.environ.get("SIMLAB", os.path.expanduser("~/Documents/GitHub/software/simlab"))
sys.path.insert(0, os.path.join(SIMLAB, "workspace", "skills"))
OUT = os.path.join(HERE, "out")

REPO = "hqfang/pi05-so100_101"
TASK = "Pick up the blue eraser and put it in the black bin."
# our convention (simlab): deg x5 with 0 = LeRobot's calibration middle pose, gripper 0 (closed) .. 100 (open)
REST_DEG = np.array([0.0, -99.7, 91.2, 55.5, -89.0, 10.0], np.float32)
H, D_ACT, A = 50, 32, 6
ASPECT = 4.0 / 3.0
IMG = 224
VLM_DTYPE = getattr(torch, os.environ.get("VLM_DTYPE", "bfloat16"))
t00 = time.time()


def log(*a):
    print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------ units
def so100_units():
    from so101.vla_common import SO100_SIGN, SO100_OFFSET, SO100_ROLL_ZERO_DEG
    return SO100_SIGN, SO100_OFFSET, SO100_ROLL_ZERO_DEG


def to_so100(state_deg):
    from so101.vla_common import to_so100_units
    return to_so100_units(state_deg)


def from_so100(state_old):
    from so101.vla_common import from_so100_units
    return from_so100_units(state_old)


# ------------------------------------------------------------------------------------------ images
def prep_image(rgb, aspect=ASPECT, size=IMG):
    """HxWx3 uint8 RGB -> [1, 3, size, size] float in [0, 1]: centre crop to `aspect`, letterbox to a
    square with black bars, bilinear resize (what simlab's server did for the runs that grasped)."""
    from so101.vla_common import crop_aspect
    img = crop_aspect(rgb, aspect) if aspect else rgb
    h, w = img.shape[:2]
    if abs(w / h - 1.0) > 0.02:
        if w > h:
            p = (w - h) // 2
            img = np.pad(img, ((p, w - h - p), (0, 0), (0, 0)))
        else:
            p = (h - w) // 2
            img = np.pad(img, ((0, 0), (p, h - w - p), (0, 0)))
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float()
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False) / 255.0


def observation(rgb, state_deg, aspect=ASPECT):
    """Wrist frame + our state -> the lerobot observation dict the checkpoint's preprocessor expects."""
    return {"observation.state": torch.as_tensor(to_so100(state_deg), dtype=torch.float32)[None],
            "observation.images.camera_0": prep_image(rgb, aspect)}


# ------------------------------------------------------------------------------------------ model
def load(dtype=torch.float32, vlm_dtype=VLM_DTYPE):
    from huggingface_hub import snapshot_download
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    import lerobot.policies.pi05.processor_pi05  # noqa: F401  (registers the sentencepiece step)
    from lerobot.policies.factory import make_pre_post_processors
    import sentencepiece
    from pi05flow import Pi05Flow, DEVICE
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = snapshot_download(REPO)
    cfg = PI05Config.from_pretrained(root)
    cfg.device, cfg.dtype, cfg.compile_model, cfg.gradient_checkpointing = "cpu", "float32", False, False   # precision is a string at this commit
    cfg.text_tokenizer_name = os.path.join(root, "tokenizer", "tokenizer.model")
    t0 = time.time()
    pol = PI05Policy.from_pretrained(root, config=cfg)
    pre, post = make_pre_post_processors(cfg, pretrained_path=root,
                                         preprocessor_overrides={"device_processor": {"device": str(DEVICE)}})
    f = Pi05Flow(pol, pre, post, dtype=dtype, vlm_dtype=vlm_dtype)
    f.m.to(f.dev)                                              # stragglers (dtype unchanged)
    f.sp = sentencepiece.SentencePieceProcessor(model_file=cfg.text_tokenizer_name)
    f.root = root
    assert f.H == H and f.A == A, (f.H, f.A)
    log(f"{REPO} loaded in {time.time() - t0:.0f}s: prefix LM {vlm_dtype}, expert {dtype}, "
        f"GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB | action q01 {np.round(f.a_shift - f.a_scale, 1).tolist()} "
        f"q99 {np.round(f.a_shift + f.a_scale, 1).tolist()} (old SO-100 degrees)")
    return f


_MODEL = None


def get_model():
    """The loaded flow map, shared by the commands (loading takes minutes)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = load(torch.float32)
    return _MODEL


def chunk_deg_t(f, a):
    """Normalised chunk [..., H, 6] (torch) -> our degrees [..., H, 6] (gripper 0-100), differentiable."""
    sign, off, roll0 = so100_units()
    sc = torch.as_tensor(f.a_scale, dtype=a.dtype, device=a.device)
    sh = torch.as_tensor(f.a_shift, dtype=a.dtype, device=a.device)
    old = a * sc + sh
    ours = (old - torch.as_tensor(off, dtype=a.dtype, device=a.device)) * torch.as_tensor(sign, dtype=a.dtype, device=a.device)
    return ours + torch.as_tensor([0, 0, 0, 0, roll0, 0], dtype=a.dtype, device=a.device)


def chunk_deg(f, a):
    """numpy version for [..., H, 6] normalised chunks."""
    return chunk_deg_t(f, torch.as_tensor(np.asarray(a), dtype=torch.float64)).numpy()


# ------------------------------------------------------------------------------------------ kinematics
def quat_to_mat(q):
    """[..., 4] (w, x, y, z) -> [..., 3, 3]"""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def axis_angle_mat(axis, ang):
    """axis [3], ang [...] -> [..., 3, 3] (Rodrigues)"""
    a = axis / axis.norm()
    K = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=ang.dtype, device=ang.device)
    s, c = torch.sin(ang)[..., None, None], torch.cos(ang)[..., None, None]
    I = torch.eye(3, dtype=ang.dtype, device=ang.device)
    return I + s * K + (1 - c) * (K @ K)


def mv(R, v):
    """Batched matrix-vector product: R [..., 3, 3], v [3] or [..., 3] -> [..., 3]."""
    v = v.expand(*R.shape[:-2], 3) if v.dim() == 1 else v
    return (R @ v.unsqueeze(-1)).squeeze(-1)


class Kin:
    """SO-101 forward kinematics in torch (any batch shape), read from the MJCF compiled by MuJoCo in
    simlab's ArmModel: the body chain from the world to the `tcp` site, hinge joints about jnt_axis
    through jnt_pos. Checked against MuJoCo's own kinematics in `check`."""

    def __init__(self, arm=None, device="cpu", dtype=torch.float64):
        import mujoco
        from so101.robot_model import ArmModel, ARM_JOINTS
        self.arm = arm or ArmModel()
        m = self.arm.model
        tcp = m.site("tcp").id
        b = m.site_bodyid[tcp]
        chain = []
        while b != 0:
            js = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b]
            chain.append((m.body_pos[b].copy(), m.body_quat[b].copy(),
                          [(m.jnt_pos[j].copy(), m.jnt_axis[j].copy(), m.joint(j).name) for j in js]))
            b = m.body_parentid[b]
        self.chain = chain[::-1]
        self.names = [n for _, _, js in self.chain for _, _, n in js]
        assert self.names == ARM_JOINTS, self.names
        self.site = (m.site_pos[tcp].copy(), m.site_quat[tcp].copy())
        self._cache = {}

    def to(self, device, dtype):
        """Kept for compatibility: fk follows the device / dtype of its input."""
        return self

    def _consts(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._cache:
            T = lambda x: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)  # noqa: E731
            self._cache[key] = ([(T(p), quat_to_mat(T(q)), [(T(jp), T(ax)) for jp, ax, _ in js]) for p, q, js in self.chain],
                                T(self.site[0]), quat_to_mat(T(self.site[1])))
        return self._cache[key]

    def fk(self, q):
        """q [..., 5] rad -> tcp position [..., 3] m in the base frame (same device / dtype as q)."""
        chain, sp, sR = self._consts(q.device, q.dtype)
        shape = q.shape[:-1]
        R = torch.eye(3, dtype=q.dtype, device=q.device).expand(*shape, 3, 3)
        p = torch.zeros(*shape, 3, dtype=q.dtype, device=q.device)
        k = 0
        for bp, bR, js in chain:
            p = p + mv(R, bp)
            R = R @ bR
            for jp, ax in js:
                Rj = axis_angle_mat(ax, q[..., k])
                p = p + mv(R, jp - mv(Rj, jp))
                R = R @ Rj
                k += 1
        return p + mv(R, sp)


def path_cm_t(f, kin, a):
    """Normalised chunk [..., H, 6] -> tcp path [..., H, 3] in cm (differentiable)."""
    deg = chunk_deg_t(f, a)
    return kin.fk(torch.deg2rad(deg[..., :5])) * 100.0


def path_cm(f, kin, a):
    with torch.no_grad():
        return path_cm_t(f, kin, torch.as_tensor(np.asarray(a), dtype=torch.float64)).numpy()


def state_path_cm(kin, state_deg):
    with torch.no_grad():
        return (kin.fk(torch.deg2rad(torch.as_tensor(np.asarray(state_deg)[:5], dtype=torch.float64))) * 100.0).numpy()


# ------------------------------------------------------------------------------------------ observations
def get_observation(args):
    """(rgb frame, our state deg) from the live camera / bus, or from a file (+ --state)."""
    state = np.array(args.state, np.float32) if getattr(args, "state", None) else REST_DEG.copy()
    if getattr(args, "frame", None):
        import cv2
        rgb = cv2.cvtColor(cv2.imread(args.frame), cv2.COLOR_BGR2RGB)
        return rgb, state, f"frame {args.frame}"
    from arm7 import Arm
    arm = Arm(dry_run=True, bus=not getattr(args, "no_bus", False), camera=True, quiet=True)
    rgb = arm.frame()
    src = "live camera"
    if arm.rb is not None and not getattr(args, "state", None):
        state = arm.state_deg()
        src += " + bus"
    arm.close()
    os.makedirs(OUT, exist_ok=True)
    import cv2
    cv2.imwrite(os.path.join(OUT, "last_frame.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return rgb, state, src


# ------------------------------------------------------------------------------------------ prompt space
def prompt_slots(f, obs, text):
    """Indices in E of the prompt's own tokens (between 'Task:' and ', State:'), and their pieces."""
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
    ids = f.batch([obs], [text])[OBS_LANGUAGE_TOKENS][0].cpu().numpy()
    n_img = sum(n * n for n in f.n_img)
    pcs = [f.sp.id_to_piece(int(i)) for i in ids]
    j = pcs.index("▁State")
    k = j - 1 if pcs[j - 1] == "," else j
    return list(range(n_img + 3, n_img + k)), pcs[3:k]


def plans(f, kin, E, pad, att, zp, bs=4):
    """E [N, P, W] (pad/att [1, P] broadcast) -> plans [N, n_z, 150] (tcp path, cm), batch-1 prefixes."""
    from pi05flow import cexpand
    out = []
    with torch.no_grad():
        for i in range(E.shape[0]):
            c = f.prefix_from(E[i:i + 1], pad, att)
            a = f.flow(f._z(zp), cexpand(c, len(zp)))[..., :f.A]
            out.append(path_cm_t(f, kin.to(f.dev, a.dtype), a).flatten(1))
            del c
    return torch.stack(out)


def plan_from_E(f, kin, E, pad, att, zp):
    """Differentiable: E [1, P, W] -> mean plan over the noises zp, [150]."""
    from pi05flow import cexpand
    c = cexpand(f.prefix_from(E, pad, att), len(zp))
    a = f.flow(f._z(zp), c)[..., :f.A]
    return path_cm_t(f, kin.to(f.dev, a.dtype), a).flatten(1).mean(0)


def top_dir_E(f, kin, E, pad, att, slots, zp, n_iter=4):
    """Top right singular direction (over the prompt slots) and value of E -> mean plan, by power
    iteration with autograd only: J v by forward mode (double backward), J^T w by reverse mode."""
    r = E[0, slots].norm(dim=1).mean()

    def Jv(v):
        Eg = E.detach().requires_grad_(True)
        with torch.enable_grad():
            p = plan_from_E(f, kin, Eg, pad, att, zp)
            u = torch.zeros_like(p, requires_grad=True)
            g, = torch.autograd.grad(p @ u, Eg, create_graph=True)
            jv, = torch.autograd.grad((g[0, slots] * v).sum(), u)
        return jv.detach()

    def JTw(w):
        Eg = E.detach().requires_grad_(True)
        with torch.enable_grad():
            p = plan_from_E(f, kin, Eg, pad, att, zp)
            g, = torch.autograd.grad(p @ w, Eg)
        return g[0, slots].detach()

    v = torch.randn(len(slots), E.shape[2], device=E.device, dtype=E.dtype); v /= v.norm()
    s1 = float("nan")
    for _ in range(n_iter):
        w = Jv(v)
        u = JTw(w / max(float(w.norm()), 1e-12))
        s1 = float(u.norm())                      # |J^T w_hat| = sigma_1 once v has converged
        v = u / u.norm()
    return v, s1, float(r)


ALPHAS = [0.025, 0.05, 0.1, 0.2, 0.4]
D_JUMP = float(os.environ.get("D_JUMP", 10.0))   # cm (150-D path norm) for a ridge; LIBERO used 20 on a bigger arm. At this
                                                  # arm's decision states every prompt jumps 16-40 cm at alpha 0.025: D_JUMP=30 keeps the strongest


def is_step(jump):
    """jump: cm along the alphas; a ridge = a step, not a ramp (robot5/monitor5 criterion, scaled)."""
    return bool(jump[0] >= D_JUMP and jump[0] >= 0.6 * jump[2])


def ridge_test(f, kin, obs, text, zp, n_iter=4, alphas=ALPHAS, v1=None, with_random=True):
    """Prompt-space ridge at one observation: sigma_1 of dplan/dE_prompt, the plan displacement along
    +-v1 and +-a random direction at the alphas (fractions of the token norm), the two side paths."""
    E, pad, att = f.embed([obs], [text])
    slots, pcs = prompt_slots(f, obs, text)
    t0 = time.time()
    if v1 is None:
        v1, s1, r = top_dir_E(f, kin, E, pad, att, slots, zp, n_iter)
    else:
        r = float(E[0, slots].norm(dim=1).mean()); s1 = float("nan")
    t_v1 = time.time() - t0
    rd = torch.randn_like(v1); rd /= rd.norm()
    dirs = (v1, -v1, rd, -rd) if with_random else (v1, -v1)
    Es = []
    for d in dirs:
        for a in alphas:
            Ep = E.clone(); Ep[0, slots] += a * r * d; Es.append(Ep)
    t0 = time.time()
    P = plans(f, kin, torch.cat(Es + [E]), pad, att, zp).mean(1)            # [len(dirs)*len(alphas)+1, 150]
    t_pl = time.time() - t0
    p0 = P[-1]
    d = (P[:-1] - p0).norm(dim=1).view(len(dirs), len(alphas)).double().cpu().numpy()
    dv = d[:2].max(0)
    dr = d[2:].max(0) if with_random else np.full(len(alphas), np.nan)
    side = (P[0:len(alphas)] - P[len(alphas):2 * len(alphas)]).norm(dim=1).double().cpu().numpy()   # |P+ - P-|
    j = next((k for k, x in enumerate(side) if x >= D_JUMP), len(alphas) - 1)
    Pp, Pm = P[j].view(H, 3).double().cpu().numpy(), P[len(alphas) + j].view(H, 3).double().cpu().numpy()
    jump = [a for a, x in zip(alphas, dv) if x >= D_JUMP]
    return dict(text=text, tokens=[p.replace("▁", "") for p in pcs], sigma1=s1, r=r, alphas=list(alphas),
                d_v=dv.tolist(), d_rand=dr.tolist(), side=side.tolist(), alpha_star=jump[0] if jump else float("inf"),
                step=is_step(dv), alpha_side=alphas[j], path0=p0.view(H, 3).double().cpu().numpy().tolist(),
                path_plus=Pp.tolist(), path_minus=Pm.tolist(), t_v1=t_v1, t_plans=t_pl), v1


# ------------------------------------------------------------------------------------------ noise space
S_GRID = np.round(np.arange(-40, 41) * 0.1, 10)
I0 = 40


def ftle_scan(f, kin, obs, text, n_seed=4, steps=15, seed=0, bs=1, chunk=50):
    """Per seed: Jacobian of the first `steps` chunk steps wrt the 1600-D noise (and of the full chunk),
    FTLE, v1; the label-free ridge test of robot2: the tcp-path jump between neighbouring points
    along z + s v1, compared with a random direction."""
    from pi05flow import ftle
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_seed, H * D_ACT))
    c = f.prefix([obs], [text])
    t0 = time.time()
    a0, J = f.jac(z, c, steps=steps, bs=bs, chunk=chunk)                   # [n, steps*6, 1600]
    tj = time.time() - t0
    _, sv, Vt = np.linalg.svd(J, full_matrices=False)
    v1 = Vt[:, 0]
    res = dict(seeds=n_seed, steps=steps, ftle_exec=np.log(sv[:, 0]).tolist(), sv_exec=sv[:, :6].tolist(), t_jac=tj)
    if steps < H:
        t0 = time.time()
        _, Jf = f.jac(z, c, steps=H, bs=bs, chunk=chunk)
        res["t_jac_full"] = time.time() - t0
        svf = np.linalg.svd(Jf, compute_uv=False)
        res["ftle_full"] = np.log(svf[:, 0]).tolist()
        del Jf
    # energy of v1 in the task noise dims (robot4: pi0's flow only listens to the task dims)
    e = v1.reshape(n_seed, H, D_ACT)
    res["v1_task_energy"] = (np.linalg.norm(e[:, :, :A].reshape(n_seed, -1), axis=1) ** 2).tolist()
    rd = rng.standard_normal(v1.shape); rd /= np.linalg.norm(rd, axis=1, keepdims=True)
    out = {}
    for name, dirs in (("v1", v1), ("random", rd)):
        zz = (z[:, None] + S_GRID[None, :, None] * dirs[:, None]).reshape(-1, H * D_ACT)
        with torch.no_grad():
            a = f.flow_np(zz, c, bs=64).reshape(n_seed, len(S_GRID), H, A)
        p = path_cm(f, kin, a)                                               # [n, S, H, 3]
        pe = p[:, :, :steps].reshape(n_seed, len(S_GRID), -1)
        jump = np.linalg.norm(np.diff(pe, axis=1), axis=-1)                  # [n, S-1] cm per 0.1 of s
        disp = np.linalg.norm(pe - pe[:, I0:I0 + 1], axis=-1)                # [n, S]
        out[name] = dict(max_jump=jump.max(1).tolist(), s_max_jump=S_GRID[jump.argmax(1) + 1].tolist(),
                         disp_at_1=disp[:, I0 + 10].tolist(), disp_at_4=disp[:, -1].tolist(),
                         path_spread_cm=float(np.linalg.norm(p[:, I0, :steps] - p[:, I0, :steps].mean(0), axis=-1).mean()))
    res.update(out)
    res["path0"] = path_cm(f, kin, a0).tolist()
    return res, z, v1


# ------------------------------------------------------------------------------------------ commands
def cmd_check(args):
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    from pi05flow import ftle, N_STEPS, cslice
    rgb, state, src = get_observation(args)
    log(f"observation: {src}, frame {rgb.shape[1]}x{rgb.shape[0]}, state (ours, deg) {np.round(state, 1).tolist()} "
        f"-> old SO-100 degrees {np.round(to_so100(state), 1).tolist()}")
    f = get_model()
    kin = Kin()
    # FK vs MuJoCo
    from so101.robot_model import ArmModel
    rng = np.random.default_rng(0)
    qs = rng.uniform(kin.arm.lo, kin.arm.hi, (64, 5))
    pk = kin.fk(torch.as_tensor(qs)).numpy()
    pm = np.stack([kin.arm.fk(q)[0] for q in qs])
    log(f"torch FK vs MuJoCo, 64 random poses: max |diff| {np.abs(pk - pm).max():.2e} m")
    obs = observation(rgb, state)
    texts = [args.task]
    z = rng.standard_normal((args.seeds, H * D_ACT))
    t0 = time.time()
    with torch.no_grad():
        b = f.batch([obs], texts)
        images, img_masks = f.pol._preprocess_images(b)
        log(f"image slots {len(images)} x {tuple(images[0].shape[1:])}, masks {[bool(m[0]) for m in img_masks]}; "
            f"prompt tokens {int(b[OBS_LANGUAGE_ATTENTION_MASK].sum())}: "
            f"'{f.sp.decode([int(i) for i in b[OBS_LANGUAGE_TOKENS][0] if i > 0])[:120]}'")
        ref = np.concatenate([f.m.sample_actions(images, img_masks, b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK],
                                                 noise=f._z(z[i:i + 1]), num_steps=N_STEPS)[..., :A].double().cpu().numpy()
                              for i in range(args.seeds)])                      # one observation, one seed per call
        t1 = time.time()
        c32 = f.prefix([obs], texts)
        t2 = time.time()
        mine = f.flow_np(z, c32)
        t3 = time.time()
    log(f"explicit flow vs lerobot sample_actions (fp32, {args.seeds} seeds): max abs diff {np.abs(mine - ref).max():.2e} "
        f"(|a| max {np.abs(ref).max():.2f}) | sample_actions {t1 - t0:.2f}s, prefix {t2 - t1:.2f}s, flow x{args.seeds} {t3 - t2:.2f}s")
    deg = chunk_deg(f, mine)
    for i in range(min(2, args.seeds)):
        log(f"seed {i}: chunk starts {np.round(deg[i, 0], 1).tolist()} (state {np.round(state, 1).tolist()}), "
            f"ends {np.round(deg[i, -1], 1).tolist()}; max |start - state| {np.abs(deg[i, 0, :5] - state[:5]).max():.1f} deg")
    p = path_cm(f, kin, mine)
    p0 = state_path_cm(kin, state)
    log(f"tcp now {np.round(p0, 1).tolist()} cm; chunk paths end at {np.round(p[:, -1], 1).tolist()}; "
        f"seed spread of the path (rms over steps) {np.sqrt(((p - p.mean(0)) ** 2).sum(-1).mean()):.2f} cm")
    small = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 24e9
    vlm = f.m.paligemma_with_expert.paligemma
    if small:                        # 12 GB cards: the VLM is not needed for the Jacobian (prefix K/V cached)
        vlm.to("cpu"); torch.cuda.empty_cache()
    f.set_dtype(torch.float64)
    c = f.recast(c32)
    steps = args.steps
    t0 = time.time()
    a, J = f.jac(z[:1], c, steps=steps, bs=1, chunk=50)
    log(f"jacobian [{J.shape[1]} x {J.shape[2]}] (float64, first {steps} steps): {time.time() - t0:.1f}s | FTLE {np.round(ftle(J), 3).tolist()}")
    n_dir, eps = 6, 1e-4
    U = rng.standard_normal((n_dir, H * D_ACT)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    fd = (f.flow_np(z[0] + eps * U, c)[:, :steps] - f.flow_np(z[0] - eps * U, c)[:, :steps]) / (2 * eps)
    fd = fd.reshape(n_dir, -1)
    err = np.abs(U @ J[0].T - fd).max() / np.abs(fd).max()
    log(f"dPhi/dz (float64) vs central FD, {n_dir} random directions: max rel err {err:.2e}")
    f.set_dtype(torch.float32)
    if small:
        vlm.to(f.dev)
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(source=src, state=state.tolist(), flow_vs_lerobot=float(np.abs(mine - ref).max()),
                   fk_vs_mujoco=float(np.abs(pk - pm).max()), jac_vs_fd=float(err), ftle=ftle(J).tolist(),
                   chunk_start_deg=deg[:, 0].tolist(), path_end_cm=p[:, -1].tolist()),
              open(os.path.join(OUT, "check.json"), "w"), indent=1)
    log(f"-> {os.path.join(OUT, 'check.json')}")


def cmd_ftle(args):
    rgb, state, src = get_observation(args)
    log(f"observation: {src}, state {np.round(state, 1).tolist()}")
    f = get_model()
    kin = Kin()
    obs = observation(rgb, state)
    res, z, v1 = ftle_scan(f, kin, obs, args.task, n_seed=args.seeds, steps=args.steps)
    log(f"FTLE of the executed {args.steps} steps: {np.round(res['ftle_exec'], 2).tolist()} "
        f"(jac {res['t_jac']:.0f}s); full chunk: {np.round(res.get('ftle_full', []), 2).tolist()}")
    log(f"v1 energy in the 6 task dims: {np.round(res['v1_task_energy'], 3).tolist()}")
    for k in ("v1", "random"):
        r = res[k]
        log(f"along {k:6s}: max path jump per 0.1 s {np.round(r['max_jump'], 2).tolist()} cm at s {r['s_max_jump']}; "
            f"displacement at s=1 {np.round(r['disp_at_1'], 1).tolist()} cm, at s=4 {np.round(r['disp_at_4'], 1).tolist()} cm")
    log(f"seed-to-seed path spread (executed steps): {res['v1']['path_spread_cm']:.2f} cm")
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "ftle.npz"), z=z, v1=v1, state=state, **{k: np.asarray(v) for k, v in res.items() if not isinstance(v, dict)})
    json.dump(dict(source=src, task=args.task, state=state.tolist(), **res), open(os.path.join(OUT, "ftle.json"), "w"), indent=1)
    log(f"-> {os.path.join(OUT, 'ftle.json')}")


def cmd_ridge(args):
    rgb, state, src = get_observation(args)
    log(f"observation: {src}, state {np.round(state, 1).tolist()}")
    f = get_model()
    kin = Kin()
    obs = observation(rgb, state)
    zp = np.random.default_rng(0).standard_normal((args.nz, H * D_ACT))
    rows = []
    for text in args.task if isinstance(args.task, list) else [args.task]:
        res, v1 = ridge_test(f, kin, obs, text, zp, n_iter=args.iters)
        log(f"'{text}' tokens {res['tokens']}: sigma1 {res['sigma1']:.2f} (v1 {res['t_v1']:.0f}s), token norm {res['r']:.0f}")
        log(f"   plan displacement (cm) along +-v1 {np.round(res['d_v'], 1).tolist()} vs random {np.round(res['d_rand'], 1).tolist()} "
            f"at alpha {res['alphas']} ({res['t_plans']:.0f}s); |P+ - P-| {np.round(res['side'], 1).tolist()}; "
            f"alpha* {res['alpha_star']}; step: {res['step']}")
        p0, pp, pm = np.array(res["path0"]), np.array(res["path_plus"]), np.array(res["path_minus"])
        log(f"   path ends (cm): executed {np.round(p0[-1], 1).tolist()}, +v1 side {np.round(pp[-1], 1).tolist()}, "
            f"-v1 side {np.round(pm[-1], 1).tolist()} (alpha {res['alpha_side']})")
        os.makedirs(OUT, exist_ok=True)
        res["v1_file"] = os.path.join(OUT, "v1_" + "".join(ch if ch.isalnum() else "_" for ch in text)[:40] + ".npy")
        np.save(res["v1_file"], v1.double().cpu().numpy())
        rows.append(res)
    json.dump(dict(source=src, state=state.tolist(), results=rows), open(os.path.join(OUT, "ridge.json"), "w"), indent=1)
    log(f"-> {os.path.join(OUT, 'ridge.json')}")



# ------------------------------------------------------------------------------------------ decoding the ridge into words
DICT = "/usr/share/dict/american-english"
MAX_ID = 50000                                        # Gemma piece ids track frequency: a common-word prior (monitor5)
SHORT = 4


def dictionary(f):
    """Vocabulary pieces that are plain dictionary words (lower case, >= 2 letters, common), and the
    embedding table (the rows pi0.5 feeds into the prefix: embed_language_tokens is a bare lookup)."""
    import re
    sp = f.sp
    V = sp.get_piece_size()
    pieces = [sp.id_to_piece(i) for i in range(V)]
    dic = {w.strip().lower() for w in open(DICT, errors="ignore") if w.strip().isalpha()} if os.path.exists(DICT) else None
    ok = np.array([re.fullmatch(r"▁[a-z]{2,}", pc) is not None and (dic is None or pc[1:] in dic) and i < MAX_ID
                   for i, pc in enumerate(pieces)])
    emb = f.m.paligemma_with_expert.paligemma.model.language_model.get_input_embeddings()
    with torch.no_grad():
        Evoc = torch.cat([emb(torch.arange(i, min(i + 32768, V), device=f.dev)) for i in range(0, V, 32768)])
    return pieces, ok, Evoc


def grad_E(f, kin, E, pad, att, zp, d):
    """gradient of mean_z <plan(z; E), d> with respect to E -> [P, W]"""
    Eg = E.detach().requires_grad_(True)
    with torch.enable_grad():
        obj = plan_from_E(f, kin, Eg, pad, att, zp) @ d
        g, = torch.autograd.grad(obj, Eg)
    return g[0].detach()


def decode(f, kin, obs, text, zp, voc, n_iter=4, alphas=ALPHAS):
    """Candidate-free, simulator-free: the two prompts that select the two sides of the ridge along v1
    (robot5/monitor5): plans P+ / P- on either side, Delta = P+ - P-, one gradient, first-order score of
    every dictionary word in every prompt slot, verification by plan projection (one batch)."""
    pieces, ok, Evoc = voc
    E, pad, att = f.embed([obs], [text])
    slots, pcs = prompt_slots(f, obs, text)
    v1, s1, r = top_dir_E(f, kin, E, pad, att, slots, zp, n_iter)
    Es = []
    for a in alphas:
        for sgn in (1, -1):
            Ep = E.clone(); Ep[0, slots] += sgn * a * r * v1; Es.append(Ep)
    P = plans(f, kin, torch.cat(Es + [E]), pad, att, zp).mean(1)                  # [2*len(alphas)+1, 150]
    side = (P[0:-1:2] - P[1:-1:2]).norm(dim=1).double().cpu().numpy()             # |P+ - P-| per alpha
    j = next((k for k, x in enumerate(side) if x >= D_JUMP), len(alphas) - 1)
    Pp, Pm = P[2 * j], P[2 * j + 1]
    d = Pp - Pm; sep = float(d.norm())
    g = grad_E(f, kin, E, pad, att, zp, d / max(sep, 1e-9))
    cands = {}
    for i, (k, pc) in enumerate(zip(slots, pcs)):
        sc = ((Evoc.to(E.dtype) - E[0, k]) @ g[k].to(E.dtype)).double().cpu().numpy()
        sc[~ok] = np.nan
        if pc in pieces:
            sc[pieces.index(pc)] = np.nan
        order = np.argsort(np.nan_to_num(sc, nan=0.0))
        for sign, vs in (("+", order[-SHORT:][::-1]), ("-", order[:SHORT])):
            for v in vs:
                if np.isnan(sc[v]):
                    continue
                new = pcs.copy(); new[i] = pieces[v]
                tx = "".join(new).replace("▁", " ").strip()
                cands[tx] = dict(slot=pc.replace("▁", ""), word=pieces[v][1:], sign=sign, fo=float(sc[v]), piece_id=int(v), slot_idx=k)
    Ec = []
    for tx, c in cands.items():
        Ep = E.clone(); Ep[0, c["slot_idx"]] = Evoc[c["piece_id"]].to(E.dtype); Ec.append(Ep)
    Pc = plans(f, kin, torch.cat(Ec), pad, att, zp).mean(1)
    tw = ((Pc - Pm).norm(dim=1) - (Pc - Pp).norm(dim=1)).double().cpu().numpy() / max(sep, 1e-9)
    texts = list(cands.keys())
    for tx, t_ in zip(texts, tw):
        cands[tx]["toward"] = float(t_)
    ip, im = int(np.argmax(tw)), int(np.argmin(tw))
    lam0 = float(((P[-1] - Pm) @ d) / max(float(d @ d), 1e-9))
    return dict(text=text, tokens=[pc.replace("▁", "") for pc in pcs], sigma1=s1, r=r, alpha=alphas[j], side=side.tolist(),
                sep=sep, lam0=lam0, cplus=texts[ip], cminus=texts[im], toward_plus=float(tw[ip]), toward_minus=float(tw[im]),
                end_plus=Pp.view(H, 3)[-1].double().cpu().numpy().tolist(), end_minus=Pm.view(H, 3)[-1].double().cpu().numpy().tolist(),
                end0=P[-1].view(H, 3)[-1].double().cpu().numpy().tolist(),
                cands=sorted([dict(text=tx, **c) for tx, c in cands.items()], key=lambda c: -c["toward"])), v1


def cmd_decode(args, rgb=None, state=None, src=None):
    if rgb is None:
        rgb, state, src = get_observation(args)
        log(f"observation: {src}, state {np.round(state, 1).tolist()}")
    f = get_model()
    kin = Kin()
    obs = observation(rgb, state)
    zp = np.random.default_rng(0).standard_normal((args.nz, H * D_ACT))
    t0 = time.time()
    voc = dictionary(f)
    log(f"vocabulary: {int(voc[1].sum())} dictionary words of {len(voc[0])} pieces ({time.time() - t0:.0f}s)")
    rows = []
    for text in (args.decode if isinstance(args.decode, list) else [args.decode]):
        t0 = time.time()
        res, v1 = decode(f, kin, obs, text, zp, voc, n_iter=args.iters)
        os.makedirs(OUT, exist_ok=True)
        res["v1_file"] = os.path.join(OUT, "v1dec_" + "".join(ch if ch.isalnum() else "_" for ch in text)[:40] + ".npy")
        np.save(res["v1_file"], v1.double().cpu().numpy())
        log(f"'{text}': sigma1 {res['sigma1']:.2f}, |P+ - P-| {res['sep']:.0f} cm at alpha {res['alpha']}, lambda0 {res['lam0']:.2f} ({time.time() - t0:.0f}s)")
        log(f"   +side -> '{res['cplus']}' (toward {res['toward_plus']:+.2f}), -side -> '{res['cminus']}' (toward {res['toward_minus']:+.2f})")
        log(f"   path ends (cm): executed {np.round(res['end0'], 1).tolist()}, +v1 {np.round(res['end_plus'], 1).tolist()}, -v1 {np.round(res['end_minus'], 1).tolist()}")
        top = [f"{c['text']!r} {c['toward']:+.2f}" for c in res["cands"][:4]] + ["..."] + [f"{c['text']!r} {c['toward']:+.2f}" for c in res["cands"][-4:]]
        log("   candidates by plan projection: " + " | ".join(top))
        rows.append(res)
    os.makedirs(OUT, exist_ok=True)
    tag = "".join(ch if ch.isalnum() else "_" for ch in rows[0]["text"])[:30]
    json.dump(dict(source=src, state=None if state is None else np.asarray(state).tolist(), results=rows),
              open(os.path.join(OUT, f"decode_{tag}.json"), "w"), indent=1)
    log(f"-> {os.path.join(OUT, f'decode_{tag}.json')}")


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "ftle", "ridge", "decode", "all", "scan"],
                    help="all = check + ftle + ridge, scan = ftle + ridge + decode, with one model load")
    ap.add_argument("--decode", nargs="*", default=None, help="prompts to decode into the two side prompts (default: the --task list)")
    ap.add_argument("--live", action="store_true", help="wrist camera (+ bus if it answers)")
    ap.add_argument("--no-bus", action="store_true", help="--live without the servo bus (state = rest or --state)")
    ap.add_argument("--frame", help="saved RGB frame instead of the camera")
    ap.add_argument("--state", type=float, nargs=6, metavar="V", help="our state: pan lift elbow flex roll (deg) gripper (0-100)")
    ap.add_argument("--task", nargs="+", default=[TASK])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=15, help="executed chunk steps for the Jacobian / FTLE")
    ap.add_argument("--nz", type=int, default=2, help="noises averaged in the prompt-space plan")
    ap.add_argument("--iters", type=int, default=4, help="power iterations for v1")
    return ap


if __name__ == "__main__":
    args = parser().parse_args()
    tasks = list(args.task)
    if args.decode is None:
        args.decode = tasks
    if args.cmd in ("check", "all"):
        args.task = tasks[0]
        cmd_check(args)
    if args.cmd in ("ftle", "all", "scan"):
        args.task = tasks[0]
        cmd_ftle(args)
    if args.cmd in ("ridge", "all", "scan"):
        args.task = tasks
        cmd_ridge(args)
    if args.cmd in ("decode", "scan"):
        cmd_decode(args)
