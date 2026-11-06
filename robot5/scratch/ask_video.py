"""figs/ask_pi05.mp4: the stop-and-ask loop on 'move the bowl', LIBERO-goal state 2, seed 0, as a video.
Every control step of the policy's camera view (raw render rotated 180 deg, 3x), with the chunk being
executed (dashed), the two ridge-side paths of the newest replan (grey while the jump is under the
threshold), the trigger reading of the delivered direction, the stop at t 56, the question with the
VLM's names (out/vlm_assign.json), and both answers executed (embedding held on that side, same noise
as the figure's rollouts). Overlays at the replans come from out/movebowl_fig_r10a1s6i4.npz; the physics
is replayed with the same noise. 10 fps = half speed."""
import json, os, subprocess, sys
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "/home/james/Projects/flow-matching-lcs/robot5")
import pi05flow
from pi05flow import H, D_ACT
from env5 import Scene, GOAL_NAMES, EVENT_NAMES, TERMINAL
from goal5 import path_cm
from pridge5 import prompt_slots, plans, top_dir_E
import monitor5 as M
from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera
import imageio_ffmpeg

HERE = "/home/james/Projects/flow-matching-lcs/robot5"
TEXT, INIT, SEED, STOP, ALPHA, TRIG, FPS = "move the bowl", 2, 0, 56, 0.025, 50.0, 10
A = H * D_ACT; S = 3; W = 256 * S; TOP, BOT = 118, 54
names = GOAL_NAMES + EVENT_NAMES
BLUE, ORANGE, GREY, SURF, INK, INK2 = (42, 120, 214), (235, 104, 52), (120, 118, 112), (252, 252, 251), (11, 11, 11), (82, 81, 78)
FONT = lambda n, b=False: ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if b else ''}.ttf", n)
OUTDIR = "/tmp/claude-1000/-home-james-Projects-flow-matching-lcs/fc615ee8-3dcc-4cb5-8739-f0992e1fde17/scratchpad/video_frames"


def px(p):
    p = np.asarray(p, float); return np.stack([255 - p[..., 1], p[..., 0]], -1) * S


class Frames:
    def __init__(self):
        os.makedirs(OUTDIR, exist_ok=True); [os.remove(os.path.join(OUTDIR, f)) for f in os.listdir(OUTDIR)]; self.n = 0

    def add(self, img, lines, paths=(), dashed=None, bar=None, labels=(), hold_px=None, repeat=1):
        im = Image.new("RGB", (W, TOP + W + BOT), SURF)
        cam = Image.fromarray(np.ascontiguousarray(img[::-1, ::-1])).resize((W, W), Image.BICUBIC)
        d = ImageDraw.Draw(cam)
        if dashed is not None:
            p = px(dashed)
            for i in range(0, len(p) - 1, 2):
                d.line([tuple(p[i]), tuple(p[i + 1])], fill=(255, 255, 255), width=3)
        for p, col in paths:
            p = px(p); d.line([tuple(q) for q in p], fill=col, width=6, joint="curve")
            x, y = p[-1]; d.ellipse([x - 12, y - 12, x + 12, y + 12], fill=col, outline=(255, 255, 255), width=3)
        if hold_px is not None:
            x, y = hold_px; d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(255, 255, 255), outline=INK, width=2)
        for (x, y), txt, col in labels:
            f = FONT(30, True); tw = d.textlength(txt, font=f)
            d.rectangle([x - 6, y - 4, x + tw + 6, y + 36], fill=SURF); d.text((x, y), txt, fill=col, font=f)
        im.paste(cam, (0, TOP)); d = ImageDraw.Draw(im)
        d.text((14, 8), "pi0.5 “move the bowl”, LIBERO-goal state 2 · stop and ask", fill=INK, font=FONT(19, True))
        y = 36
        for txt, col in lines:
            f16 = FONT(16); words, cur, out = txt.split(), "", []
            for w_ in words:
                if d.textlength((cur + " " + w_).strip(), font=f16) > W - 28:
                    out.append(cur); cur = w_
                else:
                    cur = (cur + " " + w_).strip()
            out.append(cur)
            for ln in out[:2]:
                d.text((14, y), ln, fill=col, font=f16); y += 20
        if bar is not None:
            val, txt = bar; x0, y0, bw = 14, TOP + W + 14, W - 28
            d.text((x0, y0 - 2), txt, fill=INK2, font=FONT(14))
            d.rectangle([x0, y0 + 18, x0 + bw, y0 + 30], fill=(230, 228, 223))
            d.rectangle([x0, y0 + 18, x0 + bw * min(val, 180) / 180, y0 + 30], fill=BLUE if val >= TRIG else GREY)
            xt = x0 + bw * TRIG / 180; d.line([xt, y0 + 14, xt, y0 + 34], fill=INK, width=2)
            d.text((xt + 4, y0 + 30), f"{TRIG:.0f} cm", fill=INK, font=FONT(12))
        for _ in range(repeat):
            im.save(os.path.join(OUTDIR, f"f{self.n:05d}.png")); self.n += 1


def main():
    f = pi05flow.load(); zp = np.random.default_rng(777).standard_normal((M.N_Z, A))
    D = np.load(f"{HERE}/out/movebowl_fig_r10a1s6i4.npz"); J = json.load(open(f"{HERE}/out/movebowl_fig_r10a1s6i4.json"))
    rows = {r["t"]: r for r in J["rows"]}; VN = json.load(open(f"{HERE}/out/vlm_assign.json"))[str(STOP)]
    s = Scene(0, INIT, 0, suite_name="libero_goal"); obs = s.reset(); x0 = s.get_state(); obs = s.set_state(x0)
    W2P = get_camera_transform_matrix(s.sim, "agentview", 256, 256)
    proj = lambda v: project_points_from_world_to_camera(np.asarray(v, float).reshape(-1, 3), W2P, 256, 256)
    rng = np.random.default_rng(1000 + SEED); F = Frames(); t = 0
    # ---- execution to the stop ----------------------------------------------------------------------------
    while t < STOP:
        o = s.obs_to_lerobot(obs, TEXT); z = rng.standard_normal((1, A))
        r = rows[t]; a = f.unnorm(f.flow_np(z, f.prefix([o], [TEXT]))[0])
        sides = [(D[f"pix_plus_{t}"], GREY), (D[f"pix_minus_{t}"], GREY)]
        ja = r["jump_async"]
        if ja is None:
            trig = (0.0, "trigger: no ridge direction delivered yet (first refresh takes 1.0 s)")
        else:
            trig = (ja[0], f"trigger: jump between the two sides of the v₁ delivered from t {r['newest_dir']}  =  {ja[0]:.0f} cm" + ("   → FIRES; result in 0.3 s" if r["fire"] else ""))
        phase = {0: "start: the policy reaches for the bowl", 10: "reaching", 20: "reaching", 30: "closing on the bowl", 40: "grasping", 50: "bowl lifted: the two sides part"}[t]
        n_step = STOP - t if r["fire"] else 10
        for j in range(n_step):
            img = np.asarray(obs["pixels"]["image"])
            lines = [(f"t {t}   {phase}", INK), (f"replan every 10 steps; grey: the chunk with the prompt embedding pushed to either side of the newest ridge direction", INK2)]
            if r["fire"]:
                lines[1] = (f"the test at t 50 fired (122 cm): its result arrives at t {STOP}; the robot keeps executing until then", BLUE)
            F.add(img, lines, paths=sides, dashed=D[f"pix_exec_{t - j}"] if j == 0 else D[f"pix_exec_{t - j}"], bar=trig, hold_px=px(proj(s.eef())[0]))
            obs, _, _ = s.step(a[j]); t += 1
    # ---- the stop: full test on the frozen state, the question -------------------------------------------
    x = s.get_state(); o = s.obs_to_lerobot(obs, TEXT); img = np.asarray(obs["pixels"]["image"])
    E, pad, att = f.embed([o], [TEXT]); slots, _ = prompt_slots(f, o, TEXT)
    torch.manual_seed(0); v4, s4, rr = top_dir_E(f, E, pad, att, slots, zp, n_iter=4)
    F.add(img, [(f"t {STOP}   STOP: the robot holds position", INK), ("full ridge test on the frozen state: v₁ (4 iterations), the two side plans, projection into the image: 3.4 s", INK2)],
          bar=(122.0, "trigger: 122 cm along the delivered direction → stopped"), hold_px=px(proj(s.eef())[0]), repeat=int(3.4 * FPS))
    col = {"plus": BLUE if VN["plus"][0] == "stove" else ORANGE, "minus": BLUE if VN["minus"][0] == "stove" else ORANGE}
    sides = [(D[f"pix_plus_{STOP}"], col["plus"]), (D[f"pix_minus_{STOP}"], col["minus"])]
    labels = []
    for side in ("plus", "minus"):
        x_, y_ = px(D[f"pix_{side}_{STOP}"])[-1]; labels.append(((x_ + (18 if VN[side][0] != "stove" else -150), y_ - 48), f"{VN[side][0]}?", col[side]))
    F.add(img, [(f"t {STOP}   the question: which way?", INK), ("the two paths are the policy's; the names come from Qwen3-VL-4B's object boxes and the paths' directions (no simulator)", INK2)],
          paths=sides, labels=labels, bar=(float(np.linalg.norm(D[f"path_plus_{STOP}"] - D[f"path_minus_{STOP}"]) * 100), "sides of the frozen-state ridge, cm apart"),
          hold_px=px(proj(s.eef())[0]), repeat=int(4 * FPS))
    # ---- both answers executed ------------------------------------------------------------------------------
    for side, sgn in (("minus", -1), ("plus", 1)):
        name = VN[side][0]; c = col[side]
        F.add(img, [(f"operator answers “{name}”", c), (f"resume with the prompt embedding held on the “{name}” side of the ridge (2.5 % of the token norm); same noise as the figure's rollout", INK2)],
              paths=[sd for sd in sides if sd[1] == c], labels=[l for l in labels if l[2] == c], hold_px=px(proj(s.eef())[0]), repeat=int(2 * FPS))
        obs2 = s.set_state(x); tt, label = STOP, None; rng2 = np.random.default_rng(90_000 + 0)
        while tt < STOP + 110 and label is None:
            o2 = s.obs_to_lerobot(obs2, TEXT); E2, pad2, att2 = f.embed([o2], [TEXT]); E2[0, slots] += sgn * ALPHA * rr * v4
            with torch.no_grad():
                c2 = f.prefix_from(E2, pad2, att2)
            a2 = f.unnorm(f.flow_np(rng2.standard_normal((1, A)), c2)[0])
            chunk = proj(s.eef() + path_cm(a2)[:50] / 100)
            for j in range(10):
                img2 = np.asarray(obs2["pixels"]["image"])
                F.add(img2, [(f"t {tt}   executing, embedding held on the “{name}” side", c), ("dashed: the chunk being executed", INK2)], dashed=chunk, hold_px=px(proj(s.eef())[0]))
                obs2, _, _ = s.step(a2[j]); tt += 1
                for i, q in enumerate(s.preds()):
                    if q and TERMINAL[i]:
                        label = names[i]; break
                if label is not None:
                    break
        img2 = np.asarray(obs2["pixels"]["image"])
        F.add(img2, [(f"t {tt}   {label.replace('bowl->', 'bowl on the ')}", c), ("", INK2)], hold_px=px(proj(s.eef())[0]), repeat=int(2.5 * FPS))
        M.log(f"answer {name}: {label} at t {tt}")
    s.close()
    out = f"{HERE}/figs/ask_pi05.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-framerate", str(FPS), "-i", os.path.join(OUTDIR, "f%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    M.log(f"wrote {out}: {F.n} frames at {FPS} fps = {F.n / FPS:.0f} s")


if __name__ == "__main__":
    main()
