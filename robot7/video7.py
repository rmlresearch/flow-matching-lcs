"""
Proof video of a live7 run: the wrist camera as recorded by RealBackend (8 fps, runs/<ts>/video/),
with the time, the measured joints, the replan index and the monitor's verdict drawn in, and, on the
right, the frame the policy saw at the current replan (the ridge overlay `ask_*.png` when the monitor
found one). Encoded with OpenCV (mp4v), then re-encoded to H.264 when imageio-ffmpeg's binary is
around (simlab's venv on the Thor), so it plays everywhere.

    python video7.py runs/20261106_201500 [--out videos/run.mp4] [--fps 8]
"""
import os, sys, json, glob, argparse, subprocess
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["pan", "lift", "elbow", "flex", "roll", "grip"]


def put(img, text, y, scale=0.55, color=(255, 255, 255)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--lead", type=float, default=2.0, help="seconds of recording to keep before the first replan (the model load is cut)")
    a = ap.parse_args()
    run = a.run.rstrip("/")
    out = a.out or os.path.join(HERE, "videos", ("_".join(run.rstrip("/").split("/")[-2:]) if os.path.basename(run).startswith("ep") else os.path.basename(run)) + ".mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rec = json.load(open(os.path.join(run, "run.json")))
    parent = os.path.dirname(run) if os.path.basename(run).startswith("ep") and not os.path.exists(os.path.join(run, "log.json")) else run
    if os.path.exists(os.path.join(parent, "log.json")):           # RealBackend: joints [(t, q rad[5], gripper rad)], frames [(t, file)]
        log = json.load(open(os.path.join(parent, "log.json")))
        log["frames"] = [(t, os.path.join(parent, name)) for t, name in log["frames"]]
    else:                                                          # no bus (camera-only dry run): one frame per replan
        log = dict(t0=0.0, joints=[], frames=[(r["t"], f"frame_{r['n']:03d}.jpg") for r in rec["replans"]])
        rec["t_offset"] = 0.0
        a.fps = 1.0
    frames = log["frames"]
    joints = [(t, q, g) for t, q, g in log["joints"] if q is not None]
    jt = np.array([j[0] for j in joints]) if joints else np.zeros(0)
    replans = rec["replans"]
    rt = np.array([r["t"] for r in replans]) if replans else np.zeros(0)
    t0_rec = log["t0"]
    # live7's clock starts at its t_start; RealBackend's at its construction. Align by the first replan's
    # frame: replan n's frame_nnn.jpg was grabbed at live7 time r["t"]; find the recorded frame nearest it.
    # Both clocks are wall clocks offset by a constant, which run.json does not store, so align on content:
    # the recorder started before the loop; replan 0 happens within a second of the first chunk. Use
    # rec["t_offset"] when present, else assume the loop started `load_s` after the recorder (the model load).
    off = rec.get("t_offset")
    if off is None:
        off = float(rec.get("t_loop_start", 0.0)) - t0_rec
    W, Hh = 960, 540
    writer = cv2.VideoWriter(out + ".mp4v.mp4", cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W + 480, Hh))
    asks = {int(os.path.basename(p)[4:7]): p for p in glob.glob(os.path.join(run, "ask_*.png")) if "_full" not in p}
    n_out = 0
    for t, name in frames:
        img = cv2.imread(name if os.path.isabs(name) or name.startswith(parent) else os.path.join(run, name))
        if img is None:
            continue
        img = cv2.resize(img, (W, Hh))
        tl = t - off                                       # live7 time
        if tl < -a.lead or (len(rt) and tl > rt[-1] + 6.0):
            continue
        k = int(np.searchsorted(rt, tl, side="right") - 1) if len(rt) else -1
        r = replans[k] if k >= 0 else None
        j = int(np.argmin(np.abs(jt - t))) if len(jt) else -1
        put(img, f"t = {max(tl, 0):5.1f} s   pi0.5 SO-100/101 generalist through the explicit flow map", 24, 0.6)
        if j >= 0:
            q = np.degrees(joints[j][1]); g = joints[j][2]
            gp = 0 if g is None else (g + 0.17) / (1.55 + 0.17) * 100
            put(img, "measured  " + "  ".join(f"{n} {v:+.0f}" for n, v in zip(NAMES, list(q) + [gp])), 48, 0.5, (200, 230, 255))
        if r is not None:
            put(img, f"replan {r['n']}  chunk -> " + "  ".join(f"{v:+.0f}" for v in r["chunk_end"]) + f"   tool {r['tool_z_cm']:.1f} cm up", 72, 0.5, (200, 255, 200))
            if "ridge" in r:
                rg = r["ridge"]
                put(img, f"prompt ridge: sigma1 {rg['sigma1']:.1f}, jump along v1 {rg['d_v'][0]:.0f} cm at 2.5% -> {'STEP: ask' if rg['step'] else 'no step'}",
                    96, 0.5, (120, 200, 255) if rg["step"] else (200, 200, 200))
        panel = np.zeros((Hh, 480, 3), np.uint8)
        src = None
        if r is not None:
            src = asks.get(r["n"]) or os.path.join(run, f"frame_{r['n']:03d}.jpg")
        if src and os.path.exists(src):
            pf = cv2.imread(src)
            if pf is not None:
                ph = int(480 * pf.shape[0] / pf.shape[1])
                panel[:min(ph, Hh), :] = cv2.resize(pf, (480, ph))[:Hh]
                cv2.putText(panel, "what the policy saw" + (" + the question" if r["n"] in asks else ""), (10, min(ph, Hh) + 24 if ph + 30 < Hh else Hh - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(np.concatenate([img, panel], 1))
        n_out += 1
    writer.release()
    # H.264 if a ffmpeg binary is around (imageio-ffmpeg in simlab's venv on the Thor, or a system ffmpeg)
    ff = None
    for cand in ("ffmpeg", os.path.expanduser("~/simlab/.venv/bin/python")):
        if cand == "ffmpeg":
            if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0:
                ff = "ffmpeg"; break
        elif os.path.exists(cand):
            r_ = subprocess.run([cand, "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"], capture_output=True, text=True)
            if r_.returncode == 0:
                ff = r_.stdout.strip(); break
    if ff:
        r_ = subprocess.run([ff, "-y", "-loglevel", "error", "-i", out + ".mp4v.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out])
        if r_.returncode == 0:
            os.remove(out + ".mp4v.mp4")
        else:
            os.rename(out + ".mp4v.mp4", out)
    else:
        os.rename(out + ".mp4v.mp4", out)
    print(f"{n_out} frames at {a.fps} fps -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
