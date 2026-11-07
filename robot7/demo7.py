"""
Assemble the choice / ask-for-help demo from a multi-episode live7 run: a title card, the question
episode (the monitor stops the arm and draws the two candidate paths into the wrist view), then each
answer executed (embedding held on either side of the ridge, and the re-prompted words), each with a
caption, and an end card with the outcomes.

    python demo7.py runs/<timestamp> [--out videos/demo_<timestamp>.mp4]
"""
import os, sys, json, glob, argparse, subprocess
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
W, Hh = 1440, 540
FPS = 8


def card(lines, seconds, big=0.9):
    img = np.full((Hh, W, 3), 24, np.uint8)
    y = 150
    for i, ln in enumerate(lines):
        sc = big if i == 0 else 0.62
        cv2.putText(img, ln, (60, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (235, 235, 235), 2 if i == 0 else 1, cv2.LINE_AA)
        y += 60 if i == 0 else 38
    return [img] * int(seconds * FPS)


def caption(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, Hh - 44), (W, Hh), (0, 0, 0), -1)
    cv2.putText(img, text, (14, Hh - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def clip(path, text, max_s=None):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, fr = cap.read()
        if not ok or (max_s and len(out) >= max_s * FPS):
            break
        if fr.shape[:2] != (Hh, W):
            fr = cv2.resize(fr, (W, Hh))
        out.append(caption(fr, text))
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out")
    ap.add_argument("--libero", help="robot5/figs/ask_pi05.mp4: append the same loop in LIBERO as the simulation counterpart")
    ap.add_argument("--question", help="a single-episode run dir whose stop-and-ask output (ask_*.png, run.json) and video stand in for episode 0")
    a = ap.parse_args()
    run = a.run.rstrip("/")
    ts = os.path.basename(run)
    eps = json.load(open(os.path.join(run, "episodes.json")))["episodes"]
    rep = {r["ep"]: r for r in json.load(open(os.path.join(run, "report.json")))} if os.path.exists(os.path.join(run, "report.json")) else {}
    out = a.out or os.path.join(HERE, "videos", f"demo_{ts}.mp4")
    frames = []
    e0 = eps[0]
    frames += card([f"pi0.5 (SO-100/101 generalist) on the real SO-101, two identical bowls",
                    f"prompt: \"{e0['task']}\"   (no simulator; the policy as an explicit flow map)",
                    "1. the monitor: is the plan on a ridge in prompt space? do the two sides go to different bowls?"], 4)
    qdir = a.question.rstrip("/") if a.question else os.path.join(run, "ep0")
    v0 = os.path.join(HERE, "videos", (os.path.basename(qdir) if a.question else f"{ts}_ep0") + ".mp4")
    if os.path.exists(v0):
        frames += clip(v0, f"\"{e0['task']}\": the arm holds while the prompt-space ridge test runs", max_s=14)
    asks = sorted(glob.glob(os.path.join(qdir, "ask_*_full.png"))) or sorted(glob.glob(os.path.join(qdir, "ask_*.png")))
    if a.question and os.path.exists(os.path.join(qdir, "run.json")):
        e0 = dict(e0, asked=json.load(open(os.path.join(qdir, "run.json"))).get("asked"))
    if asks:
        q = cv2.imread(asks[-1])
        q = cv2.resize(q, (960, 540))
        panel = np.full((Hh, 480, 3), 24, np.uint8)
        lines = ["The question, from one observation:", "", "blue  = the plan with the prompt embedding", "        pushed to the + side of the ridge",
                 "orange = pushed to the - side", "white = the plan as prompted", "", "The sides split left / right: a few cm in", "this chunk, a bowl apart once executed.",
                 "Which bowl do you mean?"]
        dec = (e0.get("asked") or {}).get("decoded")
        if dec:
            lines += ["", "decoded words for the sides:", f"  + '{dec['cplus']}'", f"  - '{dec['cminus']}'"]
        for i, ln in enumerate(lines):
            cv2.putText(panel, ln, (14, 40 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        frames += [caption(np.concatenate([q, panel], 1), "the robot stops and asks")] * (6 * FPS)
    for e in eps[1:]:
        v = os.path.join(HERE, "videos", f"{ts}_ep{e['ep']}.mp4")
        if not os.path.exists(v):
            continue
        side = e.get("side")
        if side == "+":
            text = f"answer A: embedding held on the + side (blue path)   prompt still \"{e['task']}\""
        elif side == "-":
            text = f"answer B: embedding held on the - side (orange path)   prompt still \"{e['task']}\""
        else:
            text = f"re-prompted with words: \"{e['task']}\""
        r = rep.get(e["ep"])
        if r:
            text += f"   -> tcp ends at y {r['final'][1]:+.1f} cm"
        frames += clip(v, text, max_s=30)
    if a.libero and os.path.exists(a.libero):
        frames += card(["The same loop in simulation (robot5, LIBERO-goal, pi0.5)",
                        "\"move the bowl\": the ridge appears at t 50 when the plan reaches past the grasp,",
                        "the robot stops at t 56, the two paths are drawn, and holding the prompt embedding",
                        "on either side sends it to the stove 3/3 or to the plate 3/3"], 4)
        cap = cv2.VideoCapture(a.libero)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        k = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if int(k * FPS / src_fps) > int((k - 1) * FPS / src_fps):      # resample to FPS
                h, w = fr.shape[:2]
                sc = min(W / w, Hh / h)
                fr2 = cv2.resize(fr, (int(w * sc), int(h * sc)))
                canvas = np.full((Hh, W, 3), 24, np.uint8)
                y0, x0 = (Hh - fr2.shape[0]) // 2, (W - fr2.shape[1]) // 2
                canvas[y0:y0 + fr2.shape[0], x0:x0 + fr2.shape[1]] = fr2
                frames.append(caption(canvas, "LIBERO (robot5/figs/ask_pi05.mp4): stop-and-ask on 'move the bowl', then the two answers executed"))
            k += 1
        cap.release()
    lines = ["Outcomes (tcp at the end of each episode, base frame, y = left)"]
    for e in eps:
        r = rep.get(e["ep"])
        if r:
            lines.append(f"ep{e['ep']}  {e['task']!r:34s} held {str(e.get('side') or '-'):2s} -> x {r['final'][0]:5.1f}  y {r['final'][1]:+5.1f}  z {r['final'][2]:4.1f} cm")
    frames += card(lines, 6, big=0.8)
    tmp = out + ".mp4v.mp4"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, Hh))
    for fr in frames:
        wr.write(fr)
    wr.release()
    ff = None
    py = os.path.expanduser("~/simlab/.venv/bin/python")
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0:
        ff = "ffmpeg"
    elif os.path.exists(py):
        r_ = subprocess.run([py, "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"], capture_output=True, text=True)
        ff = r_.stdout.strip() if r_.returncode == 0 else None
    if ff and subprocess.run([ff, "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out]).returncode == 0:
        os.remove(tmp)
    else:
        os.rename(tmp, out)
    print(f"{len(frames)} frames ({len(frames) / FPS:.0f} s) -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
