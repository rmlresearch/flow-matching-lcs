"""
One video for a multi-episode live7 run (the closed-loop-by-seed tests): per episode a title card (prompt,
seed, start pan, outcome), then the wrist camera as recorded, sped up, with the measured tcp position and
the bowl it is heading for drawn in; a summary card at the end. Outcomes come from report7's report.json
(run `report7.py <run> --bowls-cm x1 y1 x2 y2` first).

    python seedvideo7.py runs/20261107_091313 [--episodes 0 1 9 12] [--speed 2] [--out videos/seeds_....mp4]
"""
import os, sys, json, argparse, subprocess
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
W, Hh = 960, 540


def put(img, text, y, scale=0.6, color=(255, 255, 255), x=12):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def card(lines, seconds, fps, writer, bg=(28, 28, 28)):
    img = np.full((Hh, W, 3), bg, np.uint8)
    y = 90
    for text, scale, col in lines:
        put(img, text, y, scale, col, x=40); y += int(46 * scale + 16)
    for _ in range(int(seconds * fps)):
        writer.write(img)


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    for cand in ("ffmpeg", os.path.expanduser("~/simlab/.venv/bin/python")):
        if cand == "ffmpeg" and subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0:
            return "ffmpeg"
        if cand != "ffmpeg" and os.path.exists(cand):
            r = subprocess.run([cand, "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"], capture_output=True, text=True)
            if r.returncode == 0:
                return r.stdout.strip()
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--speed", type=float, default=2.0, help="playback speed (recorded at 8 fps)")
    ap.add_argument("--tail", type=float, default=3.0, help="seconds kept after the last replan")
    ap.add_argument("--out")
    ap.add_argument("--title", default="pi0.5 SO-100/101 generalist on the real SO-101, two identical bowls: the same start pose, different noise seeds")
    a = ap.parse_args()
    run = a.run.rstrip("/")
    from so101flow import Kin, state_path_cm
    kin = Kin()
    eps = json.load(open(os.path.join(run, "episodes.json")))["episodes"]
    rep = {r["ep"]: r for r in json.load(open(os.path.join(run, "report.json")))} if os.path.exists(os.path.join(run, "report.json")) else {}
    log = json.load(open(os.path.join(run, "log.json")))
    frames = [(t, os.path.join(run, name)) for t, name in log["frames"]]
    ft = np.array([f[0] for f in frames])
    joints = [(t, q, g) for t, q, g in log["joints"] if q is not None]
    jt = np.array([j[0] for j in joints])
    fps_rec = 8.0
    fps_out = fps_rec * a.speed
    out = a.out or os.path.join(HERE, "videos", f"seeds_{os.path.basename(run)}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    writer = cv2.VideoWriter(out + ".mp4v.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (W, Hh))
    sel = [e for e in eps if a.episodes is None or e["ep"] in a.episodes]
    card([(a.title, 0.62, (255, 255, 255)),
          (f"run {os.path.basename(run)}: {len(sel)} episodes, replan every 40 actions, tool floor 6 cm (no grasp)", 0.55, (200, 200, 200)),
          ("each episode: the prompt, the noise seed, the start pan, then the wrist camera at %gx" % a.speed, 0.55, (200, 200, 200)),
          ("outcome = the bowl nearest the final tcp (report7)", 0.55, (200, 200, 200))], 4.0, fps_out, writer)
    outcomes = []
    for e in sel:
        d = os.path.join(run, f"ep{e['ep']}")
        rec = json.load(open(os.path.join(d, "run.json")))
        rp = rec["replans"]
        if not rp:
            continue
        off = rec.get("t_offset", 0.0)
        r_ = rep.get(e["ep"], {})
        bowl = r_.get("bowl", "?"); q = r_.get("q_final")
        start = e.get("start_state") or rec["args"].get("start_state") or [float("nan")] * 6
        head = f"ep {e['ep']}   \"{e['task']}\"   seed {e['seed']}   start pan {start[0]:+.1f} deg"
        col = (120, 170, 255) if bowl == "left" else (80, 200, 255) if bowl == "right" else (200, 200, 200)
        card([(head, 0.62, (255, 255, 255)), (f"outcome: {bowl} bowl" + (f"  (q {q:+.1f} cm along the bowl axis)" if q is not None else ""), 0.62, col)],
             1.6, fps_out, writer)
        t_lo, t_hi = rp[0]["t"] + off - 0.5, rp[-1]["t"] + off + a.tail
        for t, name in frames:
            if t < t_lo or t > t_hi:
                continue
            img = cv2.imread(name)
            if img is None:
                continue
            img = cv2.resize(img, (W, Hh))
            tl = t - off
            k = int(np.searchsorted([r["t"] for r in rp], tl, side="right") - 1)
            j = int(np.argmin(np.abs(jt - t))) if len(jt) else -1
            put(img, head, 26, 0.55)
            if j >= 0:
                qd = np.degrees(joints[j][1])
                tcp = state_path_cm(kin, np.concatenate([qd, [0.0]]))
                put(img, f"t {max(tl, 0):4.1f} s   replan {max(k, 0)}   tcp x {tcp[0]:.1f}  y {tcp[1]:+.1f}  z {tcp[2]:.1f} cm   pan {qd[0]:+.0f} deg", 52, 0.5, (200, 230, 255))
            put(img, f"outcome: {bowl} bowl", Hh - 16, 0.6, col)
            writer.write(img)
        outcomes.append((e["task"], e["seed"], start[0], bowl))
    lines = [("outcomes", 0.7, (255, 255, 255))]
    for task in dict.fromkeys(o[0] for o in outcomes):
        row = [o for o in outcomes if o[0] == task]
        pans = sorted(set(round(o[2], 2) for o in row))
        for p in pans:
            rr = [o for o in row if round(o[2], 2) == p]
            nl = sum(o[3] == "left" for o in rr); nr = sum(o[3] == "right" for o in rr)
            lines.append((f"\"{task}\"  start pan {p:+.2f}: {nl} left / {nr} right of {len(rr)} seeds", 0.55, (220, 220, 220)))
    card(lines, 6.0, fps_out, writer)
    writer.release()
    ff = ffmpeg_exe()
    if ff:
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", out + ".mp4v.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out])
        if r.returncode == 0:
            os.remove(out + ".mp4v.mp4")
        else:
            os.rename(out + ".mp4v.mp4", out)
    else:
        os.rename(out + ".mp4v.mp4", out)
    print(f"-> {out} ({os.path.getsize(out) / 1e6:.1f} MB), {len(sel)} episodes")


if __name__ == "__main__":
    main()
