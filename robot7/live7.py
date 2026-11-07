"""
Closed loop on the real SO-101 through the explicit flow map, with the simulator-free monitor of
robot5 running at every replan: a fresh top direction v1 of the plan with respect to the prompt
embedding, the jump test along it (the prompt-space ridge), and, when a ridge is found, the two
paths on either side of it drawn into the wrist camera image as the question to ask.

    python live7.py run --dry-run [--no-bus]            # everything but motion (frames, chunks, monitor -> runs/<ts>/)
    python live7.py run --duration 60                   # moves the arm: torque on, 3 s countdown, Ctrl-C holds the pose
    python live7.py run --monitor off                   # plain pi0.5 closed loop (seeded noise), no diagnostics

Per replan: observe (wrist frame + joints) -> prefix (batch 1) -> seeded noise -> chunk -> [monitor]
-> execute n_exec actions with simlab's clamps (ramp, low-pass, joint limits, per-step delta, tool
floor). Monitor `fast` = one power iteration for v1 and one alpha (0.025 of the token norm) along +-v1;
`full` = 4 iterations, 5 alphas and a random direction (slow, ~20 s). `--stop-on-ridge` holds the arm
at the first ridge, runs the full test on the frozen state and writes the question (runs/<ts>/ask_*.png).
"""
import os, sys, json, time, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from so101flow import (TASK, REST_DEG, H, D_ACT, A, load, Kin, observation, chunk_deg, path_cm, state_path_cm,  # noqa: E402
                       ridge_test, log, ALPHAS)

RUNS = os.path.join(HERE, "runs")


def project_paths(arm, q_rad, paths_cm, size):
    """Paths [k, H, 3] (cm, base frame) -> pixel polylines in the wrist camera at joint angles q."""
    from so101.robot_model import CameraModel
    cam_p, cam_R = arm.camera_pose(q_rad)
    cm = CameraModel(arm.fovy, *size)
    out = []
    for P in paths_cm:
        pts = []
        for p in np.asarray(P) / 100.0:
            pc = cam_R.T @ (p - cam_p)
            uv = cm.project(pc)
            pts.append(None if uv is None else (float(uv[0]), float(uv[1])))
        out.append(pts)
    return out


def draw_question(rgb, arm, state_deg, res, path):
    """The two sides of the ridge (and the executed plan) drawn into the wrist frame."""
    import cv2
    img = np.ascontiguousarray(rgb).copy()
    q = np.radians(np.asarray(state_deg)[:5])
    polys = project_paths(arm, q, [path, np.array(res["path_plus"]), np.array(res["path_minus"])], (img.shape[1], img.shape[0]))
    cols = [(255, 255, 255), (66, 133, 244), (244, 133, 66)]       # executed white, +v1 blue, -v1 orange (RGB)
    for pts, col in zip(polys, cols):
        pp = [p for p in pts if p is not None]
        for a, b in zip(pp[:-1], pp[1:]):
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, 3)
        if pp:
            cv2.circle(img, (int(pp[-1][0]), int(pp[-1][1])), 9, col, -1)
    txt = f"ridge: sigma1 {res['sigma1']:.1f}  |P+ - P-| {res['side'][0]:.0f} cm at alpha {res['alphas'][0]}  prompt '{' '.join(res['tokens'])}'"
    cv2.putText(img, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--dry-run", action="store_true", help="compute everything, move nothing")
    ap.add_argument("--no-bus", action="store_true", help="camera only (state = --state or rest; implies --dry-run)")
    ap.add_argument("--state", type=float, nargs=6, metavar="V")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--exec", type=int, default=40, help="actions executed per chunk")
    ap.add_argument("--max-step-deg", type=float, default=2.0)
    ap.add_argument("--z-min", type=float, default=0.008)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--monitor", choices=["off", "fast", "full"], default="fast")
    ap.add_argument("--iters", type=int, default=1, help="power iterations for v1 (fast monitor)")
    ap.add_argument("--reuse-v1", action="store_true", help="monitor: keep the first v1 of the episode instead of refreshing it")
    ap.add_argument("--decode-at-stop", action="store_true", help="at the stop, decode the two sides of the ridge into prompts (so101flow.decode, ~100 s)")
    ap.add_argument("--monitor-episodes", type=int, nargs="*", default=None, help="run the monitor only in these episodes (default: all)")
    ap.add_argument("--stop-on-ridge", action="store_true")
    ap.add_argument("--ask-sep", type=float, default=None,
                    help="alternative trigger for the question: the two side paths (smallest alpha) end at least this far apart "
                         "in the table plane (cm), i.e. the two sides of the ridge point at different places")
    ap.add_argument("--release", action="store_true", help="torque off at the end")
    ap.add_argument("--start-state", type=float, nargs=6, metavar="V", help="move here (our deg, gripper 0-100) before the loop")
    ap.add_argument("--hold", help="ridge.json / decode_*.json from so101flow: hold the prompt embedding on one side of the prompt's ridge")
    ap.add_argument("--side", choices=["+", "-"], default="+")
    ap.add_argument("--alpha", type=float, default=None, help="hold offset as a fraction of the token norm (default: the file's alpha)")
    ap.add_argument("--episode", nargs="+", default=None,
                    help="several episodes in one model load, each from --start-state: 'prompt' or 'prompt::+' / 'prompt::-' "
                         "(embedding held on that side of the prompt's ridge from --hold); seeds --seed, --seed+1, ...; "
                         "'prompt@pan,lift,elbow,flex,roll,gripper' gives that episode its own start state")
    ap.add_argument("--dry-replans", type=int, default=1, help="compute (and log) this many replans before the first motion")
    ap.add_argument("--max-start-jump", type=float, default=15.0,
                    help="abort before any motion if the first chunk starts further than this (deg) from the measured pose")
    args = ap.parse_args()
    if args.no_bus:
        args.dry_run = True

    run_dir = os.path.join(RUNS, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    from arm7 import Arm
    arm = Arm(dry_run=args.dry_run, bus=not args.no_bus, camera=True, run_dir=run_dir)
    f = load(torch.float32)
    kin = Kin(arm.arm)
    rng = np.random.default_rng(args.seed)
    zp = np.random.default_rng(1000 + args.seed).standard_normal((2, H * D_ACT))
    from so101flow import prompt_slots
    hj = json.load(open(args.hold)) if args.hold else None

    def make_hold(task, side):
        """The held side of a prompt's ridge: v1 from --hold's file when it has one, else computed in the run at
        the first observation of the first held episode of that prompt (and reused for the other side), oriented
        so that the + side is the one whose plan ends farther to the robot's left (+y)."""
        if side is None:
            return None
        row = next((r for r in hj["results"] if r["text"] == task), None) if hj else None
        alpha = args.alpha if args.alpha is not None else (row or {}).get("alpha_side", (row or {}).get("alpha", 0.1))
        h = dict(v1=None, r=None, alpha=float(alpha), sign=1.0 if side == "+" else -1.0, side=side, task=task)
        if row and row.get("v1_file") and os.path.exists(row["v1_file"]):
            h["v1"], h["r"] = torch.as_tensor(np.load(row["v1_file"]), device=f.dev), float(row["r"])
        elif task in V1CACHE:
            h["v1"], h["r"] = V1CACHE[task]
            log(f"   (v1 of {task!r} from this run's question / earlier held episode)")
        log(f"holding the prompt embedding on the {side} side of the ridge of {task!r}: alpha {alpha}"
            + (" (v1 computed at the first observation)" if h["v1"] is None else f" x token norm {h['r']:.0f}"))
        return h

    def prefix_for(obs, task, hold):
        if hold is None:
            return f.prefix([obs], [task])
        with torch.no_grad():
            E, pad, att = f.embed([obs], [task])
            slots, _ = prompt_slots(f, obs, task)
            E[0, slots] += hold["sign"] * hold["alpha"] * hold["r"] * hold["v1"].to(E.dtype)
            return f.prefix_from(E, pad, att)

    episodes = []
    for k, spec in enumerate(args.episode or [args.task]):
        start = args.start_state
        if "@" in spec:                                          # 'prompt@pan,lift,elbow,flex,roll,gripper': this episode's own start state
            spec, _, st = spec.rpartition("@")
            start = [float(v) for v in st.split(",")]
            if len(start) != 6:
                raise SystemExit(f"bad start state in episode spec {spec!r}@{st}")
        task, side = (spec.rsplit("::", 1) + [None])[:2] if "::" in spec else (spec, None)
        if side is not None and side not in ("+", "-"):
            raise SystemExit(f"bad episode spec {spec!r}")
        if args.episode is None and args.hold:
            side = args.side
        episodes.append((task, side, args.seed + k, start))

    if not args.dry_run:
        arm.enable()
        print("Torque ON - the policy will move the arm. Starting in 3 s (Ctrl-C holds the pose)...", flush=True)
        time.sleep(3)
    import cv2
    summary = []
    try:
      for ep, (task, side, seed, start) in enumerate(episodes):
        ep_dir = os.path.join(run_dir, f"ep{ep}") if len(episodes) > 1 else run_dir
        os.makedirs(ep_dir, exist_ok=True)
        hold = make_hold(task, side)
        if start:
            st = np.array(start, np.float32)
            raised = st.copy(); raised[1] -= 25.0                      # lift the tool clear of the objects first
            log(f"episode {ep}: moving to the start state via a raised pose")
            arm.goto(raised, 3.0)
            arm.goto(st, 3.0)
            arm.last_cmd = None
        rng = np.random.default_rng(seed)
        log(f"episode {ep}: prompt {task!r}, side {side}, seed {seed} -> {ep_dir}")
        rec_ep = run_episode(args, arm, f, kin, task, hold, rng, zp, ep_dir, ep=ep)
        summary.append(dict(ep=ep, task=task, side=side, seed=seed, dir=ep_dir, start_state=start, **rec_ep))
    except KeyboardInterrupt:
        log("interrupted - the arm holds its pose")
    finally:
        if len(episodes) > 1:
            json.dump(dict(episodes=summary, args=vars(args)), open(os.path.join(run_dir, "episodes.json"), "w"), indent=1)
        arm.close(release=args.release)
    log(f"done: {len(summary)} episode(s) -> {run_dir}")


V1CACHE = {}


def orient_sides(res, v1):
    """Make the + side of a ridge result the side whose path ends farther to the robot's left (+y), so that the
    overlay colours and the held episodes mean the same thing across replans and episodes."""
    if np.array(res["path_plus"])[-1, 1] < np.array(res["path_minus"])[-1, 1]:
        res = dict(res)
        res["path_plus"], res["path_minus"] = res["path_minus"], res["path_plus"]
        v1 = -v1
    return res, v1


def run_episode(args, arm, f, kin, task, hold, rng, zp, run_dir, ep=0):
    monitor_on = args.monitor != "off" and (not args.monitor_episodes or ep in args.monitor_episodes)
    import cv2
    from so101flow import prompt_slots, top_dir_E, plans

    def prefix_for(obs):
        if hold is None:
            return f.prefix([obs], [task])
        E, pad, att = f.embed([obs], [task])
        slots, _ = prompt_slots(f, obs, task)
        if hold["v1"] is None:                                   # first held episode of this prompt: v1 here
            t0 = time.time()
            v1, s1, r = top_dir_E(f, kin, E, pad, att, slots, zp, n_iter=4)
            Es = []
            for sgn in (1, -1):
                Ep = E.clone(); Ep[0, slots] += sgn * hold["alpha"] * r * v1; Es.append(Ep)
            P = plans(f, kin, torch.cat(Es), pad, att, zp).mean(1).view(2, H, 3).double().cpu().numpy()
            if P[0, -1, 1] < P[1, -1, 1]:                        # + side = the plan that ends farther to the left (+y)
                v1 = -v1; P = P[::-1]
            hold["v1"], hold["r"] = v1, float(r)
            V1CACHE[task] = (v1, float(r))
            hold["sides_end_cm"] = P[:, -1].tolist()
            log(f"v1 of {task!r} at the start state: sigma1 {s1:.2f}, token norm {r:.0f}, side paths end at "
                f"+ {np.round(P[0, -1], 1).tolist()} / - {np.round(P[1, -1], 1).tolist()} cm ({time.time() - t0:.0f}s)")
        with torch.no_grad():
            E[0, slots] += hold["sign"] * hold["alpha"] * hold["r"] * hold["v1"].to(E.dtype)
            return f.prefix_from(E, pad, att)

    state = np.array(args.state, np.float32) if args.state else REST_DEG.copy()
    if hold is not None and hold["v1"] is None:                 # v1 before the episode clock starts (it takes ~40 s)
        if arm.rb is not None:
            state = arm.state_deg()
        prefix_for(observation(arm.frame(), state))
    records, t_start, n = [], time.time(), 0
    t_offset = t_start - (arm.rb.t0 if arm.rb is not None else t_start)   # loop clock vs the recorder's
    asked = None
    try:
        while time.time() - t_start < args.duration:
            rgb = arm.frame()
            if arm.rb is not None:
                state = arm.state_deg()
            elif arm.last_cmd is not None:
                state = arm.last_cmd.copy()
            obs = observation(rgb, state)
            t0 = time.time()
            c = prefix_for(obs)
            z = rng.standard_normal((1, H * D_ACT))
            a = f.flow_np(z, c)[0]
            t_pol = time.time() - t0
            deg = chunk_deg(f, a)
            path = path_cm(f, kin, a)
            rec = dict(n=n, t=round(time.time() - t_start, 2), state=state.tolist(), t_policy=round(t_pol, 2),
                       chunk_start=deg[0].tolist(), chunk_end=deg[-1].tolist(), path_end=path[-1].tolist(),
                       tool_z_cm=round(arm.tool_z(state) * 100, 1))
            cv2.imwrite(os.path.join(run_dir, f"frame_{n:03d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            np.save(os.path.join(run_dir, f"chunk_{n:03d}.npy"), np.concatenate([z[0], a.flatten()]).astype(np.float32))
            msg = (f"replan {n} t {rec['t']:5.1f}s: state {np.round(state, 1).tolist()} tool {rec['tool_z_cm']:.1f} cm up | "
                   f"chunk starts {np.round(deg[0], 1).tolist()} ends {np.round(deg[-1], 1).tolist()} ({t_pol:.2f}s)")
            if monitor_on:
                fast = args.monitor == "fast"
                res, v1r = ridge_test(f, kin, obs, task, zp, n_iter=args.iters if fast else 4,
                                      alphas=[0.025, 0.05, 0.1] if fast else ALPHAS, with_random=not fast,
                                      v1=V1CACHE[task][0] if (task in V1CACHE and args.reuse_v1) else None)
                res, v1r = orient_sides(res, v1r)                     # + side = the path that ends farther left (+y)
                V1CACHE[task] = (v1r, float(res["r"]))               # the held episodes then execute exactly these sides
                ep = np.array(res["path_plus"])[-1, :2] - np.array(res["path_minus"])[-1, :2]
                res["sides_sep_xy"] = float(np.linalg.norm(ep))
                if args.ask_sep is not None:
                    res["step"] = bool(res["sides_sep_xy"] >= args.ask_sep)
                rec["ridge"] = {k: res[k] for k in ("sigma1", "d_v", "d_rand", "side", "alphas", "step", "alpha_star", "t_v1", "t_plans", "sides_sep_xy")}
                msg += (f" | ridge: sigma1 {res['sigma1']:.1f}, jump along v1 {np.round(res['d_v'], 1).tolist()} cm "
                        f"(random {np.round(res['d_rand'], 1).tolist()}), sides end {res['sides_sep_xy']:.1f} cm apart, "
                        f"{'ASK' if res['step'] else 'no question'} [{res['t_v1'] + res['t_plans']:.1f}s]")
                if res["step"]:
                    img = draw_question(rgb, arm.arm, state, res, path)
                    cv2.imwrite(os.path.join(run_dir, f"ask_{n:03d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    json.dump(res, open(os.path.join(run_dir, f"ask_{n:03d}.json"), "w"))
                    msg += f" -> ask_{n:03d}.png"
                    if args.stop_on_ridge:
                        log(msg)
                        log("ridge found: holding the arm, full test on the frozen state")
                        full, v1f = ridge_test(f, kin, obs, task, zp, n_iter=4, alphas=ALPHAS, with_random=True)
                        full, v1f = orient_sides(full, v1f)
                        V1CACHE[task] = (v1f, float(full["r"]))
                        full["sides_sep_xy"] = float(np.linalg.norm(np.array(full["path_plus"])[-1, :2] - np.array(full["path_minus"])[-1, :2]))
                        if args.ask_sep is not None:
                            full["step"] = bool(full["sides_sep_xy"] >= args.ask_sep)
                        img = draw_question(rgb, arm.arm, state, full, path)
                        cv2.imwrite(os.path.join(run_dir, f"ask_{n:03d}_full.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        json.dump(full, open(os.path.join(run_dir, f"ask_{n:03d}_full.json"), "w"))
                        asked = dict(n=n, step=full["step"], sigma1=full["sigma1"], side=full["side"],
                                     end_plus=np.array(full["path_plus"])[-1].tolist(), end_minus=np.array(full["path_minus"])[-1].tolist())
                        log(f"question: +v1 side ends at {asked['end_plus']} cm, -v1 side at {asked['end_minus']} cm "
                            f"(|P+ - P-| {np.round(full['side'], 0).tolist()} cm; step on the frozen state: {full['step']})")
                        if args.decode_at_stop:
                            from so101flow import dictionary, decode
                            t0 = time.time()
                            dec, _ = decode(f, kin, obs, task, zp, dictionary(f), n_iter=4)
                            asked["decoded"] = dict(cplus=dec["cplus"], cminus=dec["cminus"], toward_plus=dec["toward_plus"],
                                                    toward_minus=dec["toward_minus"], end_plus=dec["end_plus"], end_minus=dec["end_minus"])
                            json.dump(dec, open(os.path.join(run_dir, f"ask_{n:03d}_decode.json"), "w"))
                            log(f"decoded sides ({time.time() - t0:.0f}s): '{dec['cplus']}' (toward {dec['toward_plus']:+.2f}, ends y {dec['end_plus'][1]:.1f}) "
                                f"vs '{dec['cminus']}' (toward {dec['toward_minus']:+.2f}, ends y {dec['end_minus'][1]:.1f})")
                        records.append(rec)
                        break
            log(msg)
            jump = float(np.abs(deg[0, :5] - state[:5]).max())
            if n < args.dry_replans:                      # sanity gate before the first motion (one model load)
                log(f"replan {n}: dry (no motion); chunk starts {jump:.1f} deg from the measured pose")
                if not args.dry_run and jump > args.max_start_jump:
                    log(f"ABORT: the chunk starts {jump:.1f} deg from the pose (> {args.max_start_jump}): wrong joint convention?")
                    records.append(rec)
                    break
                records.append(rec)
                n += 1
                continue
            done, n_clamp, n_floor = arm.execute(deg, hz=args.hz, n_exec=args.exec, max_step_deg=args.max_step_deg, z_min=args.z_min)
            rec.update(executed=len(done), clamped=n_clamp, floor=n_floor)
            if arm.rb is None:
                arm.last_cmd = done[-1] if len(done) else state
            records.append(rec)
            n += 1
    except KeyboardInterrupt:
        log("interrupted - the arm holds its pose")
        raise
    finally:
        json.dump(dict(task=task, args=vars(args), hold=None if hold is None else dict(alpha=hold["alpha"], sign=hold["sign"], r=hold["r"], side=hold["side"],
                                                                                      sides_end_cm=hold.get("sides_end_cm")),
                       replans=records, asked=asked, t_offset=t_offset, t_loop_start=t_start),
                  open(os.path.join(run_dir, "run.json"), "w"), indent=1)
    last = records[-1] if records else {}
    log(f"episode done: {len(records)} replans, final state {np.round(last.get('state', []), 1).tolist()}, "
        f"tool {last.get('tool_z_cm')} cm -> {run_dir}")
    return dict(replans=len(records), final_state=last.get("state"), tool_z_cm=last.get("tool_z_cm"), asked=asked,
                min_gripper=min((r["state"][5] for r in records), default=None))


if __name__ == "__main__":
    main()
