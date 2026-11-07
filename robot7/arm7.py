"""
The real SO-101 for robot7: simlab's RealBackend (Feetech STS3215 bus on /dev/ttyACM*, the USB
wrist camera through V4L2) behind a small interface, plus the execution safety of simlab's VLA
client (joint limits, per-step delta clamp, tool floor through FK, gripper range).

Needs read/write access to the bus adapter: `sudo usermod -aG dialout $USER` (log in again) or, per
boot, `sudo setfacl -m u:$USER:rw /dev/ttyACM0`. Without it, `bus=False` gives the camera alone.

    python arm7.py status          # bus scan + joint readings + a camera frame, nothing moves
    python arm7.py release         # torque off (after a run the arm holds its last pose with torque on)
    python arm7.py goto p l e f r g [s]   # move to a joint state (our deg, gripper 0-100) over s seconds
"""
import os, sys, time, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIMLAB = os.environ.get("SIMLAB", os.path.expanduser("~/Documents/GitHub/software/simlab"))
sys.path.insert(0, os.path.join(SIMLAB, "workspace", "skills"))


class Arm:
    def __init__(self, dry_run=True, bus=True, camera=True, run_dir=None, quiet=False):
        from so101.real_backend import RealBackend, Camera, load_config
        from so101.robot_model import ArmModel
        self.arm = ArmModel()
        self.cfg = load_config()
        self.dry_run = dry_run
        self.rb, self.cam = None, None
        if bus:
            from pathlib import Path
            self.rb = RealBackend(self.cfg, self.arm, dry_run=dry_run, run_dir=Path(run_dir) if run_dir else None, camera=camera)
            self.cam = self.rb.cam
            if not quiet:
                print(f"[arm] bus {self.cfg['port']}: servos {sorted(self.rb.ids.values())} answer; "
                      f"state {np.round(self.state_deg(), 1).tolist()} (ours, deg; gripper 0-100)", flush=True)
        elif camera:
            self.cam = Camera(self.cfg["camera_name"], self.cfg.get("camera_index"), tuple(self.cfg["image_size"]),
                              self.cfg.get("camera_rotation", 0))
        self.table_z = float(self.rb.table_z) if self.rb is not None else \
            (float(self.cfg["table_z"]) if self.cfg.get("table_z") is not None else float(self.arm.plate_z)) + float(self.cfg.get("z_offset", 0.0))
        self.lo, self.hi = np.degrees(self.arm.lo) - 8, np.degrees(self.arm.hi) + 8
        self.last_cmd = None

    # ------------------------------------------------------------------ observe
    def state_deg(self):
        """[pan, lift, elbow, flex, roll (deg, model zero), gripper 0-100]"""
        from so101.vla_common import state_from_backend
        return state_from_backend(self.rb.q(), self.rb.gripper_angle())

    def frame(self):
        if self.rb is not None:
            return self.rb.image()
        return self.cam.read()

    # ------------------------------------------------------------------ act
    def enable(self):
        if self.rb is not None and not self.dry_run:
            self.rb.enable(True)

    def goto(self, state_deg, seconds=3.0):
        """Move to a state (our deg, gripper 0-100) along a smooth joint-space ramp; torque must be on."""
        from so101.vla_common import state_to_backend
        q_t, g_t = state_to_backend(np.asarray(state_deg, np.float32))
        if self.rb is None or self.dry_run:
            print(f"[dry-run] goto {np.round(state_deg, 1).tolist()}")
            return
        if self.tool_z(state_deg) < 0.0:
            raise RuntimeError("goto: the target puts the tool below the desk")
        self.rb.move_q(q_t, seconds, settle=0.3)
        self.rb.gripper(g_t, settle=0.3)
        self.last_cmd = np.asarray(state_deg, np.float32).copy()

    def tool_z(self, state_deg):
        return float(self.arm.fk(np.radians(np.asarray(state_deg)[:5]))[0][2] - self.table_z)

    def execute(self, actions_deg, hz=30.0, n_exec=None, max_step_deg=2.0, z_min=0.008, ramp=10, smooth=0.3,
                on_step=None):
        """Play a chunk [n, 6] (our deg, gripper 0-100) with simlab's clamps: ramp from the last command,
        low-pass, joint limits, per-step delta, gripper range, tool floor. Returns the executed targets."""
        from so101.vla_common import state_to_backend
        period = 1.0 / hz
        state = self.state_deg() if self.rb is not None else np.asarray(actions_deg[0], np.float32)
        cur = state.copy() if self.last_cmd is None else self.last_cmd.copy()
        cur0, filt = cur.copy(), cur.copy()
        n = min(n_exec or len(actions_deg), len(actions_deg))
        done, n_clamp, n_floor = [], 0, 0
        for i in range(n):
            a = np.asarray(actions_deg[i], np.float32).copy()
            if ramp > 0 and i < ramp:
                w = (i + 1) / ramp
                a = (1 - w) * cur0 + w * a
            if 0 < smooth < 1:
                filt = smooth * a + (1 - smooth) * filt
                a = filt.copy()
            a[:5] = np.clip(a[:5], self.lo, self.hi)
            a[5] = float(np.clip(a[5], 0.0, 100.0))
            d = a[:5] - cur[:5]
            if (np.abs(d) > max_step_deg).any():
                n_clamp += 1
                a[:5] = cur[:5] + np.clip(d, -max_step_deg, max_step_deg)
            q_t, g_t = state_to_backend(a)
            z = self.tool_z(a)
            if z < z_min:
                n_floor += 1
                time.sleep(period)
                continue
            if self.rb is not None and not self.dry_run:
                self.rb.move_q(q_t, period, settle=0.0)
                self.rb.gripper(g_t, settle=0.0)
            else:
                time.sleep(max(0.0, period - 0.005))
            cur = a
            self.last_cmd = a.copy()
            done.append(a)
            if on_step:
                on_step(i, a)
        return np.array(done), n_clamp, n_floor

    def close(self, release=False):
        if self.rb is not None:
            self.rb.close(release_torque=release and not self.dry_run)
        elif self.cam is not None:
            self.cam.close()


def status(camera=True):
    import cv2
    out = os.path.join(HERE, "out")
    os.makedirs(out, exist_ok=True)
    try:
        arm = Arm(dry_run=True, bus=True, camera=camera)
        for j, sid in arm.rb.ids.items():
            st = arm.rb.bus.status(sid)
            print(f"  {j:14s} id {sid}: pos {st['position']} ticks, {st['voltage']} V, {st['temperature']} C, load {st['load']}, torque {st['torque']}")
        st = arm.state_deg()
        print(f"  state (ours, deg; gripper 0-100): {np.round(st, 1).tolist()} | old SO-100 degrees: "
              f"{np.round(__import__('so101.vla_common', fromlist=['to_so100_units']).to_so100_units(st), 1).tolist()}")
        print(f"  tool {arm.tool_z(st) * 100:.1f} cm above the desk; tcp {np.round(arm.arm.fk(np.radians(st[:5]))[0] * 100, 1).tolist()} cm")
    except Exception as e:
        print(f"[arm] bus unavailable ({type(e).__name__}: {e}); camera only", flush=True)
        arm = Arm(dry_run=True, bus=False, camera=True)
    if not camera or arm.cam is None:
        arm.close()
        return
    rgb = arm.frame()
    cv2.imwrite(os.path.join(out, "status_frame.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"  camera: {rgb.shape[1]}x{rgb.shape[0]}, mean brightness {rgb.mean():.0f} -> {os.path.join(out, 'status_frame.jpg')}")
    arm.close()


def release():
    """Torque off on every servo (the arm goes limp: lower it to the desk by hand first if it is raised)."""
    arm = Arm(dry_run=True, bus=True, camera=False)
    arm.rb.bus.torque(list(arm.rb.ids.values()), False)
    print("torque OFF; state", np.round(arm.state_deg(), 1).tolist())
    arm.close()


if __name__ == "__main__":
    if sys.argv[1] == "status":
        status(camera="--no-camera" not in sys.argv)
    elif sys.argv[1] == "release":
        release()
    elif sys.argv[1] == "goto":                        # arm7.py goto pan lift elbow flex roll gripper [seconds]
        vals = [float(x) for x in sys.argv[2:8]]
        secs = float(sys.argv[8]) if len(sys.argv) > 8 else 3.0
        arm = Arm(dry_run=False, bus=True, camera=False)
        arm.enable()
        print(f"moving to {vals} over {secs}s from {np.round(arm.state_deg(), 1).tolist()}")
        arm.goto(vals, secs)
        print("now at", np.round(arm.state_deg(), 1).tolist())
        arm.close()
