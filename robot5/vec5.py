"""
Batched closed-loop rollouts: one LIBERO scene per worker process (spawned, so each has its own
EGL context), the policy evaluated for all active scenes in one GPU batch.

    v = Vec(n_workers)
    v.load(i, task=..., init=..., layout=..., suite_name=...)   # (re)build worker i's scene
    obs = v.reset(idx)                                           # list of lerobot observations
    v.save(idx, key); obs = v.restore(idx, key)                  # exact branching (env5.Scene.get_state)
    res = v.episodes(f, idx, obs, texts, rngs, first=..., stop="lift")

Memory: each worker holds a full LIBERO env, 2-3.5 GB RSS; 8 workers fit next to the policy on
a 62 GB machine, 16 did not (OOM).

Each worker tracks the episode step t and the first tracked object lifted (the mode); both are
part of a saved state.
"""
import multiprocessing as mp
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_EXEC = 10


def _worker(conn):
    sys.path.insert(0, HERE)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import warnings
    warnings.filterwarnings("ignore")
    from env5 import Scene, TERMINAL
    s, saved, ep = None, {}, None
    while True:
        cmd, a = conn.recv()
        try:
            out = None
            if cmd == "load":
                if s is not None:
                    s.close()
                s, saved = Scene(**a), {}
                out = dict(task=s.task, max_steps=s.max_steps, track=s.track)
            elif cmd == "reset":
                obs = s.reset()
                ep = dict(t=0, first=-1, lift_t=-1, success=False)
                if s.suite_name == "libero_goal":            # predicate bookkeeping (env5.GOALS + EVENTS)
                    n = len(s.preds())
                    ep.update(goal=-1, goal_t=-1, ever_t=[-1] * n, final=[False] * n)
                out = s.obs_to_lerobot(obs, "")
            elif cmd == "save":
                saved[a] = (s.get_state(), dict(ep))
            elif cmd == "restore":
                x, e = saved[a]
                ep = dict(e)
                out = s.obs_to_lerobot(s.set_state(x), "")
            elif cmd == "steps":
                actions, stop, max_t = a
                for act in actions:
                    obs, succ, term = s.step(act)
                    ep["t"] += 1
                    if ep["first"] < 0:
                        ep["first"] = s.lifted()
                        if ep["first"] >= 0:
                            ep["lift_t"] = ep["t"]
                    ep["success"] = succ
                    if "goal" in ep:
                        g = s.preds()
                        ep["final"] = g.tolist()
                        ep["ever_t"] = [et if et >= 0 or not gi else ep["t"] for et, gi in zip(ep["ever_t"], g)]
                        if ep["goal"] < 0 and (g & TERMINAL).any():     # first goal / placement event
                            ep["goal"], ep["goal_t"] = int(np.argmax(g & TERMINAL)), ep["t"]
                    stopped = ((stop == "lift" and ep["first"] >= 0) or (stop == "goal" and ep.get("goal", -1) >= 0)
                               or (stop in ("lift", "success") and succ))   # as before for lift / success
                    if stopped or term or ep["t"] >= max_t:
                        break
                done = stopped or term or ep["t"] >= max_t
                out = dict(obs=s.obs_to_lerobot(obs, ""), done=done, eef=s.eef(), objects=s.objects(), **ep)
            elif cmd == "close":
                if s is not None:
                    s.close()
                conn.send(None)
                return
            conn.send(out)
        except Exception as e:  # report to the parent instead of hanging it
            import traceback
            conn.send(RuntimeError(f"worker: {e}\n{traceback.format_exc()}"))


class Vec:
    def __init__(self, n):
        self.ctx = mp.get_context("spawn")
        self.conns, self.procs = [None] * n, [None] * n
        for i in range(n):
            self._spawn(i)
        self.n = n
        self.info = [None] * n

    def _spawn(self, i):
        a, b = self.ctx.Pipe()
        p = self.ctx.Process(target=_worker, args=(b,), daemon=True)
        p.start()
        self.conns[i], self.procs[i] = a, p

    def _respawn(self, idx):
        """Replace workers with fresh processes: LIBERO leaks ~0.5-1 GB per scene build / hard
        reset, so a long-lived worker grows until the machine runs out of memory."""
        for i in idx:
            self.conns[i].send(("close", None))
        for i in idx:
            try:
                self.conns[i].recv()
            except EOFError:
                pass
            self.procs[i].join(timeout=30)
            if self.procs[i].is_alive():
                self.procs[i].terminate()
            self._spawn(i)

    def _call(self, idx, cmd, args):
        for i, a in zip(idx, args):
            self.conns[i].send((cmd, a))
        out = [self.conns[i].recv() for i in idx]
        for o in out:
            if isinstance(o, Exception):
                raise o
        return out

    def load(self, idx, specs):
        """Build each worker's scene in a fresh process (see _respawn)."""
        self._respawn([i for i in idx if self.info[i] is not None])
        for i, info in zip(idx, self._call(idx, "load", specs)):
            self.info[i] = info

    def reset(self, idx):
        return self._call(idx, "reset", [None] * len(idx))

    def save(self, idx, key):
        self._call(idx, "save", [key] * len(idx))

    def restore(self, idx, key):
        return self._call(idx, "restore", [key] * len(idx))

    def episodes(self, f, idx, obs, texts, rngs, first=None, stop="lift", max_t=None, n_exec=N_EXEC):
        """Closed loop for workers idx from observations obs, replanning every n_exec steps with
        noise from rngs[k] (one standard normal draw of 1600 per chunk, as rollout5). first[k]:
        optional env-action chunk [H, 7] executed (its first n_exec steps) instead of the first
        plan. Runs until done (stop: 'lift' = a tracked object lifted, 'success' = the loaded task's
        goal, 'goal' = any LIBERO-goal goal or placement event
        (env5.TERMINAL), 'none' = max_t). Returns the final per-worker
        dicts (t, first, lift_t, success, eef, objects; on LIBERO-goal also goal, goal_t = first
        terminal predicate reached (index into GOALS + EVENTS), ever_t = first step each of env5.GOALS + EVENTS held, final = their truth at the end)."""
        from pi05flow import H, D_ACT
        idx = list(idx)
        res = {i: None for i in idx}
        cur = dict(zip(idx, obs))
        pending = {i: (first[k] if first is not None and first[k] is not None else None) for k, i in enumerate(idx)}
        rng = dict(zip(idx, rngs)); text = dict(zip(idx, texts))
        active = list(idx)
        while active:
            plan = [i for i in active if pending[i] is None]
            acts = {i: pending[i] for i in active if pending[i] is not None}
            if plan:
                c = f.prefix([cur[i] for i in plan], [text[i] for i in plan])
                z = np.concatenate([rng[i].standard_normal((1, H * D_ACT)) for i in plan])
                a = f.unnorm(f.flow_np(z, c))
                acts.update({i: a[k] for k, i in enumerate(plan)})
            for i in active:
                pending[i] = None
            mt = {i: (max_t or self.info[i]["max_steps"]) for i in active}
            outs = self._call(active, "steps", [(acts[i][:n_exec], stop, mt[i]) for i in active])
            nxt = []
            for i, o in zip(active, outs):
                cur[i] = o.pop("obs")
                res[i] = o
                if not o["done"]:
                    nxt.append(i)
            active = nxt
        return [res[i] for i in idx]

    def close(self):
        try:
            self._call(list(range(self.n)), "close", [None] * self.n)
        except Exception:
            pass
        for p in self.procs:
            p.join(timeout=10)
