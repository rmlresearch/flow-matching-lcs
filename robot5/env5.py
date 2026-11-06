"""
LIBERO (MuJoCo / robosuite, Franka Panda, OSC delta-pose control at 20 Hz) for the FTLE / LCS
analysis of pi0.5.

Observation -> policy batch goes through lerobot's own pipeline, exactly as `lerobot-eval`:
LiberoEnv (obs_type pixels_agent_pos, 256x256 cameras like the HuggingFaceVLA/libero training
data) -> preprocess_observation -> LiberoProcessorStep (180 deg image flip; state = eef pos,
eef axis-angle, gripper qpos = 8-D) -> the checkpoint's policy preprocessor (state
normalisation, discretised state + prompt tokenised into the prefix).

Modes. LIBERO-spatial scenes all contain two identical black bowls (akita_black_bowl_1 is the
one the instruction refers to, _2 the distractor) and a plate; the instruction picks the bowl
by spatial relation. The mode of a rollout is the bowl that gets lifted.

LIBERO-10 tasks whose instruction leaves the order of two pick-and-place subtasks open (TRACK10):
the mode is the object lifted first.

Reproducibility: reset is seeded by `layout` (fixture placement), and get_state/set_state
also carry robosuite's hidden state (observable sampling timers, the gripper's ramped command),
so a branched rollout replays bit-for-bit.
"""
import os
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

SUITE = "libero_spatial"
CAMS = "agentview_image,robot0_eye_in_hand_image"
IMG = 256
BOWLS = ("akita_black_bowl_1", "akita_black_bowl_2")
LIFT = 0.05                      # bowl counts as lifted when 5 cm above its initial height
AMBIG = "pick up the black bowl and place it on the plate"
# LIBERO-10: the two objects moved by tasks with two pick-and-place subtasks (instruction order)
TRACK10 = {0: ("alphabet_soup_1", "tomato_sauce_1"), 1: ("cream_cheese_1", "butter_1"),
           4: ("porcelain_mug_1", "white_yellow_mug_1"), 6: ("porcelain_mug_1", "chocolate_pudding_1"),
           7: ("alphabet_soup_1", "cream_cheese_1"), 8: ("moka_pot_1", "moka_pot_2")}


# LIBERO-goal: one scene (identical BDDL apart from goal and wording) for all 10 tasks.
GOAL_OBJECTS = ("akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1")
GOALS = [("open", "wooden_cabinet_1_middle_region"),                 # 0 open the middle drawer of the cabinet
         ("on", "akita_black_bowl_1", "flat_stove_1_cook_region"),    # 1 put the bowl on the stove
         ("on", "wine_bottle_1", "wooden_cabinet_1_top_side"),        # 2 put the wine bottle on top of the cabinet
         ("in", "akita_black_bowl_1", "wooden_cabinet_1_top_region"), # 3 open the top drawer and put the bowl inside
         ("on", "akita_black_bowl_1", "wooden_cabinet_1_top_side"),   # 4 put the bowl on top of the cabinet
         ("on", "plate_1", "main_table_stove_front_region"),          # 5 push the plate to the front of the stove
         ("on", "cream_cheese_1", "akita_black_bowl_1"),              # 6 put the cream cheese in the bowl
         ("turnon", "flat_stove_1"),                                  # 7 turn on the stove
         ("on", "akita_black_bowl_1", "plate_1"),                     # 8 put the bowl on the plate
         ("on", "wine_bottle_1", "wine_rack_1_top_region")]           # 9 put the wine bottle on the rack
GOAL_NAMES = ["middle drawer", "bowl->stove", "bottle->cabinet top", "bowl->top drawer", "bowl->cabinet top",
              "plate->stove front", "cheese->bowl", "stove on", "bowl->plate", "bottle->rack"]
# extra events that label off-task behaviour (never goals of a training task)
EVENTS = [("open", "wooden_cabinet_1_top_region"), ("open", "wooden_cabinet_1_bottom_region"),
          ("on", "wine_bottle_1", "flat_stove_1_cook_region"), ("on", "cream_cheese_1", "plate_1"),
          ("on", "cream_cheese_1", "flat_stove_1_cook_region"), ("on", "wine_bottle_1", "plate_1"),
          ("in", "akita_black_bowl_1", "wooden_cabinet_1_middle_region"),
          ("in", "cream_cheese_1", "wooden_cabinet_1_top_region"), ("in", "cream_cheese_1", "wooden_cabinet_1_middle_region")]
TERMINAL = np.array([True] * len(GOALS) + [False, False] + [True] * (len(EVENTS) - 2))   # drawer opening is a step
EVENT_NAMES = ["top drawer open", "bottom drawer open", "bottle->stove", "cheese->plate", "cheese->stove",
               "bottle->plate", "bowl->middle drawer", "cheese->top drawer", "cheese->middle drawer"]


def tracked_objects(suite_name, task_id):
    if suite_name == "libero_goal":
        return GOAL_OBJECTS
    return BOWLS if suite_name == "libero_spatial" else TRACK10[task_id]


def suite(name=SUITE):
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()[name]()


class Scene:
    """One LIBERO task, reset to one of its 50 fixed initial states, with a seeded fixture layout.

    LIBERO's init states only set joint positions. Fixtures without joints (stove, cabinet) are
    re-placed at random within ~1 cm on every hard reset, which changes ~11% of the camera
    pixels and pi0.5's chunk by O(1) at the same physics state. `layout` seeds that placement,
    so (task, init, layout) is one reproducible scene."""

    def __init__(self, task_id, init_id=0, layout=0, suite_name=SUITE, track=None):
        from lerobot.envs.libero import LiberoEnv
        from lerobot.processor.env_processor import LiberoProcessorStep
        from lerobot.processor import PolicyProcessorPipeline
        self.suite = suite(suite_name)
        self.task_id, self.init_id, self.layout = task_id, init_id, layout
        self.suite_name = suite_name
        self.track = tuple(track or tracked_objects(suite_name, task_id))
        self.env = LiberoEnv(self.suite, task_id, suite_name, camera_name=CAMS, obs_type="pixels_agent_pos",
                             observation_width=IMG, observation_height=IMG, episode_index=init_id)
        self.task = self.env.task_description
        self.max_steps = self.env._max_episode_steps
        self.env_pre = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

    @property
    def sim(self):
        return self.env._env.env.sim

    def body_pos(self, name):
        e = self.env._env.env
        return np.array(self.sim.data.body_xpos[e.obj_body_id[name]])

    def objects(self):
        """Positions [n, 3] of the tracked objects (the two bowls on LIBERO-spatial)."""
        return np.stack([self.body_pos(b) for b in self.track])

    bowls = objects

    def preds(self, preds=None):
        """Truth of BDDL predicates in the current state (default: LIBERO-goal GOALS + EVENTS)."""
        e = self.env._env.env
        return np.array([bool(e._eval_predicate(list(p))) for p in (preds or GOALS + EVENTS)])

    def eef(self):
        r = self.env._env.env.robots[0]
        return np.array(self.sim.data.site_xpos[r.eef_site_id])

    def reset(self):
        self.env.init_state_id = self.init_id       # always the same initial state
        obs, _ = self.env.reset(seed=self.layout)
        self.bowl0 = self.objects()
        return obs

    def lifted(self):
        """Index of the tracked object lifted above LIFT (the highest if several), else -1."""
        dz = self.objects()[:, 2] - self.bowl0[:, 2]
        return int(np.argmax(dz)) if dz.max() > LIFT else -1

    _OBS_FIELDS = ("_time_since_last_sample", "_sampled", "_current_observed_value", "_current_delay")

    def get_state(self):
        """Physics state + step counter + every observable's sampling timer and last value.
        robosuite samples cameras/proprio on a float timer at the start of each control period,
        so the observation is not a function of the physics state alone; without the timers a
        replay from a restored state renders different images."""
        e = self.env._env.env
        obs = {k: tuple(np.copy(getattr(o, f)) if f == "_current_observed_value" else getattr(o, f)
                        for f in self._OBS_FIELDS) for k, o in e._observables.items()}
        grip = [np.copy(r.gripper.current_action) for r in e.robots]   # Panda gripper's ramped command
        return self.sim.get_state().flatten().copy(), e.timestep, obs, grip

    def set_state(self, x):
        """Restore a state saved with get_state. Returns the observation the policy saw there."""
        e = self.env._env.env
        self.sim.set_state_from_flattened(x[0])
        self.sim.forward()
        e.timestep = x[1]
        for r, g in zip(e.robots, x[3]):
            r.gripper.current_action = np.copy(g)
        for k, vals in x[2].items():
            for f, v in zip(self._OBS_FIELDS, vals):
                setattr(e._observables[k], f, np.copy(v) if f == "_current_observed_value" else v)
        return self.env._format_raw_obs(e._get_observations())

    def obs_to_lerobot(self, obs, text):
        """Raw env observation -> lerobot observation dict (before the policy preprocessor)."""
        from lerobot.envs.utils import preprocess_observation

        def b(x):                                      # add the vector-env batch axis
            return {k: b(v) for k, v in x.items()} if isinstance(x, dict) else np.asarray(x)[None]
        o = preprocess_observation(b(obs))
        o["task"] = [text]
        return self.env_pre(o)

    def step(self, action):
        obs, _, term, _, info = self.env.step(np.asarray(action, np.float64))
        return obs, bool(info["is_success"]), bool(term)

    def close(self):
        self.env.close()
