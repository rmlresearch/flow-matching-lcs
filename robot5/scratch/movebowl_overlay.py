"""Data for the operator view: 'move the bowl', state 2, seed 0, replans t 40 / 50 / 60. The agentview
image the policy sees, the camera projection, and the eef paths (executed chunk, unperturbed mean
plan, the two sides of the prompt-space ridge) as world points, plus the projected object positions
for checking the pixel convention."""
import sys
import numpy as np, torch
sys.path.insert(0, "/home/james/Projects/flow-matching-lcs/robot5")
import pi05flow
from pi05flow import H, D_ACT
from env5 import Scene
from goal5 import path_cm
from pridge5 import prompt_slots, plans, top_dir_E
import monitor5 as M
from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera

TEXT, SUITE, INIT, SEED = "move the bowl", "libero_goal", 2, 0
TS, ALPHA = (40, 50, 60), 0.025
A = H * D_ACT
OUT = "/home/james/Projects/flow-matching-lcs/robot5/out/movebowl_overlay.npz"


def main():
    f = pi05flow.load()
    zp = np.random.default_rng(777).standard_normal((M.N_Z, A))
    s = Scene(0, INIT, 0, suite_name=SUITE); obs = s.reset(); x0 = s.get_state(); obs = s.set_state(x0)
    rng = np.random.default_rng(1000 + SEED)
    W2P = get_camera_transform_matrix(s.sim, "agentview", 256, 256)
    data, t = {}, 0
    while t <= max(TS):
        o = s.obs_to_lerobot(obs, TEXT)
        z = rng.standard_normal((1, A))
        if t in TS:
            img = np.asarray(obs["pixels"]["image"])                            # what the policy sees (raw robosuite render)
            eef = s.eef()
            E, pad, att = f.embed([o], [TEXT]); slots, _ = prompt_slots(f, o, TEXT)
            v1, s1, r = top_dir_E(f, E, pad, att, slots, zp)
            Es = [E.clone(), E.clone(), E.clone()]; Es[1][0, slots] += ALPHA * r * v1; Es[2][0, slots] -= ALPHA * r * v1
            Pm = plans(f, torch.cat(Es), pad, att, zp).mean(1).view(3, 50, 3).double().cpu().numpy() / 100   # m, relative to eef
            p_exec = path_cm(f.unnorm(f.flow_np(z, f.prefix([o], [TEXT]))[0]))[:50] / 100
            paths = dict(exec=eef + p_exec, base=eef + Pm[0], plus=eef + Pm[1], minus=eef + Pm[2])
            objs = dict(eef=eef, bowl=s.body_pos("akita_black_bowl_1"), plate=s.body_pos("plate_1"), stove=s.body_pos("flat_stove_1"))
            data[f"img_{t}"] = img; data[f"sigma1_{t}"] = s1
            for k, v in paths.items():
                data[f"path_{k}_{t}"] = v; data[f"pix_{k}_{t}"] = project_points_from_world_to_camera(v, W2P, 256, 256)
            for k, v in objs.items():
                data[f"obj_{k}_{t}"] = v; data[f"objpix_{k}_{t}"] = project_points_from_world_to_camera(v[None], W2P, 256, 256)[0]
            M.log(f"t {t}: sigma1 {s1:.1f}; img {img.shape} {img.dtype}; eef px {data[f'objpix_eef_{t}']} bowl px {data[f'objpix_bowl_{t}']} "
                  f"plate px {data[f'objpix_plate_{t}']} stove px {data[f'objpix_stove_{t}']}; plus end px {data[f'pix_plus_{t}'][-1]} minus end px {data[f'pix_minus_{t}'][-1]}")
        a = f.unnorm(f.flow_np(z, f.prefix([o], [TEXT]))[0])
        for j in range(10):
            obs, _, _ = s.step(a[j]); t += 1
    np.savez(OUT, **data)
    s.close()


if __name__ == "__main__":
    main()
