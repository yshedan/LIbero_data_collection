"""Generate fixed LIBERO init states from custom BDDL files.

The saved states are flattened MuJoCo simulator states. They can be passed to
`env.set_init_state(...)` during evaluation to make custom-task rollouts
repeatable.
"""

from __future__ import annotations

import argparse
import os
import pathlib

# Keep this script independent of the caller's shell environment. Init-state
# generation does not need rendering, and EGL can segfault on this workstation.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
os.environ.setdefault("MUJOCO_GL", "glx")
if os.environ.get("PYOPENGL_PLATFORM") == "egl":
    os.environ["PYOPENGL_PLATFORM"] = "glx"

import numpy as np
import torch
from libero.libero.envs.env_wrapper import ControlEnv


def _make_env(bddl_file: pathlib.Path, resolution: int, seed: int) -> ControlEnv:
    # Init-state generation only needs MuJoCo reset state, not rendered images.
    # Keeping all renderers disabled avoids EGL / GPU offscreen segfaults.
    env = ControlEnv(
        bddl_file_name=str(bddl_file),
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env


def generate_init_states(
    bddl_file: pathlib.Path,
    num_states: int,
    seed: int,
    resolution: int,
) -> np.ndarray:
    env = _make_env(bddl_file, resolution, seed)
    states = []
    try:
        for idx in range(num_states):
            # Re-seed per state so each saved state is reproducible by index.
            env.seed(seed + idx)
            env.reset()
            states.append(np.asarray(env.get_sim_state(), dtype=np.float32))
    finally:
        env.close()
    return np.stack(states, axis=0)


def _default_output_path(output_dir: pathlib.Path, bddl_file: pathlib.Path) -> pathlib.Path:
    return output_dir / f"{bddl_file.stem}.pruned_init"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl", type=pathlib.Path, help="Path to one custom .bddl file.")
    parser.add_argument("--bddl-dir", type=pathlib.Path, help="Directory containing custom .bddl files.")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("assets/Atomic_libero/custom_tasks/init_states"),
    )
    parser.add_argument("--num-states", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--save-npy", action="store_true", help="Also save a .npy copy next to the LIBERO file.")
    args = parser.parse_args()

    if (args.bddl is None) == (args.bddl_dir is None):
        raise ValueError("Pass exactly one of --bddl or --bddl-dir.")

    if args.bddl is not None:
        bddl_files = [args.bddl]
    else:
        bddl_files = sorted(args.bddl_dir.glob("*.bddl"))

    if not bddl_files:
        raise FileNotFoundError("No .bddl files found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for bddl_file in bddl_files:
        bddl_file = bddl_file.resolve()
        output_path = _default_output_path(args.output_dir, bddl_file)
        states = generate_init_states(
            bddl_file=bddl_file,
            num_states=args.num_states,
            seed=args.seed,
            resolution=args.resolution,
        )
        torch.save(states, output_path)
        if args.save_npy:
            np.save(output_path.with_suffix(output_path.suffix + ".npy"), states)
        print(f"saved {states.shape} init states: {output_path}")


if __name__ == "__main__":
    main()
