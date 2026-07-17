"""Convert keyboard-collected LIBERO HDF5 demos to libero_demand_raw_v1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

# Conversion replays demos and renders camera images. Use the desktop GLX
# backend instead of inheriting EGL/headless settings that can segfault on
# machines with an active display.
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("PYOPENGL_PLATFORM", None)
os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)
os.environ.setdefault("MUJOCO_GL", "glx")

import h5py
import imageio.v2 as imageio
import numpy as np
import robosuite.utils.transform_utils as T

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(SCRIPTS_ROOT))

import init_path  # noqa: F401
import libero.libero.envs.bddl_utils as BDDLUtils
import libero.libero.utils.utils as libero_utils
from libero.libero.envs import TASK_MAPPING


def _json_value(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


def _capture_semantic_state(env, obs: dict[str, Any]) -> dict[str, Any]:
    semantic_state: dict[str, Any] = {
        "eef_position": _json_value(obs["robot0_eef_pos"]),
        "eef_quaternion": _json_value(obs["robot0_eef_quat"]),
        "gripper_qpos": _json_value(obs["robot0_gripper_qpos"]),
        "objects": {},
        "sites": {},
        "contacts": [],
        "gripper_contact_objects": [],
        "grasped_objects": [],
        "goal_predicates": [],
        "task_success": bool(env._check_success()),
    }

    object_names = sorted(getattr(env, "obj_body_id", {}).keys())
    for object_name in object_names:
        body_id = env.obj_body_id[object_name]
        object_state = {
            "position": _json_value(env.sim.data.body_xpos[body_id]),
            "quaternion": _json_value(env.sim.data.body_xquat[body_id]),
            "joint_qpos": [],
        }
        try:
            sim_object = env.get_object(object_name)
            for joint_name in getattr(sim_object, "joints", []):
                qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)
                if isinstance(qpos_addr, tuple):
                    joint_qpos = env.sim.data.qpos[qpos_addr[0] : qpos_addr[1]]
                else:
                    joint_qpos = env.sim.data.qpos[qpos_addr]
                object_state["joint_qpos"].append(_json_value(joint_qpos))
        except (KeyError, TypeError, ValueError):
            pass
        semantic_state["objects"][object_name] = object_state

    for site_name, site_object in sorted(env.object_sites_dict.items()):
        semantic_state["sites"][site_name] = {
            "position": _json_value(env.sim.data.get_site_xpos(site_name)),
            "parent_name": getattr(site_object, "parent_name", None),
        }

    try:
        gripper_geoms = set(env.robots[0].gripper.contact_geoms)
    except (AttributeError, IndexError, TypeError):
        gripper_geoms = set()

    contact_objects = set()
    for contact_index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[contact_index]
        geom1 = env.sim.model.geom_id2name(contact.geom1)
        geom2 = env.sim.model.geom_id2name(contact.geom2)
        semantic_state["contacts"].append([geom1, geom2])
        if geom1 not in gripper_geoms and geom2 not in gripper_geoms:
            continue
        other_geom = geom2 if geom1 in gripper_geoms else geom1
        for object_name in object_names:
            try:
                contact_geoms = set(env.get_object(object_name).contact_geoms)
            except (AttributeError, KeyError, TypeError):
                continue
            if other_geom in contact_geoms:
                contact_objects.add(object_name)

    semantic_state["gripper_contact_objects"] = sorted(contact_objects)
    for object_name in object_names:
        try:
            sim_object = env.get_object(object_name)
            if env._check_grasp(env.robots[0].gripper, sim_object.contact_geoms):
                semantic_state["grasped_objects"].append(object_name)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue

    for predicate in env.parsed_problem.get("goal_state", []):
        try:
            value = bool(env._eval_predicate(predicate))
        except (AssertionError, AttributeError, KeyError, TypeError, ValueError):
            value = False
        semantic_state["goal_predicates"].append(
            {
                "predicate": [str(part) for part in predicate],
                "value": value,
            }
        )
    return semantic_state


def _motion_metrics(actions: np.ndarray) -> dict[str, float]:
    motion = actions[:, :6]
    norms = np.linalg.norm(motion, axis=1)
    if len(motion) > 1:
        deltas = np.linalg.norm(np.diff(motion, axis=0), axis=1)
        smoothness = float(np.mean(deltas))
        max_delta = float(np.max(deltas))
    else:
        smoothness = 0.0
        max_delta = 0.0
    return {
        "mean_action_norm": float(np.mean(np.linalg.norm(actions, axis=1))),
        "action_energy": float(np.sum(np.square(actions))),
        "smoothness_score": smoothness,
        "max_action_delta": max_delta,
        "mean_motion_norm": float(np.mean(norms)),
        "motion_energy_per_step": float(np.mean(np.square(norms))),
        "motion_smoothness_score": smoothness,
        "max_motion_delta": max_delta,
    }


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> bool:
    try:
        with imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
        return True
    except Exception as exc:
        print(f"[warning] could not write {path}: {exc}")
        path.unlink(missing_ok=True)
        return False


def _load_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _next_episode_index(
    episodes_root: Path,
    existing_records: list[dict[str, Any]],
) -> int:
    indices = [
        int(record["episode_index"])
        for record in existing_records
        if "episode_index" in record
    ]
    for path in episodes_root.glob("episode_*"):
        if path.is_dir():
            try:
                indices.append(int(path.name.rsplit("_", 1)[-1]))
            except ValueError:
                continue
    return max(indices, default=-1) + 1


def _load_append_manifest(
    path: Path,
    *,
    fps: int,
    frame_stride: int,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "format": "libero_demand_raw_v1",
            "source_format": "libero_keyboard_hdf5",
            "demands": [],
            "source_demo_files": [],
            "append_runs": [],
            "save_semantic_states": True,
            "video_fps": fps,
            "frame_stride": frame_stride,
        }
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format") != "libero_demand_raw_v1":
        raise ValueError(f"Incompatible manifest format in {path}")
    if manifest.get("source_format") != "libero_keyboard_hdf5":
        raise ValueError(f"Incompatible source format in {path}")
    if int(manifest.get("video_fps", fps)) != fps:
        raise ValueError("Cannot append with a different --fps value")
    if int(manifest.get("frame_stride", frame_stride)) != frame_stride:
        raise ValueError("Cannot append with a different --frame-stride value")
    return manifest


def _canonical_path(path: Path) -> str:
    return os.fspath(path.expanduser().resolve())


def _discover_demo_files(path: Path) -> list[Path]:
    path = path.expanduser()
    if path.is_file():
        if path.name != "demo.hdf5":
            raise ValueError(f"Expected a demo.hdf5 file, got: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"No such demo file or directory: {path}")

    demo_files = sorted(path.rglob("demo.hdf5"))
    if not demo_files:
        raise FileNotFoundError(f"No demo.hdf5 files found under: {path}")
    return demo_files


def _converted_source_paths(manifest: dict[str, Any]) -> set[str]:
    source_files = set()
    legacy_source = manifest.get("source_demo_file")
    if legacy_source:
        source_files.add(_canonical_path(Path(legacy_source)))
    for source_file in manifest.get("source_demo_files", []):
        source_files.add(_canonical_path(Path(source_file)))
    for run in manifest.get("append_runs", []):
        source_demo_file = run.get("source_demo_file")
        if source_demo_file:
            source_files.add(_canonical_path(Path(source_demo_file)))
    return source_files


def _make_env(
    bddl_file: str,
    env_info: str,
    *,
    width: int,
    height: int,
):
    problem_info = BDDLUtils.get_problem_info(bddl_file)
    env_kwargs = json.loads(env_info)
    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_file,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=["robot0_eye_in_hand", "agentview"],
        camera_heights=height,
        camera_widths=width,
        camera_depths=False,
        camera_segmentations=None,
        reward_shaping=True,
        control_freq=20,
    )
    return TASK_MAPPING[problem_info["problem_name"]](**env_kwargs)


def _convert_demo(
    env,
    demo,
    episode_dir: Path,
    *,
    fps: int,
    frame_stride: int,
    save_videos: bool,
) -> tuple[dict[str, Any], bool]:
    states = np.asarray(demo["states"][()])[::frame_stride]
    actions = np.asarray(demo["actions"][()], dtype=np.float32)[::frame_stride]
    if len(states) != len(actions):
        raise ValueError(
            f"{demo.name}: states/actions length mismatch: {len(states)} != {len(actions)}"
        )

    model_xml = libero_utils.postprocess_model_xml(demo.attrs["model_file"], {})
    env.reset()
    env.reset_from_xml_string(model_xml)
    env.sim.reset()

    image_dir = episode_dir / "image"
    wrist_dir = episode_dir / "wrist_image"
    image_dir.mkdir(parents=True)
    wrist_dir.mkdir(parents=True)

    policy_states = []
    semantic_frames = []
    agent_frames = []
    for frame_index, state in enumerate(states):
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        obs = env._get_observations(force_update=True)

        # Match the interactive MuJoCo viewer: robosuite images need a
        # vertical flip, but a horizontal flip would mirror left and right.
        agent_image = np.ascontiguousarray(
            obs["agentview_image"][::-1], dtype=np.uint8
        )
        wrist_image = np.ascontiguousarray(
            obs["robot0_eye_in_hand_image"][::-1], dtype=np.uint8
        )
        imageio.imwrite(image_dir / f"{frame_index:06d}.png", agent_image)
        imageio.imwrite(wrist_dir / f"{frame_index:06d}.png", wrist_image)
        agent_frames.append(agent_image)

        policy_states.append(
            np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    T.quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ).astype(np.float32)
        )
        semantic_frames.append(
            {
                "frame_index": frame_index,
                **_capture_semantic_state(env, obs),
            }
        )

    np.save(episode_dir / "states.npy", np.asarray(policy_states, dtype=np.float32))
    np.save(episode_dir / "actions.npy", np.clip(actions, -1.0, 1.0))
    with (episode_dir / "semantic_states.jsonl").open("w", encoding="utf-8") as f:
        for frame in semantic_frames:
            f.write(json.dumps(frame, ensure_ascii=False) + "\n")

    video_paths = {
        "agentview_video_path": None,
        "wrist_video_path": None,
        "combined_video_path": None,
    }
    if save_videos:
        agent_video = episode_dir / "agentview.mp4"
        if _write_video(agent_video, agent_frames, fps):
            video_paths["agentview_video_path"] = agent_video.name

    success = bool(
        demo.attrs.get("success", False)
        or any(frame["task_success"] for frame in semantic_frames)
    )
    return {
        "num_steps": len(actions),
        "num_frames": len(actions),
        "success": success,
        "total_reward": 1.0 if success else 0.0,
        "elapsed_time": len(actions) / float(fps),
        **_motion_metrics(actions),
        **video_paths,
    }, success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo-file",
        required=True,
        help=(
            "Path to one demo.hdf5 file, or a directory such as "
            "demonstration_data/task1 containing many demo.hdf5 files."
        ),
    )
    parser.add_argument("--output-dir", default="data/libero_demand_raw_keyboard")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--demand-label", default="unlabeled")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help="Sample every Nth 20 Hz control frame; 2 produces the 10 Hz legacy format.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append episodes to an existing compatible output directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.overwrite:
        args.append = False
    elif not args.append:
        args.append = True
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")

    source_paths = _discover_demo_files(Path(args.demo_file))
    output_root = Path(args.output_dir)
    metadata_path = output_root / "episode_metadata.jsonl"
    manifest_path = output_root / "manifest.json"
    if output_root.exists() and not args.append:
        if not args.overwrite:
            raise FileExistsError(
                f"{output_root} already exists; pass --append or --overwrite"
            )
        shutil.rmtree(output_root)
    episodes_root = output_root / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=args.append)

    existing_records = _load_metadata(metadata_path) if args.append else []
    next_episode_index = _next_episode_index(episodes_root, existing_records)
    manifest = _load_append_manifest(
        manifest_path,
        fps=args.fps,
        frame_stride=args.frame_stride,
    )
    already_converted = _converted_source_paths(manifest) if args.append else set()

    records = []
    converted_sources = []
    for source_path in source_paths:
        source_id = _canonical_path(source_path)
        if args.append and source_id in already_converted:
            print(f"skipping already converted source file: {source_path}")
            continue

        with h5py.File(source_path, "r") as source:
            data = source["data"]
            problem_info = json.loads(data.attrs["problem_info"])
            bddl_file = os.fspath(data.attrs["bddl_file_name"])
            env = _make_env(
                bddl_file,
                data.attrs["env_info"],
                width=args.width,
                height=args.height,
            )
            source_records = []
            first_episode_index = next_episode_index + len(records)
            try:
                demo_names = sorted(
                    data.keys(), key=lambda name: int(name.split("_")[-1])
                )
                for source_demo_index, demo_name in enumerate(demo_names):
                    episode_index = next_episode_index + len(records)
                    episode_dir = episodes_root / f"episode_{episode_index:06d}"
                    episode_dir.mkdir()
                    metrics, _ = _convert_demo(
                        env,
                        data[demo_name],
                        episode_dir,
                        fps=args.fps,
                        frame_stride=args.frame_stride,
                        save_videos=args.save_videos,
                    )
                    relative_dir = Path("episodes") / episode_dir.name
                    task_description = problem_info["language_instruction"]
                    record = {
                        "episode_index": episode_index,
                        "source_demo_file": source_id,
                        "source_demo": demo_name,
                        "task_id": args.task_id,
                        "task_description": task_description,
                        "task_prompt": task_description,
                        "demand_label": args.demand_label,
                        "init_state_index": episode_index,
                        "trial_index": 0,
                        "episode_dir": os.fspath(relative_dir),
                        "image_dir": os.fspath(relative_dir / "image"),
                        "wrist_image_dir": os.fspath(relative_dir / "wrist_image"),
                        "states_path": os.fspath(relative_dir / "states.npy"),
                        "actions_path": os.fspath(relative_dir / "actions.npy"),
                        "semantic_states_path": os.fspath(
                            relative_dir / "semantic_states.jsonl"
                        ),
                        "video_fps": args.fps if args.save_videos else None,
                        "frame_stride": args.frame_stride,
                        "source_control_frequency": 20,
                        **metrics,
                    }
                    for key in (
                        "agentview_video_path",
                        "wrist_video_path",
                        "combined_video_path",
                    ):
                        if record[key]:
                            record[key] = os.fspath(relative_dir / record[key])
                    with (episode_dir / "meta.json").open(
                        "w", encoding="utf-8"
                    ) as f:
                        json.dump(record, f, ensure_ascii=False, indent=2)
                    source_records.append(record)
                    records.append(record)
                    print(
                        f"converted {source_path}::{demo_name}: "
                        f"frames={metrics['num_steps']} "
                        f"success={metrics['success']}"
                    )
            finally:
                env.close()

        converted_sources.append(
            {
                "source_path": source_id,
                "first_episode_index": first_episode_index,
                "num_episodes": len(source_records),
            }
        )

    metadata_mode = "a" if args.append else "w"
    with metadata_path.open(metadata_mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    source_files = manifest.setdefault("source_demo_files", [])
    legacy_source = manifest.pop("source_demo_file", None)
    if legacy_source and legacy_source not in source_files:
        source_files.append(legacy_source)
    for converted_source in converted_sources:
        if converted_source["source_path"] not in source_files:
            source_files.append(converted_source["source_path"])
    demands = manifest.setdefault("demands", [])
    if args.demand_label not in demands:
        demands.append(args.demand_label)
    append_runs = manifest.setdefault("append_runs", [])
    for converted_source in converted_sources:
        append_runs.append(
            {
                "source_demo_file": converted_source["source_path"],
                "first_episode_index": converted_source["first_episode_index"],
                "num_episodes": converted_source["num_episodes"],
                "save_videos": args.save_videos,
            }
        )
    manifest["save_videos"] = bool(
        manifest.get("save_videos", False) or args.save_videos
    )
    manifest["num_episodes"] = len(existing_records) + len(records)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(
        f"saved {len(records)} episode(s) starting at "
        f"episode_{next_episode_index:06d} to: {output_root}"
    )


if __name__ == "__main__":
    main()
