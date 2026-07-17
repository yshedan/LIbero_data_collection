from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import shutil
import sys
import time
from typing import Any, Literal, Optional

import imageio.v2 as imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro


def _import_libero():
    try:
        from libero.libero import benchmark
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        return benchmark, get_libero_path, OffScreenRenderEnv
    except ModuleNotFoundError as original_error:
        script_path = pathlib.Path(__file__).resolve()
        repo_root = next(
            (
                parent
                for parent in script_path.parents
                if (parent / "third_party" / "LIBERO").exists()
            ),
            script_path.parents[2],
        )
        candidate_roots = (
            repo_root / "third_party" / "LIBERO",
            repo_root / "third_party" / "LIBERO" / "libero",
            repo_root / "third_party" / "libero",
            repo_root / "third_party" / "libero" / "libero",
        )
        for candidate_root in candidate_roots:
            if candidate_root.exists():
                candidate_root_str = str(candidate_root)
                if candidate_root_str not in sys.path:
                    sys.path.insert(0, candidate_root_str)
                try:
                    from libero.libero import benchmark
                    from libero.libero import get_libero_path
                    from libero.libero.envs import OffScreenRenderEnv

                    logging.info("Imported LIBERO from local path: %s", candidate_root)
                    return benchmark, get_libero_path, OffScreenRenderEnv
                except ModuleNotFoundError:
                    continue

        raise ModuleNotFoundError(
            "Could not import LIBERO. Install the LIBERO dependency or initialize the submodule and add it to "
            "PYTHONPATH, e.g. `export PYTHONPATH=$PYTHONPATH:$PWD/third_party/LIBERO`."
        ) from original_error


benchmark = None
get_libero_path = None
OffScreenRenderEnv = None


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    policy_mode: Literal["random", "server"] = "random"
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    random_action_scale: float = 0.05
    random_gripper_prob: float = 0.15

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_demand: int = 1
    max_init_states_per_task: Optional[int] = None
    demands: tuple[str, ...] = ("fast", "safe", "stable")
    seed: int = 7
    gpu_id: Optional[int] = None

    #################################################################################################################
    # Raw dataset parameters
    #################################################################################################################
    output_dir: str = "data/libero_demand_raw"
    metadata_filename: str = "episode_metadata.jsonl"
    manifest_filename: str = "manifest.json"
    save_semantic_states: bool = True
    save_videos: bool = True
    save_wrist_video: bool = False
    save_combined_video: bool = False
    video_fps: int = 10
    append: bool = False
    overwrite: bool = False


def _get_max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 270
    if task_suite_name == "libero_object":
        return 300
    if task_suite_name == "libero_goal":
        return 320
    if task_suite_name == "libero_10":
        return 700
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _get_libero_env(task, resolution: int, seed: int):
    if get_libero_path is None or OffScreenRenderEnv is None:
        raise RuntimeError("LIBERO environment is not initialized.")
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _format_task_with_demand(task_description: str, demand: str) -> str:
    demand_adverbs = {
        "fast": "quickly",
        "safe": "safely",
        "stable": "steadily",
    }
    if demand not in demand_adverbs:
        raise ValueError(f"Unsupported execution requirement: {demand}")
    task_description = task_description.strip().rstrip(".")
    return f"{task_description} {demand_adverbs[demand]}."


def _make_output_dir(
    output_dir: str,
    *,
    append: bool,
    overwrite: bool,
) -> tuple[pathlib.Path, pathlib.Path]:
    output_path = pathlib.Path(output_dir)
    if append and overwrite:
        raise ValueError("--append and --overwrite cannot be used together.")

    if output_path.exists():
        if overwrite:
            shutil.rmtree(output_path)
        elif not append:
            raise FileExistsError(
                f"Output path already exists: {output_path}. Pass --append to continue "
                "or --overwrite to replace it."
            )
    elif append:
        logging.info("Append target does not exist; creating a new dataset at %s.", output_path)

    episodes_root = output_path / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    return output_path, episodes_root


def _next_episode_index(
    episodes_root: pathlib.Path,
    metadata_path: pathlib.Path,
) -> int:
    existing_indices: set[int] = set()
    for episode_dir in episodes_root.glob("episode_*"):
        if not episode_dir.is_dir():
            continue
        try:
            existing_indices.add(int(episode_dir.name[len("episode_") :]))
        except ValueError:
            logging.warning("Ignoring unexpected episode directory: %s", episode_dir)

    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    existing_indices.add(int(json.loads(line)["episode_index"]))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"Invalid metadata at {metadata_path}:{line_number}"
                    ) from error
    return max(existing_indices, default=-1) + 1


def _validate_append_manifest(
    manifest_path: pathlib.Path,
    *,
    task_suite_name: str,
) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    existing_suite = manifest.get("task_suite_name")
    if existing_suite is not None and existing_suite != task_suite_name:
        raise ValueError(
            f"Cannot append task suite {task_suite_name!r} to existing suite {existing_suite!r}."
        )
    return manifest


def _to_json_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _capture_semantic_state(env, obs: dict[str, Any]) -> dict[str, Any]:
    """Capture simulator state needed for trajectory-based atomic-skill labels."""
    inner_env = getattr(env, "env", env)
    semantic_state: dict[str, Any] = {
        "eef_position": _to_json_value(obs["robot0_eef_pos"]),
        "eef_quaternion": _to_json_value(obs["robot0_eef_quat"]),
        "gripper_qpos": _to_json_value(obs["robot0_gripper_qpos"]),
        "objects": {},
        "sites": {},
        "contacts": [],
        "gripper_contact_objects": [],
        "grasped_objects": [],
        "goal_predicates": [],
        "task_success": bool(inner_env._check_success()),
    }

    object_names = sorted(getattr(inner_env, "obj_body_id", {}).keys())
    for object_name in object_names:
        body_id = inner_env.obj_body_id[object_name]
        object_state = {
            "position": _to_json_value(inner_env.sim.data.body_xpos[body_id]),
            "quaternion": _to_json_value(inner_env.sim.data.body_xquat[body_id]),
            "joint_qpos": [],
        }
        try:
            sim_object = inner_env.get_object(object_name)
            for joint_name in getattr(sim_object, "joints", []):
                qpos_addr = inner_env.sim.model.get_joint_qpos_addr(joint_name)
                if isinstance(qpos_addr, tuple):
                    joint_qpos = inner_env.sim.data.qpos[qpos_addr[0] : qpos_addr[1]]
                else:
                    joint_qpos = inner_env.sim.data.qpos[qpos_addr]
                object_state["joint_qpos"].append(_to_json_value(joint_qpos))
        except (KeyError, TypeError, ValueError):
            pass
        semantic_state["objects"][object_name] = object_state

    for site_name, site_object in sorted(getattr(inner_env, "object_sites_dict", {}).items()):
        semantic_state["sites"][site_name] = {
            "position": _to_json_value(inner_env.sim.data.get_site_xpos(site_name)),
            "parent_name": getattr(site_object, "parent_name", None),
        }

    gripper_geoms: set[str] = set()
    try:
        gripper_geoms = set(inner_env.robots[0].gripper.contact_geoms)
    except (AttributeError, IndexError, TypeError):
        pass

    contact_objects: set[str] = set()
    for contact_index in range(inner_env.sim.data.ncon):
        contact = inner_env.sim.data.contact[contact_index]
        geom1 = inner_env.sim.model.geom_id2name(contact.geom1)
        geom2 = inner_env.sim.model.geom_id2name(contact.geom2)
        semantic_state["contacts"].append([geom1, geom2])
        if geom1 in gripper_geoms or geom2 in gripper_geoms:
            other_geom = geom2 if geom1 in gripper_geoms else geom1
            for object_name in object_names:
                try:
                    contact_geoms = set(inner_env.get_object(object_name).contact_geoms)
                except (AttributeError, KeyError, TypeError):
                    continue
                if other_geom in contact_geoms:
                    contact_objects.add(object_name)

    semantic_state["gripper_contact_objects"] = sorted(contact_objects)
    for object_name in object_names:
        try:
            sim_object = inner_env.get_object(object_name)
            if inner_env._check_grasp(inner_env.robots[0].gripper, sim_object.contact_geoms):
                semantic_state["grasped_objects"].append(object_name)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue

    for predicate in inner_env.parsed_problem.get("goal_state", []):
        try:
            value = bool(inner_env._eval_predicate(predicate))
        except (AssertionError, AttributeError, KeyError, TypeError, ValueError):
            value = False
        semantic_state["goal_predicates"].append(
            {
                "predicate": [str(part) for part in predicate],
                "value": value,
            }
        )
    return semantic_state


def _collect_episode(
    client: _websocket_client_policy.WebsocketClientPolicy | None,
    env,
    init_state: np.ndarray,
    task_prompt: str,
    *,
    policy_mode: Literal["random", "server"],
    demand: str,
    resize_size: int,
    replan_steps: int,
    num_steps_wait: int,
    max_steps: int,
    random_action_scale: float,
    random_gripper_prob: float,
    save_semantic_states: bool,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bad_action_template = np.array(
        [[0.078563, 0.0360005, 0.065438, -0.00272493, -0.00059135, -0.01451939, -0.0001995]] * 50
    )

    env.reset()
    obs = env.set_init_state(init_state)
    if client is not None:
        client.reset()
    action_plan = collections.deque()
    frames: list[dict[str, Any]] = []
    t = 0
    total_reward = 0.0
    done = False
    start_time = time.monotonic()
    action_history: list[np.ndarray] = []

    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, reward, done, _ = env.step(LIBERO_DUMMY_ACTION)
            total_reward += float(reward)
            t += 1
            if done:
                break
            continue

        image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).astype(np.uint8)
        wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]).astype(np.uint8)
        state = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ).astype(np.float32)

        if not action_plan:
            if policy_mode == "server":
                if client is None:
                    raise ValueError("Expected a policy client in server mode.")
                policy_input = {
                    "observation/image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(image, resize_size, resize_size)
                    ),
                    "observation/wrist_image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_image, resize_size, resize_size)
                    ),
                    "observation/state": state,
                    "prompt": task_prompt,
                    "reset": 0,
                }
                action_chunk = client.infer(policy_input)["actions"]
                while np.allclose(action_chunk, bad_action_template, atol=1e-6):
                    action_chunk = client.infer(policy_input)["actions"]
                if len(action_chunk) < replan_steps:
                    raise ValueError(
                        f"Policy predicted only {len(action_chunk)} actions, expected at least {replan_steps}."
                    )
            else:
                action_chunk = _sample_random_action_chunk(
                    replan_steps,
                    demand=demand,
                    random_action_scale=random_action_scale,
                    random_gripper_prob=random_gripper_prob,
                    rng=rng,
                )
            action_plan.extend(action_chunk[:replan_steps])

        action = np.asarray(action_plan.popleft(), dtype=np.float32)
        action_history.append(action)
        frames.append(
            {
                "image": image,
                "wrist_image": wrist_image,
                "state": state,
                "actions": action,
                "task": task_prompt,
                "semantic_state": _capture_semantic_state(env, obs) if save_semantic_states else None,
            }
        )

        obs, reward, done, _ = env.step(action.tolist())
        total_reward += float(reward)
        t += 1
        if done:
            break

    elapsed_time = time.monotonic() - start_time
    if action_history:
        action_array = np.stack(action_history, axis=0)
        action_norms = np.linalg.norm(action_array, axis=1)
        if len(action_array) > 1:
            action_deltas = np.diff(action_array, axis=0)
            action_delta_norms = np.linalg.norm(action_deltas, axis=1)
            smoothness_score = float(np.mean(action_delta_norms))
            max_action_delta = float(np.max(action_delta_norms))
        else:
            smoothness_score = 0.0
            max_action_delta = 0.0
        mean_action_norm = float(np.mean(action_norms))
        action_energy = float(np.sum(np.square(action_norms)))
    else:
        smoothness_score = 0.0
        max_action_delta = 0.0
        mean_action_norm = 0.0
        action_energy = 0.0

    metrics = {
        "success": bool(done),
        "num_steps": len(frames),
        "elapsed_time": elapsed_time,
        "total_reward": total_reward,
        "mean_action_norm": mean_action_norm,
        "action_energy": action_energy,
        "smoothness_score": smoothness_score,
        "max_action_delta": max_action_delta,
    }
    return frames, metrics


def _sample_random_action_chunk(
    replan_steps: int,
    *,
    demand: str,
    random_action_scale: float,
    random_gripper_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    # Bias motion statistics a bit by demand so the later filter has some spread to work with.
    demand_motion_scale = {
        "fast": 1.5,
        "safe": 0.5,
        "stable": 0.7,
    }.get(demand, 1.0)

    base_scale = random_action_scale * demand_motion_scale
    if demand == "stable":
        noise = rng.normal(loc=0.0, scale=base_scale * 0.35, size=(replan_steps, 6)).astype(np.float32)
        motion = np.cumsum(noise, axis=0)
        motion = np.clip(motion, -base_scale, base_scale)
    else:
        motion = rng.normal(loc=0.0, scale=base_scale, size=(replan_steps, 6)).astype(np.float32)
        motion = np.clip(motion, -2.0 * base_scale, 2.0 * base_scale)

    gripper = np.full((replan_steps, 1), -1.0, dtype=np.float32)
    toggle_mask = rng.random(replan_steps) < random_gripper_prob
    if demand == "fast":
        gripper[toggle_mask] = 1.0
    elif demand == "safe":
        gripper[toggle_mask] = -1.0
    else:
        toggle_count = int(toggle_mask.sum())
        if toggle_count > 0:
            sampled_gripper = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=toggle_count).reshape(-1, 1)
            gripper[toggle_mask] = sampled_gripper

    return np.concatenate((motion, gripper), axis=1)


def _write_episode_video(
    output_path: pathlib.Path,
    frames: list[dict[str, Any]],
    frame_key: str,
    fps: int,
) -> bool:
    try:
        with imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        ) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame[frame_key], dtype=np.uint8))
        return True
    except Exception:
        logging.exception(
            "Could not encode video %s. Install FFmpeg or imageio-ffmpeg; trajectory data is still saved.",
            output_path,
        )
        output_path.unlink(missing_ok=True)
        return False


def _write_combined_video(
    output_path: pathlib.Path,
    frames: list[dict[str, Any]],
    fps: int,
) -> bool:
    try:
        with imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        ) as writer:
            for frame in frames:
                agentview = np.asarray(frame["image"], dtype=np.uint8)
                wrist = np.asarray(frame["wrist_image"], dtype=np.uint8)
                writer.append_data(np.concatenate((agentview, wrist), axis=1))
        return True
    except Exception:
        logging.exception(
            "Could not encode combined video %s. Install FFmpeg or imageio-ffmpeg; trajectory data is still saved.",
            output_path,
        )
        output_path.unlink(missing_ok=True)
        return False


def _save_raw_episode(
    episodes_root: pathlib.Path,
    episode_index: int,
    frames: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    save_videos: bool,
    save_wrist_video: bool,
    save_combined_video: bool,
    video_fps: int,
) -> pathlib.Path:
    episode_dir = episodes_root / f"episode_{episode_index:06d}"
    rgb_dir = episode_dir / "image"
    wrist_dir = episode_dir / "wrist_image"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    wrist_dir.mkdir(parents=True, exist_ok=True)

    states = np.stack([np.asarray(frame["state"], dtype=np.float32) for frame in frames], axis=0)
    actions = np.stack([np.asarray(frame["actions"], dtype=np.float32) for frame in frames], axis=0)

    for frame_idx, frame in enumerate(frames):
        imageio.imwrite(rgb_dir / f"{frame_idx:06d}.png", np.asarray(frame["image"], dtype=np.uint8))
        imageio.imwrite(wrist_dir / f"{frame_idx:06d}.png", np.asarray(frame["wrist_image"], dtype=np.uint8))

    np.save(episode_dir / "states.npy", states)
    np.save(episode_dir / "actions.npy", actions)

    video_paths: dict[str, str | None] = {
        "agentview_video_path": None,
        "wrist_video_path": None,
        "combined_video_path": None,
    }
    if save_videos:
        agentview_video = episode_dir / "agentview.mp4"
        if _write_episode_video(agentview_video, frames, "image", video_fps):
            video_paths["agentview_video_path"] = str(
                pathlib.Path("episodes") / episode_dir.name / agentview_video.name
            )
        if save_wrist_video:
            wrist_video = episode_dir / "wrist.mp4"
            if _write_episode_video(wrist_video, frames, "wrist_image", video_fps):
                video_paths["wrist_video_path"] = str(
                    pathlib.Path("episodes") / episode_dir.name / wrist_video.name
                )
        if save_combined_video:
            combined_video = episode_dir / "combined.mp4"
            if _write_combined_video(combined_video, frames, video_fps):
                video_paths["combined_video_path"] = str(
                    pathlib.Path("episodes") / episode_dir.name / combined_video.name
                )

    semantic_states = [frame["semantic_state"] for frame in frames]
    if any(semantic_state is not None for semantic_state in semantic_states):
        semantic_states_path = episode_dir / "semantic_states.jsonl"
        with semantic_states_path.open("w", encoding="utf-8") as f:
            for frame_idx, semantic_state in enumerate(semantic_states):
                f.write(
                    json.dumps(
                        {"frame_index": frame_idx, **(semantic_state or {})},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    else:
        semantic_states_path = None
    episode_meta = {
        **metadata,
        "num_frames": len(frames),
        "image_dir": str(pathlib.Path("episodes") / episode_dir.name / "image"),
        "wrist_image_dir": str(pathlib.Path("episodes") / episode_dir.name / "wrist_image"),
        "states_path": str(pathlib.Path("episodes") / episode_dir.name / "states.npy"),
        "actions_path": str(pathlib.Path("episodes") / episode_dir.name / "actions.npy"),
        "video_fps": video_fps if save_videos else None,
        **video_paths,
        "semantic_states_path": (
            str(pathlib.Path("episodes") / episode_dir.name / "semantic_states.jsonl")
            if semantic_states_path is not None
            else None
        ),
    }
    with (episode_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(episode_meta, f, ensure_ascii=False, indent=2)
    return episode_dir


def main(args: Args) -> None:
    global benchmark, get_libero_path, OffScreenRenderEnv

    if args.video_fps <= 0:
        raise ValueError("video_fps must be positive.")

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu_id)

    if benchmark is None or get_libero_path is None or OffScreenRenderEnv is None:
        benchmark, get_libero_path, OffScreenRenderEnv = _import_libero()

    np.random.seed(args.seed)
    output_path, episodes_root = _make_output_dir(
        args.output_dir,
        append=args.append,
        overwrite=args.overwrite,
    )
    metadata_path = output_path / args.metadata_filename
    manifest_path = output_path / args.manifest_filename

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    max_steps = _get_max_steps(args.task_suite_name)
    client = None
    if args.policy_mode == "server":
        client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    run_manifest = {
        "format": "libero_demand_raw_v1",
        "policy_mode": args.policy_mode,
        "task_suite_name": args.task_suite_name,
        "demands": list(args.demands),
        "num_trials_per_demand": args.num_trials_per_demand,
        "max_init_states_per_task": args.max_init_states_per_task,
        "gpu_id": args.gpu_id,
        "resize_size": args.resize_size,
        "replan_steps": args.replan_steps,
        "random_action_scale": args.random_action_scale,
        "random_gripper_prob": args.random_gripper_prob,
        "num_steps_wait": args.num_steps_wait,
        "save_semantic_states": args.save_semantic_states,
        "save_videos": args.save_videos,
        "save_wrist_video": args.save_wrist_video,
        "save_combined_video": args.save_combined_video,
        "video_fps": args.video_fps,
        "seed": args.seed,
    }
    existing_manifest = (
        _validate_append_manifest(
            manifest_path,
            task_suite_name=args.task_suite_name,
        )
        if args.append
        else None
    )
    if existing_manifest is not None:
        manifest = existing_manifest
        manifest.setdefault("append_runs", []).append(run_manifest)
    else:
        manifest = run_manifest
        manifest["append_runs"] = []
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    episode_index = _next_episode_index(episodes_root, metadata_path) if args.append else 0
    metadata_mode = "a" if args.append else "w"
    logging.info("Starting collection at episode index %s.", episode_index)
    if args.append and metadata_path.exists() and metadata_path.stat().st_size > 0:
        with metadata_path.open("rb+") as metadata_binary:
            metadata_binary.seek(-1, os.SEEK_END)
            if metadata_binary.read(1) != b"\n":
                metadata_binary.write(b"\n")
    with metadata_path.open(metadata_mode, encoding="utf-8") as metadata_file:
        for task_id in tqdm.tqdm(range(task_suite.n_tasks), desc="Tasks"):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            if args.max_init_states_per_task is not None:
                initial_states = initial_states[: args.max_init_states_per_task]

            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed + task_id)
            try:
                for init_idx, init_state in enumerate(
                    tqdm.tqdm(initial_states, desc=f"Task {task_id}", leave=False)
                ):
                    for demand in args.demands:
                        task_prompt = _format_task_with_demand(task_description, demand)
                        for trial_idx in range(args.num_trials_per_demand):
                            trial_seed = args.seed + task_id * 100000 + init_idx * 1000 + trial_idx
                            env.seed(trial_seed)
                            episode_rng = np.random.default_rng(trial_seed)
                            frames, metrics = _collect_episode(
                                client,
                                env,
                                init_state,
                                task_prompt,
                                policy_mode=args.policy_mode,
                                demand=demand,
                                resize_size=args.resize_size,
                                replan_steps=args.replan_steps,
                                num_steps_wait=args.num_steps_wait,
                                max_steps=max_steps,
                                random_action_scale=args.random_action_scale,
                                random_gripper_prob=args.random_gripper_prob,
                                save_semantic_states=args.save_semantic_states,
                                rng=episode_rng,
                            )
                            if not frames:
                                logging.warning(
                                    "Skipping empty episode for task_id=%s init_idx=%s demand=%s trial=%s",
                                    task_id,
                                    init_idx,
                                    demand,
                                    trial_idx,
                                )
                                continue

                            metadata = {
                                "episode_index": episode_index,
                                "task_id": task_id,
                                "task_description": task_description,
                                "task_prompt": task_prompt,
                                "demand_label": demand,
                                "init_state_index": init_idx,
                                "trial_index": trial_idx,
                                **metrics,
                            }
                            episode_dir = _save_raw_episode(
                                episodes_root,
                                episode_index,
                                frames,
                                metadata,
                                save_videos=args.save_videos,
                                save_wrist_video=args.save_wrist_video,
                                save_combined_video=args.save_combined_video,
                                video_fps=args.video_fps,
                            )
                            metadata["episode_dir"] = str(pathlib.Path("episodes") / episode_dir.name)
                            semantic_states_path = episode_dir / "semantic_states.jsonl"
                            metadata["semantic_states_path"] = (
                                str(pathlib.Path("episodes") / episode_dir.name / semantic_states_path.name)
                                if semantic_states_path.exists()
                                else None
                            )
                            for video_name in ("agentview.mp4", "wrist.mp4", "combined.mp4"):
                                video_path = episode_dir / video_name
                                metadata[f"{video_path.stem}_video_path"] = (
                                    str(pathlib.Path("episodes") / episode_dir.name / video_name)
                                    if video_path.exists()
                                    else None
                                )
                            metadata_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                            metadata_file.flush()
                            episode_index += 1
            finally:
                env.close()

    logging.info("Saved raw demand-conditioned LIBERO episodes to: %s", output_path)
    logging.info("Saved raw episode metadata to: %s", metadata_path)
    logging.info("Saved raw dataset manifest to: %s", manifest_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
