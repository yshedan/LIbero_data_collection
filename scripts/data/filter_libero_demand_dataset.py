from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import shutil
from typing import Any

_DEFAULT_CACHE_ROOT = pathlib.Path(
    os.environ.get("LIBERO_CONVERSION_CACHE_DIR", "data/.cache")
).expanduser()
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    str((_DEFAULT_CACHE_ROOT / "huggingface_datasets").resolve()),
)

import numpy as np
import tqdm
import tyro

from lerobot_compat import (
    HF_LEROBOT_HOME,
    LeRobotDataset,
    add_frame,
    create_dataset,
    open_dataset_for_writing,
    write_norm_stats,
)


@dataclasses.dataclass
class Args:
    source_repo_id: str = "your_hf_username/libero_demand"
    target_repo_id: str = "your_hf_username/libero_demand_filtered"
    source_dir: str | None = None
    target_dir: str | None = None
    append: bool = True
    overwrite: bool = False
    source_metadata_filename: str = "episode_metadata.jsonl"
    target_metadata_filename: str = "episode_metadata.jsonl"
    summary_filename: str = "selection_summary.json"
    copy_preview_videos: bool = False

    # If unset, each demand receives selection_fraction of the successful
    # trajectories for a task. Set this to an integer for a fixed group size.
    target_episodes_per_task_demand: int | None = None
    selection_fraction: float = 0.25
    demands: tuple[str, ...] = ("fast", "stable", "safe")

    # True safety metrics are optional because older collected data does not
    # contain contact and table-edge information.
    max_dangerous_collisions: int = 0
    min_edge_distance: float = 0.03
    max_dangerous_edge_ratio: float = 0.05


def _load_metadata(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                records.append(json.loads(stripped_line))
    return records


def _jsonl_line_count(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _target_total_episodes(output_path: pathlib.Path) -> int:
    return _jsonl_line_count(output_path / "meta" / "episodes.jsonl")


def _metadata_record_for_existing_selected_episode(
    selected_record: dict[str, Any],
    target_root: pathlib.Path,
    target_episode_index: int,
) -> dict[str, Any]:
    record = dict(selected_record)
    semantic_path = target_root / "semantic_states" / f"episode_{target_episode_index:06d}.jsonl"
    record["episode_index"] = target_episode_index
    record["semantic_states_path"] = (
        str(pathlib.Path("semantic_states") / semantic_path.name)
        if semantic_path.exists()
        else None
    )
    for field_name in (
        "agentview_video_path",
        "wrist_video_path",
        "combined_video_path",
    ):
        record.pop(field_name, None)
    return record


def _format_task_with_demand(task_description: str, demand: str) -> str:
    demand_adverbs = {
        "fast": "quickly",
        "safe": "safely",
        "stable": "steadily",
    }
    if demand not in demand_adverbs:
        raise ValueError(f"Unsupported execution requirement: {demand}")
    description = task_description.strip().rstrip(".")
    return f"{description} {demand_adverbs[demand]}."


def _make_target_dataset(
    target_repo_id: str,
    *,
    target_dir: str | None,
    append: bool,
    overwrite: bool,
) -> tuple[LeRobotDataset, pathlib.Path]:
    output_path = (
        pathlib.Path(target_dir).expanduser().resolve()
        if target_dir
        else HF_LEROBOT_HOME / target_repo_id
    )
    if output_path.exists():
        if append:
            info_path = output_path / "meta" / "info.json"
            if info_path.exists():
                return (
                    open_dataset_for_writing(
                        target_repo_id,
                        root=output_path,
                        image_writer_threads=2,
                        image_writer_processes=0,
                    ),
                    output_path,
                )
            if not any(output_path.iterdir()):
                shutil.rmtree(output_path)
            else:
                raise FileExistsError(
                    f"Target path exists but is not a valid LeRobot dataset: {output_path}. "
                    f"Missing metadata file: {info_path}. "
                    "Pass --overwrite to replace it, or choose a different --target-dir."
                )
        elif not overwrite:
            raise FileExistsError(
                f"Target dataset path already exists: {output_path}. "
                "Pass --append to extend it or --overwrite to replace it."
            )
        else:
            shutil.rmtree(output_path)

    dataset = create_dataset(
        repo_id=target_repo_id,
        robot_type="panda",
        fps=10,
        root=output_path,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=2,
        image_writer_processes=0,
    )
    return dataset, output_path


def _index_source_dataset(
    source_dataset: LeRobotDataset,
) -> tuple[dict[int, list[int]], dict[int, list[np.ndarray]]]:
    """Index rows and actions without retaining decoded images in memory."""
    rows_by_episode: dict[int, list[int]] = collections.defaultdict(list)
    actions_by_episode: dict[int, list[np.ndarray]] = collections.defaultdict(list)
    episode_indices = source_dataset.hf_dataset["episode_index"]
    actions = source_dataset.hf_dataset["actions"]
    iterator = zip(episode_indices, actions)
    for row_index, (episode_index, action) in enumerate(
        tqdm.tqdm(iterator, total=len(episode_indices), desc="Indexing source actions")
    ):
        episode_index = int(np.asarray(episode_index).item())
        rows_by_episode[episode_index].append(row_index)
        actions_by_episode[episode_index].append(np.asarray(action, dtype=np.float32))
    return rows_by_episode, actions_by_episode


def _recompute_motion_metrics(record: dict[str, Any], actions: list[np.ndarray]) -> None:
    """Recompute action metrics using only the six motion dimensions."""
    action_array = np.stack(actions, axis=0)
    motion_actions = action_array[:, :6]
    motion_norms = np.linalg.norm(motion_actions, axis=1)

    record["mean_motion_norm"] = float(np.mean(motion_norms))
    record["motion_energy_per_step"] = float(np.mean(np.square(motion_norms)))
    if len(motion_actions) > 1:
        motion_delta_norms = np.linalg.norm(np.diff(motion_actions, axis=0), axis=1)
        record["motion_smoothness_score"] = float(np.mean(motion_delta_norms))
        record["max_motion_delta"] = float(np.max(motion_delta_norms))
    else:
        record["motion_smoothness_score"] = 0.0
        record["max_motion_delta"] = 0.0


def _to_hwc_uint8(image: Any) -> np.ndarray:
    """Convert LeRobot image tensors or arrays to an HWC uint8 array."""
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    else:
        image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape}.")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))

    if np.issubdtype(image.dtype, np.floating):
        if image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0)
    return np.ascontiguousarray(image, dtype=np.uint8)


def _percentile_ranks(values: list[float], *, higher_is_better: bool = False) -> np.ndarray:
    """Return tie-aware ranks in [0, 1], where lower always means better."""
    array = np.asarray(values, dtype=np.float64)
    if higher_is_better:
        array = -array
    if len(array) <= 1:
        return np.zeros(len(array), dtype=np.float64)

    ranks = np.empty(len(array), dtype=np.float64)
    for idx, value in enumerate(array):
        less = np.count_nonzero(array < value)
        equal = np.count_nonzero(array == value)
        ranks[idx] = (less + 0.5 * (equal - 1)) / (len(array) - 1)
    return ranks


def _has_true_safety_metrics(records: list[dict[str, Any]]) -> bool:
    required = {"dangerous_collision_count", "minimum_edge_distance", "dangerous_edge_steps"}
    return bool(records) and all(required.issubset(record) for record in records)


def _compute_scores(records: list[dict[str, Any]], *, use_true_safety: bool) -> None:
    num_steps_rank = _percentile_ranks([float(record["num_steps"]) for record in records])
    elapsed_rank = _percentile_ranks([float(record["elapsed_time"]) for record in records])
    mean_motion_rank = _percentile_ranks([float(record["mean_motion_norm"]) for record in records])
    energy_rank = _percentile_ranks([float(record["motion_energy_per_step"]) for record in records])
    smoothness_rank = _percentile_ranks([float(record["motion_smoothness_score"]) for record in records])
    max_delta_rank = _percentile_ranks([float(record["max_motion_delta"]) for record in records])

    if use_true_safety:
        collision_rank = _percentile_ranks([float(record["dangerous_collision_count"]) for record in records])
        edge_steps_rank = _percentile_ranks(
            [float(record["dangerous_edge_steps"]) / max(1.0, float(record["num_steps"])) for record in records]
        )
        edge_distance_rank = _percentile_ranks(
            [float(record["minimum_edge_distance"]) for record in records],
            higher_is_better=True,
        )

    for idx, record in enumerate(records):
        record["demand_scores"] = {
            "fast": float(0.8 * num_steps_rank[idx] + 0.2 * elapsed_rank[idx]),
            "stable": float(0.6 * smoothness_rank[idx] + 0.4 * max_delta_rank[idx]),
        }
        if use_true_safety:
            record["demand_scores"]["safe"] = float(
                0.5 * collision_rank[idx] + 0.3 * edge_steps_rank[idx] + 0.2 * edge_distance_rank[idx]
            )
        else:
            record["demand_scores"]["safe"] = float(
                0.4 * mean_motion_rank[idx] + 0.4 * max_delta_rank[idx] + 0.2 * energy_rank[idx]
            )


def _safe_is_eligible(record: dict[str, Any], args: Args, *, use_true_safety: bool) -> bool:
    if not use_true_safety:
        return True
    edge_ratio = float(record["dangerous_edge_steps"]) / max(1.0, float(record["num_steps"]))
    return (
        int(record["dangerous_collision_count"]) <= args.max_dangerous_collisions
        and float(record["minimum_edge_distance"]) >= args.min_edge_distance
        and edge_ratio <= args.max_dangerous_edge_ratio
    )


def _target_count(group_size: int, args: Args) -> int:
    if args.target_episodes_per_task_demand is not None:
        return args.target_episodes_per_task_demand
    return max(1, math.floor(group_size * args.selection_fraction))


def _assign_labels(
    records: list[dict[str, Any]],
    args: Args,
    *,
    use_true_safety: bool,
) -> list[dict[str, Any]]:
    """Assign at most one demand label to each trajectory.

    Fast and stable are selected by their own demand scores. Safe is treated as
    the residual bucket because older datasets only have a motion-proxy safety
    score, which is not a reliable measure of real task safety.
    """
    target = _target_count(len(records), args)
    assigned_indices: set[int] = set()
    counts = dict.fromkeys(args.demands, 0)
    selected: list[dict[str, Any]] = []

    def add_labeled_record(index: int, demand: str) -> None:
        record = records[index]
        assigned_indices.add(index)
        counts[demand] += 1

        labeled = dict(record)
        labeled["source_episode_index"] = int(record["episode_index"])
        labeled["original_demand_label"] = record.get("demand_label")
        labeled["original_task_prompt"] = record.get("task_prompt")
        labeled["demand_label"] = demand
        labeled["demand_score"] = float(record["demand_scores"][demand])
        labeled["safety_metric_mode"] = "environment" if use_true_safety else "motion_proxy"
        labeled["task_prompt"] = _format_task_with_demand(str(record["task_description"]), demand)
        selected.append(labeled)

    for demand in args.demands:
        remaining_indices = [
            index
            for index, record in enumerate(records)
            if index not in assigned_indices
            and (
                demand != "safe"
                or _safe_is_eligible(record, args, use_true_safety=use_true_safety)
            )
        ]
        if demand == "safe":
            candidates = remaining_indices
        else:
            candidates = sorted(
                remaining_indices,
                key=lambda index: (float(records[index]["demand_scores"][demand]), index),
            )

        for index in candidates:
            if counts[demand] >= target:
                break
            add_labeled_record(index, demand)

    for demand, count in counts.items():
        if count < target:
            logging.warning(
                "Only assigned %s/%s trajectories for task=%s demand=%s.",
                count,
                target,
                records[0]["task_id"],
                demand,
            )
    return selected


def _select_records(
    records: list[dict[str, Any]],
    actions_by_episode: dict[int, list[np.ndarray]],
    args: Args,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    successful = [
        dict(record)
        for record in records
        if bool(record.get("success")) and float(record.get("total_reward", 0.0)) > 0.0
    ]
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in successful:
        episode_index = int(record["episode_index"])
        if episode_index not in actions_by_episode:
            raise ValueError(f"Missing source actions for episode {episode_index}.")
        _recompute_motion_metrics(record, actions_by_episode[episode_index])
        task_group_key = str(
            record.get("task_description")
            or record.get("task_prompt")
            or record.get("task_id")
        )
        grouped[task_group_key].append(record)

    selected: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for normalized_task_id, (task_group_key, task_records) in enumerate(grouped.items()):
        for record in task_records:
            record["original_task_id"] = record.get("task_id")
            record["task_id"] = normalized_task_id

        use_true_safety = _has_true_safety_metrics(task_records)
        if not use_true_safety:
            logging.warning(
                "Task %s has no collision/edge metrics; safe labels use motion-proxy metrics.",
                task_group_key,
            )
        _compute_scores(task_records, use_true_safety=use_true_safety)
        task_selected = _assign_labels(task_records, args, use_true_safety=use_true_safety)
        selected.extend(task_selected)

        target = _target_count(len(task_records), args)
        for demand in args.demands:
            selected_count = sum(record["demand_label"] == demand for record in task_selected)
            summary_rows.append(
                {
                    "task_id": normalized_task_id,
                    "task_group_key": task_group_key,
                    "task_name": task_records[0]["task_description"],
                    "demand_label": demand,
                    "available_successful_count": len(task_records),
                    "selected_count": selected_count,
                    "target_count": target,
                    "is_complete": selected_count >= target,
                    "safety_metric_mode": "environment" if use_true_safety else "motion_proxy",
                }
            )

    return selected, summary_rows


def _write_selected_episodes(
    source_dataset: LeRobotDataset,
    target_dataset: LeRobotDataset,
    source_root: pathlib.Path,
    target_root: pathlib.Path,
    selected_records: list[dict[str, Any]],
    rows_by_episode: dict[int, list[int]],
    *,
    start_episode_index: int,
    copy_preview_videos: bool,
) -> None:
    semantic_root = target_root / "semantic_states"
    preview_root = target_root / "preview_videos"
    for offset, record in enumerate(
        tqdm.tqdm(selected_records, desc="Writing selected episodes")
    ):
        target_episode_index = start_episode_index + offset
        source_episode_index = int(record["source_episode_index"])
        source_rows = rows_by_episode.get(source_episode_index)
        if not source_rows:
            raise ValueError(f"Missing source rows for episode {source_episode_index}.")
        for row_index in source_rows:
            frame = source_dataset.hf_dataset[row_index]
            add_frame(
                target_dataset,
                {
                    "image": _to_hwc_uint8(frame["image"]),
                    "wrist_image": _to_hwc_uint8(frame["wrist_image"]),
                    "state": np.asarray(frame["state"], dtype=np.float32),
                    "actions": np.asarray(frame["actions"], dtype=np.float32),
                },
                record["task_prompt"],
            )
        target_dataset.save_episode()
        # We only write episodes here; keeping all saved episodes in the
        # in-memory Hugging Face Dataset makes long conversions vulnerable to
        # OOM kills.
        target_dataset.hf_dataset = target_dataset.create_hf_dataset()
        record["episode_index"] = target_episode_index

        source_semantic_path = record.get("semantic_states_path")
        if source_semantic_path:
            source_path = source_root / str(source_semantic_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Semantic state file does not exist: {source_path}")
            semantic_root.mkdir(parents=True, exist_ok=True)
            target_path = semantic_root / f"episode_{target_episode_index:06d}.jsonl"
            shutil.copy2(source_path, target_path)
            record["semantic_states_path"] = str(pathlib.Path("semantic_states") / target_path.name)

        if copy_preview_videos:
            target_preview_dir = preview_root / f"episode_{target_episode_index:06d}"
            for field_name, filename in (
                ("agentview_video_path", "agentview.mp4"),
                ("wrist_video_path", "wrist.mp4"),
                ("combined_video_path", "combined.mp4"),
            ):
                source_video_path = record.get(field_name)
                if not source_video_path:
                    record[field_name] = None
                    continue
                source_path = source_root / str(source_video_path)
                if not source_path.exists():
                    logging.warning("Preview video does not exist: %s", source_path)
                    record[field_name] = None
                    continue
                target_preview_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_preview_dir / filename
                shutil.copy2(source_path, target_path)
                record[field_name] = str(
                    pathlib.Path("preview_videos") / target_preview_dir.name / filename
                )
        else:
            for field_name in (
                "agentview_video_path",
                "wrist_video_path",
                "combined_video_path",
            ):
                record.pop(field_name, None)


def main(args: Args) -> None:
    if args.overwrite:
        args.append = False
    unsupported_demands = set(args.demands) - {"fast", "safe", "stable"}
    if unsupported_demands:
        raise ValueError(f"Unsupported demands: {sorted(unsupported_demands)}")
    if not 0.0 < args.selection_fraction <= 1.0:
        raise ValueError("selection_fraction must be in (0, 1].")
    if args.target_episodes_per_task_demand is not None and args.target_episodes_per_task_demand <= 0:
        raise ValueError("target_episodes_per_task_demand must be positive.")

    source_path = (
        pathlib.Path(args.source_dir).expanduser().resolve()
        if args.source_dir
        else HF_LEROBOT_HOME / args.source_repo_id
    )
    source_records = _load_metadata(source_path / args.source_metadata_filename)
    source_dataset = LeRobotDataset(args.source_repo_id, root=source_path)
    rows_by_episode, actions_by_episode = _index_source_dataset(source_dataset)

    selected_records, summary_rows = _select_records(source_records, actions_by_episode, args)
    if not selected_records:
        raise ValueError("No successful episodes passed the filtering rules.")

    target_path = (
        pathlib.Path(args.target_dir).expanduser().resolve()
        if args.target_dir
        else HF_LEROBOT_HOME / args.target_repo_id
    )
    metadata_path = target_path / args.target_metadata_filename
    existing_records = (
        _load_metadata(metadata_path)
        if args.append and metadata_path.exists()
        else []
    )
    recovered_existing_records = False
    if args.append and not existing_records and not metadata_path.exists():
        completed_episode_count = _target_total_episodes(target_path)
        if completed_episode_count > 0:
            if completed_episode_count > len(selected_records):
                raise ValueError(
                    f"Target dataset already contains {completed_episode_count} episodes, "
                    f"but only {len(selected_records)} selected source episodes were found."
                )
            existing_records = [
                _metadata_record_for_existing_selected_episode(
                    selected_record,
                    target_path,
                    target_episode_index,
                )
                for target_episode_index, selected_record in enumerate(
                    selected_records[:completed_episode_count]
                )
            ]
            recovered_existing_records = True
            logging.warning(
                "Recovered metadata for %s already filtered episodes from %s. "
                "The previous filtering likely stopped before writing %s.",
                completed_episode_count,
                target_path / "meta" / "episodes.jsonl",
                metadata_path,
            )
    existing_source_indices = {
        int(record["source_episode_index"])
        for record in existing_records
        if record.get("source_episode_index") is not None
    }
    selected_records = [
        record
        for record in selected_records
        if int(record["source_episode_index"]) not in existing_source_indices
    ]

    target_dataset, target_path = _make_target_dataset(
        args.target_repo_id,
        target_dir=args.target_dir,
        append=args.append,
        overwrite=args.overwrite,
    )
    next_episode_index = max(
        (int(record["episode_index"]) for record in existing_records),
        default=-1,
    ) + 1
    _write_selected_episodes(
        source_dataset,
        target_dataset,
        source_path,
        target_path,
        selected_records,
        rows_by_episode,
        start_episode_index=next_episode_index,
        copy_preview_videos=args.copy_preview_videos,
    )

    metadata_path = target_path / args.target_metadata_filename
    metadata_mode = "a" if existing_records and not recovered_existing_records else "w"
    with metadata_path.open(metadata_mode, encoding="utf-8") as f:
        if recovered_existing_records:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        for record in selected_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    norm_stats_path = write_norm_stats(target_path)

    summary_path = target_path / args.summary_filename
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selection_fraction": args.selection_fraction,
                "target_episodes_per_task_demand": args.target_episodes_per_task_demand,
                "demands": list(args.demands),
                "successful_source_episodes": sum(
                    bool(record.get("success")) and float(record.get("total_reward", 0.0)) > 0.0
                    for record in source_records
                ),
                "selected_episodes": len(existing_records) + len(selected_records),
                "appended_episodes": len(selected_records),
                "rows": summary_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logging.info("Saved %s relabeled episodes to: %s", len(selected_records), target_path)
    logging.info("Saved relabeled metadata to: %s", metadata_path)
    logging.info("Saved selection summary to: %s", summary_path)
    logging.info("Wrote normalization stats to: %s", norm_stats_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
