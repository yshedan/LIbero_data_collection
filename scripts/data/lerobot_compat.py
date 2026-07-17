"""Compatibility helpers for LeRobot dataset APIs across releases."""

from __future__ import annotations

import inspect
import json
import pathlib
from typing import Any

try:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
except ModuleNotFoundError:
    from lerobot.constants import HF_LEROBOT_HOME
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


def add_frame(
    dataset: LeRobotDataset,
    frame: dict[str, Any],
    task: str | None = None,
) -> None:
    """Add a frame using either the legacy or current task API.

    Older LeRobot examples store the language instruction as ``frame["task"]``.
    Newer LeRobot releases expect it as a separate ``task=...`` argument.
    This helper lets conversion scripts keep the example-style frame format
    while producing the correct dataset for the installed LeRobot version.
    """
    task = task if task is not None else str(frame["task"])
    parameters = inspect.signature(dataset.add_frame).parameters
    if "task" in parameters:
        dataset.add_frame(
            {name: value for name, value in frame.items() if name != "task"},
            task=task,
        )
    else:
        dataset.add_frame({**frame, "task": task})


def create_dataset(**kwargs: Any) -> LeRobotDataset:
    """Ignore create options that are unavailable in the installed release."""
    parameters = inspect.signature(LeRobotDataset.create).parameters
    supported_kwargs = {
        name: value for name, value in kwargs.items() if name in parameters
    }
    return LeRobotDataset.create(**supported_kwargs)


def open_dataset_for_writing(
    repo_id: str,
    *,
    root: str | pathlib.Path,
    image_writer_threads: int = 2,
    image_writer_processes: int = 0,
) -> LeRobotDataset:
    """Open an existing LeRobot dataset for appending without loading parquet data.

    Newer LeRobotDataset(repo_id, root=...) eagerly loads all existing parquet
    files through Hugging Face datasets. That is useful for training, but costly
    for conversion resume because it creates Arrow cache files under ~/.cache.
    For appending we only need metadata plus an empty in-memory dataset for the
    next episode, so we mirror LeRobotDataset.create while loading existing
    metadata from disk.
    """
    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.meta = LeRobotDatasetMetadata(repo_id, root=root)
    dataset.repo_id = dataset.meta.repo_id
    dataset.root = dataset.meta.root
    dataset.revision = None
    dataset.tolerance_s = 1e-4
    dataset.image_writer = None
    dataset.batch_encoding_size = 1
    dataset.episodes_since_last_encoding = 0
    dataset.episodes = None
    dataset.hf_dataset = dataset.create_hf_dataset()
    dataset.image_transforms = None
    dataset.delta_timestamps = None
    dataset.delta_indices = None
    dataset.episode_data_index = None
    dataset.video_backend = None

    if image_writer_processes or image_writer_threads:
        dataset.start_image_writer(image_writer_processes, image_writer_threads)
    dataset.episode_buffer = dataset.create_episode_buffer()
    return dataset


def _as_float_list(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(value)]


def write_norm_stats(
    dataset_root: str | pathlib.Path,
    *,
    feature_names: tuple[str, ...] = ("state", "actions"),
    filename: str = "norm_stats.json",
) -> pathlib.Path:
    """Write dataset-level normalization stats merged from LeRobot episode stats."""
    dataset_root = pathlib.Path(dataset_root)
    episode_stats_path = dataset_root / "meta" / "episodes_stats.jsonl"
    if not episode_stats_path.exists():
        raise FileNotFoundError(f"Episode stats file does not exist: {episode_stats_path}")

    accumulators: dict[str, dict[str, Any]] = {}
    with episode_stats_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            episode_stats = json.loads(line)["stats"]
            for feature_name in feature_names:
                if feature_name not in episode_stats:
                    continue
                stats = episode_stats[feature_name]
                count = int(_as_float_list(stats["count"])[0])
                if count <= 0:
                    continue
                mean = _as_float_list(stats["mean"])
                std = _as_float_list(stats["std"])
                minimum = _as_float_list(stats["min"])
                maximum = _as_float_list(stats["max"])

                accumulator = accumulators.setdefault(
                    feature_name,
                    {
                        "count": 0,
                        "sum": [0.0] * len(mean),
                        "sum_sq": [0.0] * len(mean),
                        "min": minimum.copy(),
                        "max": maximum.copy(),
                    },
                )
                accumulator["count"] += count
                for index, (mean_value, std_value) in enumerate(zip(mean, std)):
                    accumulator["sum"][index] += count * mean_value
                    accumulator["sum_sq"][index] += count * (
                        std_value * std_value + mean_value * mean_value
                    )
                    accumulator["min"][index] = min(accumulator["min"][index], minimum[index])
                    accumulator["max"][index] = max(accumulator["max"][index], maximum[index])

    norm_stats = {}
    for feature_name, accumulator in accumulators.items():
        count = int(accumulator["count"])
        mean = [value / count for value in accumulator["sum"]]
        variance = [
            max(0.0, sum_sq / count - mean_value * mean_value)
            for sum_sq, mean_value in zip(accumulator["sum_sq"], mean)
        ]
        norm_stats[feature_name] = {
            "mean": mean,
            "std": [variance_value**0.5 for variance_value in variance],
            "min": accumulator["min"],
            "max": accumulator["max"],
            "count": count,
        }

    output_path = dataset_root / filename
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(norm_stats, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return output_path
