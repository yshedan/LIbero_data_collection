from __future__ import annotations

import dataclasses
import json
import logging
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

import imageio.v2 as imageio
import numpy as np
import tyro

from lerobot_compat import (
    HF_LEROBOT_HOME,
    LeRobotDataset,
    add_frame,
    create_dataset,
    open_dataset_for_writing,
    write_norm_stats,
)

LIBERO_LEROBOT_FEATURES = {
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
}


@dataclasses.dataclass
class Args:
    source_dir: str = "data/libero_demand_raw"
    target_repo_id: str = "your_hf_username/libero_demand"
    target_dir: str | None = None
    append: bool = True
    overwrite: bool = False
    source_metadata_filename: str = "episode_metadata.jsonl"
    target_metadata_filename: str = "episode_metadata.jsonl"


@dataclasses.dataclass(frozen=True)
class SourceEpisode:
    root: pathlib.Path
    record: dict[str, Any]


def _canonical_path(path: pathlib.Path) -> str:
    return str(path.expanduser().resolve(strict=False))


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


def _find_source_metadata_paths(
    source_root: pathlib.Path,
    metadata_filename: str,
) -> list[pathlib.Path]:
    direct_path = source_root / metadata_filename
    if direct_path.exists():
        return [direct_path]

    nested_paths = sorted(
        path
        for path in source_root.rglob(metadata_filename)
        if path.is_file()
    )
    if nested_paths:
        return nested_paths

    raise FileNotFoundError(
        f"Metadata file does not exist: {direct_path}. "
        f"Also found no nested {metadata_filename} under {source_root}."
    )


def _load_source_episodes(
    source_root: pathlib.Path,
    metadata_filename: str,
) -> list[SourceEpisode]:
    source_episodes: list[SourceEpisode] = []
    for metadata_path in _find_source_metadata_paths(source_root, metadata_filename):
        records = _load_metadata(metadata_path)
        if not records:
            logging.warning("Skipping empty metadata file: %s", metadata_path)
            continue
        metadata_root = metadata_path.parent
        for record in records:
            source_episodes.append(SourceEpisode(metadata_root, record))
        if metadata_path.parent != source_root:
            logging.info(
                "Loaded %s raw episodes from nested source: %s",
                len(records),
                metadata_path,
            )

    return source_episodes


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
    episodes_path = output_path / "meta" / "episodes.jsonl"
    return _jsonl_line_count(episodes_path)


def _source_episode_key(source_root: pathlib.Path, record: dict[str, Any]) -> str:
    source_demo_file = record.get("source_demo_file")
    if source_demo_file:
        return _canonical_path(pathlib.Path(str(source_demo_file)))
    return _canonical_path(source_root / str(record["episode_dir"]))


def _metadata_record_for_existing_episode(
    source_episode: SourceEpisode,
    target_root: pathlib.Path,
    target_episode_index: int,
) -> dict[str, Any]:
    record = dict(source_episode.record)
    source_episode_index = int(record["episode_index"])
    semantic_path = target_root / "semantic_states" / f"episode_{target_episode_index:06d}.jsonl"
    record["source_episode_index"] = source_episode_index
    record["source_episode_key"] = _source_episode_key(source_episode.root, record)
    record["source_raw_dir"] = _canonical_path(source_episode.root)
    record["source_raw_episode_dir"] = _canonical_path(
        source_episode.root / str(record["episode_dir"])
    )
    record["episode_index"] = target_episode_index
    record["semantic_states_path"] = (
        str(pathlib.Path("semantic_states") / semantic_path.name)
        if semantic_path.exists()
        else None
    )
    return record


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
        features=LIBERO_LEROBOT_FEATURES,
        image_writer_threads=2,
        image_writer_processes=0,
    )
    return dataset, output_path


def _convert_episode(source_root: pathlib.Path, dataset: LeRobotDataset, record: dict[str, Any]) -> None:
    episode_dir = source_root / str(record["episode_dir"])
    image_dir = episode_dir / "image"
    wrist_image_dir = episode_dir / "wrist_image"
    states = np.load(episode_dir / "states.npy")
    actions = np.load(episode_dir / "actions.npy")
    task_prompt = str(record["task_prompt"])
    num_frames = int(record["num_steps"])

    if len(states) != num_frames or len(actions) != num_frames:
        raise ValueError(
            f"Episode {record['episode_index']} has inconsistent lengths: "
            f"states={len(states)}, actions={len(actions)}, num_steps={num_frames}"
        )

    for frame_idx in range(num_frames):
        image = imageio.imread(image_dir / f"{frame_idx:06d}.png")
        wrist_image = imageio.imread(wrist_image_dir / f"{frame_idx:06d}.png")
        add_frame(
            dataset,
            {
                "image": np.asarray(image, dtype=np.uint8),
                "wrist_image": np.asarray(wrist_image, dtype=np.uint8),
                "state": np.asarray(states[frame_idx], dtype=np.float32),
                "actions": np.asarray(actions[frame_idx], dtype=np.float32),
                "task": task_prompt,
            },
        )
    dataset.save_episode()
    # The converter only appends episodes. Keeping every saved episode in the
    # in-memory Hugging Face Dataset can grow memory until the OS kills the
    # process during long multi-task conversions.
    dataset.hf_dataset = dataset.create_hf_dataset()


def _copy_semantic_states(
    source_root: pathlib.Path,
    target_root: pathlib.Path,
    record: dict[str, Any],
    target_episode_index: int,
) -> str | None:
    source_path = source_root / str(record["episode_dir"]) / "semantic_states.jsonl"
    if not source_path.exists():
        return None

    semantic_root = target_root / "semantic_states"
    semantic_root.mkdir(parents=True, exist_ok=True)
    target_path = semantic_root / f"episode_{target_episode_index:06d}.jsonl"
    shutil.copy2(source_path, target_path)
    return str(pathlib.Path("semantic_states") / target_path.name)


def main(args: Args) -> None:
    if args.overwrite:
        args.append = False

    source_root = pathlib.Path(args.source_dir)
    source_episodes = _load_source_episodes(source_root, args.source_metadata_filename)
    if not source_episodes:
        raise ValueError(f"No episode metadata found under {source_root}")

    dataset, output_path = _make_target_dataset(
        args.target_repo_id,
        target_dir=args.target_dir,
        append=args.append,
        overwrite=args.overwrite,
    )

    target_metadata_path = output_path / args.target_metadata_filename
    existing_records = (
        _load_metadata(target_metadata_path)
        if args.append and target_metadata_path.exists()
        else []
    )
    recovered_existing_records = False
    if args.append and not existing_records and not target_metadata_path.exists():
        completed_episode_count = _target_total_episodes(output_path)
        if completed_episode_count > 0:
            if completed_episode_count > len(source_episodes):
                raise ValueError(
                    f"Target dataset already contains {completed_episode_count} episodes, "
                    f"but only {len(source_episodes)} source episodes were found."
                )
            existing_records = [
                _metadata_record_for_existing_episode(
                    source_episode,
                    output_path,
                    target_episode_index,
                )
                for target_episode_index, source_episode in enumerate(
                    source_episodes[:completed_episode_count]
                )
            ]
            recovered_existing_records = True
            logging.warning(
                "Recovered metadata for %s already converted episodes from %s. "
                "The previous conversion likely stopped before writing %s.",
                completed_episode_count,
                output_path / "meta" / "episodes.jsonl",
                target_metadata_path,
            )
    existing_source_keys = {
        str(record["source_episode_key"])
        for record in existing_records
        if record.get("source_episode_key") is not None
    }
    for record in existing_records:
        source_demo_file = record.get("source_demo_file")
        if source_demo_file:
            existing_source_keys.add(_canonical_path(pathlib.Path(str(source_demo_file))))
    next_episode_index = max(
        (int(record["episode_index"]) for record in existing_records),
        default=-1,
    ) + 1

    new_records = []
    for source_episode in source_episodes:
        source_root_for_episode = source_episode.root
        source_record = source_episode.record
        source_episode_index = int(source_record["episode_index"])
        source_key = _source_episode_key(source_root_for_episode, source_record)
        if source_key in existing_source_keys:
            logging.info("Skipping already converted source episode: %s", source_key)
            continue

        record = dict(source_record)
        target_episode_index = next_episode_index + len(new_records)
        _convert_episode(source_root_for_episode, dataset, record)
        record["source_episode_index"] = source_episode_index
        record["source_episode_key"] = source_key
        record["source_raw_dir"] = _canonical_path(source_root_for_episode)
        record["source_raw_episode_dir"] = _canonical_path(
            source_root_for_episode / str(record["episode_dir"])
        )
        record["episode_index"] = target_episode_index
        record["semantic_states_path"] = _copy_semantic_states(
            source_root_for_episode,
            output_path,
            record,
            target_episode_index,
        )
        new_records.append(record)
        existing_source_keys.add(source_key)

    metadata_mode = "a" if existing_records and not recovered_existing_records else "w"
    with target_metadata_path.open(metadata_mode, encoding="utf-8") as f:
        if recovered_existing_records:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        for record in new_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    norm_stats_path = write_norm_stats(output_path)

    logging.info(
        "Appended %s raw episodes; dataset now contains %s episodes: %s",
        len(new_records),
        len(existing_records) + len(new_records),
        output_path,
    )
    logging.info("Copied episode metadata to: %s", target_metadata_path)
    logging.info("Wrote normalization stats to: %s", norm_stats_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
