from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
from typing import Any

import tyro


@dataclasses.dataclass
class Args:
    source_json: str = "data_split_json/libero10_demand_atomic_skills.json"
    output_json: str = "data_split_json/libero10_demand_atomic_skill_sequences.json"
    append: bool = True


def _load_annotations(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Atomic-skill annotation file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)
    if not isinstance(annotations, dict):
        raise ValueError("The annotation JSON must be indexed by episode id.")
    return annotations


def _extract_sequences(
    annotations: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    sequences: dict[str, dict[str, Any]] = {}
    for episode_id, annotation in annotations.items():
        segments = annotation.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"Episode {episode_id} does not contain any segments.")

        ordered_segments = sorted(segments, key=lambda segment: int(segment["start_frame"]))
        for previous, current in zip(ordered_segments, ordered_segments[1:]):
            if int(current["start_frame"]) <= int(previous["start_frame"]):
                raise ValueError(f"Episode {episode_id} contains unordered segment boundaries.")

        sequences[str(episode_id)] = {
            "atomic_skill_sequence": [
                str(segment["primary_action_verb"]) for segment in ordered_segments
            ],
            "task_id": int(annotation["task_id"]),
            "execution_requirement": str(annotation.get("demand_label", "")),
            "task_name": str(
                annotation.get("task_description")
                or annotation.get("task_prompt")
                or ordered_segments[0].get("action_name", "")
            ),
            "source_episode_index": annotation.get("source_episode_index"),
        }
    return sequences


def main(args: Args) -> None:
    source_path = pathlib.Path(args.source_json)
    annotations = _load_annotations(source_path)
    extracted_sequences = _extract_sequences(annotations)

    output_path = pathlib.Path(args.output_json)
    if args.append and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            sequences = json.load(f)
        if not isinstance(sequences, dict):
            raise ValueError("Existing sequence JSON must be indexed by episode id.")
    else:
        sequences = {}
    added_count = 0
    for episode_id, sequence in extracted_sequences.items():
        if args.append and episode_id in sequences:
            continue
        sequences[episode_id] = sequence
        added_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("{\n")
        items = list(sequences.items())
        for index, (episode_id, sequence) in enumerate(items):
            suffix = "," if index + 1 < len(items) else ""
            episode_json = json.dumps(str(episode_id), ensure_ascii=False)
            sequence_json = json.dumps(sequence, ensure_ascii=False)
            f.write(f"  {episode_json}: {sequence_json}{suffix}\n")
        f.write("}\n")

    logging.info(
        "Appended %s sequences; file now contains %s episodes: %s",
        added_count,
        len(sequences),
        output_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
