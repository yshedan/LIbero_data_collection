from __future__ import annotations

import collections
import dataclasses
import json
import logging
import pathlib
from typing import Any

import numpy as np
import tyro

from lerobot_compat import HF_LEROBOT_HOME


VALID_SKILLS = {"approach", "grasp", "lift", "transport", "release", "place", "turn"}


@dataclasses.dataclass
class Args:
    source_repo_id: str = "local/libero10_demand_filtered"
    source_dir: str | None = None
    metadata_filename: str = "episode_metadata.jsonl"
    output_json: str = "data_split_json/libero10_demand_atomic_skills.json"
    append: bool = True

    min_grasp_frames: int = 3
    pre_grasp_contact_window: int = 12
    release_duration: int = 5
    lift_height_threshold: float = 0.025
    joint_motion_threshold: float = 1e-4
    predicate_stability_frames: int = 3
    min_segment_frames: int = 2


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                records.append(json.loads(stripped_line))
    return records


def _stable_true_start(values: list[bool], start: int, stability_frames: int) -> int | None:
    for index in range(start, len(values) - stability_frames + 1):
        if all(values[index : index + stability_frames]):
            return index
    return None


def _true_runs(values: list[bool], min_length: int) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, value in enumerate([*values, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            end = index - 1
            if end - start + 1 >= min_length:
                runs.append((start, end))
            start = None
    return runs


def _object_grasp_runs(
    semantic_frames: list[dict[str, Any]],
    min_grasp_frames: int,
) -> list[tuple[int, int, str]]:
    object_names = sorted(
        {
            object_name
            for frame in semantic_frames
            for object_name in frame.get("grasped_objects", [])
        }
    )
    runs = []
    for object_name in object_names:
        grasped = [object_name in frame.get("grasped_objects", []) for frame in semantic_frames]
        runs.extend((*run, object_name) for run in _true_runs(grasped, min_grasp_frames))
    return sorted(runs)


def _goal_series(
    semantic_frames: list[dict[str, Any]],
) -> dict[tuple[str, ...], list[bool]]:
    series: dict[tuple[str, ...], list[bool]] = collections.defaultdict(list)
    predicates = {
        tuple(predicate["predicate"])
        for frame in semantic_frames
        for predicate in frame.get("goal_predicates", [])
    }
    for predicate in predicates:
        for frame in semantic_frames:
            values = {
                tuple(item["predicate"]): bool(item["value"])
                for item in frame.get("goal_predicates", [])
            }
            series[predicate].append(values.get(predicate, False))
    return series


def _object_goal_predicate(
    goal_series: dict[tuple[str, ...], list[bool]],
    object_name: str,
) -> tuple[str, ...] | None:
    candidates = [
        predicate
        for predicate in goal_series
        if len(predicate) >= 2 and predicate[1] == object_name and predicate[0] in {"in", "on"}
    ]
    return candidates[0] if candidates else None


def _object_height(semantic_frames: list[dict[str, Any]], frame_index: int, object_name: str) -> float | None:
    position = semantic_frames[frame_index].get("objects", {}).get(object_name, {}).get("position")
    if position is None:
        return None
    return float(position[2])


def _contact_start(
    semantic_frames: list[dict[str, Any]],
    object_name: str,
    grasp_start: int,
    window: int,
) -> int:
    lower = max(0, grasp_start - window)
    for index in range(lower, grasp_start + 1):
        if object_name in semantic_frames[index].get("gripper_contact_objects", []):
            return index
    return grasp_start


def _lift_end(
    semantic_frames: list[dict[str, Any]],
    object_name: str,
    start: int,
    end: int,
    threshold: float,
) -> int | None:
    initial_height = _object_height(semantic_frames, start, object_name)
    if initial_height is None:
        return None
    for index in range(start + 1, end + 1):
        height = _object_height(semantic_frames, index, object_name)
        if height is not None and height - initial_height >= threshold:
            return index
    return None


def _turn_interval(
    semantic_frames: list[dict[str, Any]],
    goal_series: dict[tuple[str, ...], list[bool]],
    args: Args,
) -> tuple[int, int, str] | None:
    turn_predicates = [predicate for predicate in goal_series if predicate and predicate[0] == "turnon"]
    if not turn_predicates:
        return None
    predicate = turn_predicates[0]
    object_name = predicate[1]
    turn_end = _stable_true_start(goal_series[predicate], 0, args.predicate_stability_frames)
    if turn_end is None:
        return None

    joint_values = []
    for frame in semantic_frames:
        qpos = frame.get("objects", {}).get(object_name, {}).get("joint_qpos", [])
        joint_values.append(float(np.ravel(qpos)[0]) if len(qpos) else np.nan)
    turn_start = 0
    for index in range(1, turn_end + 1):
        if np.isfinite(joint_values[index - 1 : index + 1]).all() and (
            abs(joint_values[index] - joint_values[index - 1]) >= args.joint_motion_threshold
        ):
            turn_start = index - 1
            break
    return turn_start, turn_end, object_name


def _append_phase(
    phases: list[dict[str, Any]],
    *,
    skill: str,
    start: int,
    end: int,
    object_name: str,
    evidence: str,
    confidence: float,
) -> None:
    if end < start:
        return
    phases.append(
        {
            "skill": skill,
            "start": start,
            "end": end,
            "object_name": object_name,
            "evidence": evidence,
            "confidence": confidence,
        }
    )


def _detect_phases(semantic_frames: list[dict[str, Any]], args: Args) -> list[dict[str, Any]]:
    num_frames = len(semantic_frames)
    goal_series = _goal_series(semantic_frames)
    grasp_runs = _object_grasp_runs(semantic_frames, args.min_grasp_frames)
    phases: list[dict[str, Any]] = []
    cursor = 0

    turn_interval = _turn_interval(semantic_frames, goal_series, args)
    if turn_interval is not None:
        turn_start, turn_end, object_name = turn_interval
        if turn_start > cursor:
            _append_phase(
                phases,
                skill="approach",
                start=cursor,
                end=turn_start - 1,
                object_name=object_name,
                evidence="eef motion before articulated-object joint motion",
                confidence=0.65,
            )
        _append_phase(
            phases,
            skill="turn",
            start=turn_start,
            end=turn_end,
            object_name=object_name,
            evidence="joint motion followed by turnon predicate",
            confidence=0.95,
        )
        cursor = turn_end + 1

    for run_index, (grasp_start, grasp_end, object_name) in enumerate(grasp_runs):
        if grasp_end < cursor:
            continue
        grasp_start = max(grasp_start, cursor)
        contact_start = _contact_start(
            semantic_frames,
            object_name,
            grasp_start,
            args.pre_grasp_contact_window,
        )
        next_grasp_start = grasp_runs[run_index + 1][0] if run_index + 1 < len(grasp_runs) else num_frames

        if contact_start > cursor:
            _append_phase(
                phases,
                skill="approach",
                start=cursor,
                end=contact_start - 1,
                object_name=object_name,
                evidence="movement before first gripper contact",
                confidence=0.8,
            )

        stable_grasp_end = min(grasp_end, grasp_start + args.min_grasp_frames - 1)
        _append_phase(
            phases,
            skill="grasp",
            start=contact_start,
            end=stable_grasp_end,
            object_name=object_name,
            evidence="gripper contact followed by stable grasp",
            confidence=0.95,
        )

        lift_end = _lift_end(
            semantic_frames,
            object_name,
            stable_grasp_end,
            grasp_end,
            args.lift_height_threshold,
        )
        transport_start = stable_grasp_end + 1
        if lift_end is not None:
            _append_phase(
                phases,
                skill="lift",
                start=transport_start,
                end=lift_end,
                object_name=object_name,
                evidence=f"object height increased by {args.lift_height_threshold:.3f} m",
                confidence=0.9,
            )
            transport_start = lift_end + 1

        _append_phase(
            phases,
            skill="transport",
            start=transport_start,
            end=grasp_end,
            object_name=object_name,
            evidence="object remained grasped while moving",
            confidence=0.9,
        )

        release_end = min(num_frames - 1, grasp_end + args.release_duration)
        _append_phase(
            phases,
            skill="release",
            start=grasp_end + 1,
            end=release_end,
            object_name=object_name,
            evidence="stable grasp ended",
            confidence=0.85,
        )

        goal_predicate = _object_goal_predicate(goal_series, object_name)
        place_end = None
        if goal_predicate is not None:
            place_end = _stable_true_start(
                goal_series[goal_predicate],
                release_end,
                args.predicate_stability_frames,
            )
        if place_end is not None:
            place_end = min(place_end, next_grasp_start - 1)
            _append_phase(
                phases,
                skill="place",
                start=release_end + 1,
                end=place_end,
                object_name=object_name,
                evidence=f"goal predicate became stable: {' '.join(goal_predicate)}",
                confidence=0.95,
            )
            cursor = place_end + 1
        else:
            cursor = release_end + 1

    if cursor < num_frames:
        fallback_skill = "place" if phases and phases[-1]["skill"] == "release" else "approach"
        _append_phase(
            phases,
            skill=fallback_skill,
            start=cursor,
            end=num_frames - 1,
            object_name=phases[-1]["object_name"] if phases else "unknown",
            evidence="trajectory tail without a stronger semantic event",
            confidence=0.35,
        )
    return _normalize_phases(phases, num_frames, args.min_segment_frames)


def _normalize_phases(phases: list[dict[str, Any]], num_frames: int, min_segment_frames: int) -> list[dict[str, Any]]:
    """Make phases contiguous and merge very short or adjacent equal-skill phases."""
    phases = sorted(phases, key=lambda phase: (phase["start"], phase["end"]))
    normalized = []
    cursor = 0
    for phase in phases:
        phase_start = int(phase["start"])
        if phase_start > cursor:
            normalized.append(
                {
                    "skill": "approach",
                    "start": cursor,
                    "end": phase_start - 1,
                    "object_name": phase["object_name"],
                    "evidence": "movement before the next detected semantic event",
                    "confidence": 0.5,
                }
            )
            cursor = phase_start

        start = max(cursor, int(phase["start"]))
        end = min(num_frames - 1, int(phase["end"]))
        if end < start:
            continue
        phase = {**phase, "start": start, "end": end}
        if normalized and phase["skill"] == normalized[-1]["skill"] and phase["object_name"] == normalized[-1]["object_name"]:
            normalized[-1]["end"] = end
            normalized[-1]["confidence"] = min(normalized[-1]["confidence"], phase["confidence"])
        elif end - start + 1 < min_segment_frames and normalized:
            normalized[-1]["end"] = end
            normalized[-1]["confidence"] = min(normalized[-1]["confidence"], phase["confidence"])
        else:
            normalized.append(phase)
        cursor = end + 1

    if not normalized:
        normalized = [
            {
                "skill": "approach",
                "start": 0,
                "end": num_frames - 1,
                "object_name": "unknown",
                "evidence": "no semantic event detected",
                "confidence": 0.1,
            }
        ]
    elif normalized[-1]["end"] < num_frames - 1:
        normalized[-1]["end"] = num_frames - 1
    for previous, current in zip(normalized, normalized[1:]):
        if current["start"] != previous["end"] + 1:
            raise ValueError(
                f"Non-contiguous annotation boundary: {previous['end']} -> {current['start']}"
            )
    return normalized


def _phase_description(phase: dict[str, Any]) -> str:
    skill = str(phase["skill"])
    object_name = str(phase.get("object_name") or "target object").replace("_", " ")
    descriptions = {
        "approach": f"approach the {object_name}",
        "grasp": f"grasp the {object_name}",
        "lift": f"lift the {object_name}",
        "transport": f"transport the {object_name} toward its target",
        "release": f"release the {object_name}",
        "place": f"place the {object_name} at its target",
        "turn": f"turn the {object_name}",
    }
    return descriptions[skill]


def _format_chain_of_thought(
    chain: list[str],
    step_index: int,
    skill: str,
) -> str:
    return (
        f"The Task Chain: [{', '.join(chain)}]. "
        f"This is step {step_index + 1}/{len(chain)} of task, "
        f"{chain[step_index]}, and the atomic skill is {skill}"
    )


def _validate_annotation(annotation: dict[str, Any], episode_index: int) -> None:
    required_episode_fields = {"all_frames", "total_steps", "segments"}
    missing_episode_fields = required_episode_fields - annotation.keys()
    if missing_episode_fields:
        raise ValueError(
            f"Episode {episode_index} is missing annotation fields: {sorted(missing_episode_fields)}"
        )

    segments = annotation["segments"]
    if annotation["total_steps"] != len(segments):
        raise ValueError(f"Episode {episode_index} has an inconsistent total_steps value.")
    if not segments or segments[0]["start_frame"] != 0:
        raise ValueError(f"Episode {episode_index} does not start at frame 0.")
    if segments[-1]["end_frame"] != annotation["all_frames"]:
        raise ValueError(f"Episode {episode_index} does not cover its final frame.")

    required_segment_fields = {
        "action_name",
        "chain_of_thought",
        "primary_action_verb",
        "start_frame",
        "end_frame",
        "timestamp_formatted",
    }
    for segment_index, segment in enumerate(segments):
        missing_segment_fields = required_segment_fields - segment.keys()
        if missing_segment_fields:
            raise ValueError(
                f"Episode {episode_index} segment {segment_index} is missing fields: "
                f"{sorted(missing_segment_fields)}"
            )
        if segment["primary_action_verb"] not in VALID_SKILLS:
            raise ValueError(
                f"Episode {episode_index} contains unsupported skill "
                f"{segment['primary_action_verb']}."
            )
        if segment_index > 0 and segment["start_frame"] != segments[segment_index - 1]["end_frame"] + 1:
            raise ValueError(f"Episode {episode_index} has a gap before segment {segment_index}.")


def _make_annotation(record: dict[str, Any], semantic_frames: list[dict[str, Any]], args: Args) -> dict[str, Any]:
    phases = _detect_phases(semantic_frames, args)
    task_prompt = str(record["task_prompt"])
    chain = [_phase_description(phase) for phase in phases]
    segments = []
    for index, phase in enumerate(phases):
        skill = str(phase["skill"])
        if skill not in VALID_SKILLS:
            raise ValueError(f"Unsupported atomic skill: {skill}")
        chain_of_thought = _format_chain_of_thought(chain, index, skill)
        segment = {
            "action_name": task_prompt,
            "chain_of_thought": chain_of_thought,
            "primary_action_verb": skill,
            "start_frame": int(phase["start"]),
            "end_frame": int(phase["end"]),
            "timestamp_formatted": f"{phase['start']} - {phase['end']}",
            "object_name": phase["object_name"],
            "boundary_evidence": phase["evidence"],
            "confidence": float(phase["confidence"]),
        }
        if index + 1 < len(phases):
            segment["update_chain_of_thought"] = _format_chain_of_thought(
                chain,
                index + 1,
                str(phases[index + 1]["skill"]),
            )
        segments.append(segment)

    return {
        "all_frames": len(semantic_frames) - 1,
        "num_frames": len(semantic_frames),
        "total_steps": len(segments),
        "task_id": int(record["task_id"]),
        "task_description": str(record["task_description"]),
        "task_prompt": task_prompt,
        "demand_label": str(record["demand_label"]),
        "source_episode_index": record.get("source_episode_index"),
        "annotation_status": "needs_review",
        "segments": segments,
    }


def main(args: Args) -> None:
    dataset_root = (
        pathlib.Path(args.source_dir).expanduser().resolve()
        if args.source_dir
        else HF_LEROBOT_HOME / args.source_repo_id
    )
    metadata = _load_jsonl(dataset_root / args.metadata_filename)
    output_path = pathlib.Path(args.output_json)
    if args.append and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            annotations = json.load(f)
        if not isinstance(annotations, dict):
            raise ValueError("Existing annotation JSON must be indexed by episode id.")
    else:
        annotations = {}
    added_count = 0
    for record in metadata:
        episode_index = int(record["episode_index"])
        if args.append and str(episode_index) in annotations:
            logging.info("Skipping already annotated episode %s", episode_index)
            continue
        semantic_path = record.get("semantic_states_path")
        if not semantic_path:
            raise ValueError(
                f"Episode {episode_index} has no semantic_states_path. Recollect and reconvert with semantic states."
            )
        semantic_frames = _load_jsonl(dataset_root / str(semantic_path))
        if len(semantic_frames) != int(record["num_steps"]):
            raise ValueError(
                f"Episode {episode_index} semantic length mismatch: "
                f"{len(semantic_frames)} != {record['num_steps']}"
            )
        annotation = _make_annotation(record, semantic_frames, args)
        _validate_annotation(annotation, episode_index)
        annotations[str(episode_index)] = annotation
        added_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    logging.info(
        "Appended %s annotations; file now contains %s episodes: %s",
        added_count,
        len(annotations),
        output_path,
    )
    logging.info("Review segments with confidence below 0.7 before training.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
