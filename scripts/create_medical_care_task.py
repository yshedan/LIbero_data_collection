"""Generate the medical care table organization task."""

from pathlib import Path

import numpy as np

from libero.libero.utils.bddl_generation_utils import (
    get_xy_region_kwargs_list_from_regions_info,
)
from libero.libero.utils.mu_utils import InitialSceneTemplates, register_mu
from libero.libero.utils.task_generation_utils import (
    generate_bddl_from_task_info,
    register_task_info,
)


@register_mu(scene_type="general")
class MedicalCareTwoBoxScene(InitialSceneTemplates):
    def __init__(self):
        super().__init__(
            workspace_name="main_table",
            fixture_num_info={
                "table": 1,
                "care_box": 1,
                "instrument_box": 1,
            },
            object_num_info={
                "medicine_bottle": 1,
                "bandage_roll": 1,
                "medical_scissors": 1,
            },
        )

    def define_regions(self):
        placements = {
            # Fixed storage boxes, aligned along the right side of the table.
            "care_box_init_region": (0.15, 0.22, 0.0001, 0.0),
            "instrument_box_init_region": (-0.24, 0.22, 0.0001, 0.0),
        }
        for name, (x, y, half_len, yaw) in placements.items():
            self.regions.update(
                self.get_region_dict(
                    region_centroid_xy=[x, y],
                    region_name=name,
                    target_name=self.workspace_name,
                    region_half_len=half_len,
                    yaw_rotation=(yaw, yaw),
                )
            )
        # Loose medical items share several candidate zones instead of each
        # object being tied to its own small area. This gives each collection
        # episode a different spatial arrangement while keeping the objects in
        # the robot's comfortable reachable workspace and away from the fixed
        # storage boxes.
        self.regions["loose_object_init_region"] = {
            "target": self.workspace_name,
            "ranges": [
                (-0.16, -0.20, -0.04, -0.08),  # left-center-front
                (-0.03, -0.21, 0.09, -0.08),  # center-front
                (0.10, -0.19, 0.20, -0.07),  # right-center-front
                (-0.13, -0.06, -0.01, 0.05),  # left-center
                (0.02, -0.06, 0.14, 0.05),  # center
            ],
            "yaw_rotation": [(0.0, 0.0)],
        }
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        return [
            ("On", "care_box_1", "main_table_care_box_init_region"),
            ("On", "instrument_box_1", "main_table_instrument_box_init_region"),
            ("On", "medicine_bottle_1", "main_table_loose_object_init_region"),
            ("On", "bandage_roll_1", "main_table_loose_object_init_region"),
            ("On", "medical_scissors_1", "main_table_loose_object_init_region"),
        ]


def main():
    register_task_info(
        language=(
            "put the medicine bottle and the bandage in the care box, and "
            "put the scissors in the instrument box"
        ),
        scene_name="medical_care_two_box_scene",
        objects_of_interest=[
            "medicine_bottle_1",
            "bandage_roll_1",
            "medical_scissors_1",
            "care_box_1",
            "instrument_box_1",
        ],
        goal_states=[
            ("In", "medicine_bottle_1", "care_box_1_contain_region"),
            ("In", "bandage_roll_1", "care_box_1_contain_region"),
            ("In", "medical_scissors_1", "instrument_box_1_contain_region"),
        ],
    )

    output_dir = (
        Path(__file__).resolve().parents[1]
        / "libero/libero/bddl_files/custom_tasks"
    )
    files, failures = generate_bddl_from_task_info(folder=str(output_dir))
    if failures:
        raise RuntimeError(f"Failed to generate BDDL: {failures}")
    for path in files:
        print(path)


if __name__ == "__main__":
    main()
