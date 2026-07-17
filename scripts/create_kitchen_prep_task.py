"""Generate the kitchen prep counter organization task."""

from pathlib import Path

from libero.libero.utils.bddl_generation_utils import (
    get_xy_region_kwargs_list_from_regions_info,
)
from libero.libero.utils.mu_utils import InitialSceneTemplates, register_mu
from libero.libero.utils.task_generation_utils import (
    generate_bddl_from_task_info,
    register_task_info,
)


@register_mu(scene_type="general")
class KitchenPrepCounterScene(InitialSceneTemplates):
    def __init__(self):
        super().__init__(
            workspace_name="main_table",
            fixture_num_info={
                "table": 1,
                "bowl_cabinet": 1,
                "knife_rack": 1,
            },
            object_num_info={
                "white_bowl": 1,
                "kitchen_spoon": 1,
                "kitchen_knife": 1,
            },
        )

    def define_regions(self):
        placements = {
            # Fixed storage places: the single-layer divided cabinet is on
            # the front side and opens upward; the knife rack is on the back
            # side.
            "bowl_cabinet_init_region": (0.0, -0.27, 0.0001, 3.14159),
            "knife_rack_init_region": (0.0, 0.25, 0.0001, 0.0),
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

        # Randomized but reachable middle-table zones. All loose objects share
        # this region so their relative order can change between episodes.
        self.regions["loose_object_init_region"] = {
            "target": self.workspace_name,
            "ranges": [
                (-0.24, -0.02, -0.10, 0.06),
                (-0.07, -0.02, 0.07, 0.06),
                (0.10, -0.02, 0.24, 0.06),
                (-0.24, 0.08, -0.10, 0.16),
                (-0.07, 0.08, 0.07, 0.16),
                (0.10, 0.08, 0.24, 0.16),
            ],
            "yaw_rotation": [(0.0, 0.0)],
        }
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        return [
            ("On", "bowl_cabinet_1", "main_table_bowl_cabinet_init_region"),
            ("On", "knife_rack_1", "main_table_knife_rack_init_region"),
            ("On", "white_bowl_1", "main_table_loose_object_init_region"),
            ("In", "kitchen_spoon_1", "bowl_cabinet_1_utensil_contain_region"),
            ("On", "kitchen_knife_1", "main_table_loose_object_init_region"),
        ]


def main():
    register_task_info(
        language=(
            "put the bowl in the divided cabinet and put the knife in the "
            "knife rack, with the spoon already stored in the cabinet"
        ),
        scene_name="kitchen_prep_counter_scene",
        objects_of_interest=[
            "white_bowl_1",
            "kitchen_spoon_1",
            "kitchen_knife_1",
            "bowl_cabinet_1",
            "knife_rack_1",
        ],
        goal_states=[
            ("In", "white_bowl_1", "bowl_cabinet_1_contain_region"),
            ("In", "kitchen_knife_1", "knife_rack_1_contain_region"),
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
