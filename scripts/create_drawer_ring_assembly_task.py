"""Generate the drawer ring-pin assembly task."""

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
class DrawerRingAssemblyScene(InitialSceneTemplates):
    def __init__(self):
        super().__init__(
            workspace_name="main_table",
            fixture_num_info={
                "table": 1,
                "wooden_cabinet": 1,
                "assembly_board": 1,
            },
            object_num_info={
                "ring_pin": 1,
            },
        )

    def define_regions(self):
        # Keep both fixtures inside the reachable tabletop area. The cabinet
        # orientation follows LIBERO's public drawer tasks so the handle is
        # visible and pullable from agentview.
        self.regions["wooden_cabinet_init_region"] = {
            "target": self.workspace_name,
            "ranges": [
                (0.08, -0.31, 0.15, -0.23),
                (0.09, -0.23, 0.16, -0.16),
            ],
            "yaw_rotation": [(3.14159, 3.14159)],
        }
        self.regions["assembly_board_init_region"] = {
            "target": self.workspace_name,
            "ranges": [
                (-0.10, 0.17, -0.03, 0.25),
                (-0.09, 0.25, -0.02, 0.31),
            ],
            "yaw_rotation": [(0.0, 0.0)],
        }
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        return [
            ("On", "wooden_cabinet_1", "main_table_wooden_cabinet_init_region"),
            ("On", "assembly_board_1", "main_table_assembly_board_init_region"),
            ("On", "ring_pin_1", "assembly_board_1_top_region"),
        ]


def main():
    register_task_info(
        language=(
            "put the round sleeve from the assembly board into the top drawer "
            "and close the drawer"
        ),
        scene_name="drawer_ring_assembly_scene",
        objects_of_interest=[
            "wooden_cabinet_1",
            "ring_pin_1",
            "assembly_board_1",
        ],
        goal_states=[
            ("In", "ring_pin_1", "wooden_cabinet_1_top_region"),
            ("Close", "wooden_cabinet_1_top_region"),
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
