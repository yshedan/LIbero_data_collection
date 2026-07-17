import os
import pathlib

from robosuite.models.objects import MujocoXMLObject

from libero.libero.envs.base_object import register_object


ASSET_ROOT = (
    pathlib.Path(__file__).parent.parent.parent / "assets" / "custom_objects" / "task3"
)


class AssemblyObject(MujocoXMLObject):
    def __init__(self, name, xml_name, joints="free"):
        if joints == "free":
            joints = [dict(type="free", damping="0.05")]
        super().__init__(
            os.fspath(ASSET_ROOT / f"{xml_name}.xml"),
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = xml_name
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        self.placement_quat_order = "xyzw"
        self.object_properties = {"vis_site_names": {}}


@register_object
class RingPin(AssemblyObject):
    def __init__(self, name="ring_pin"):
        super().__init__(name, "ring_pin")
        self.rotation = (0.0, 2.0 * 3.141592653589793)


@register_object
class HexRingPin(AssemblyObject):
    def __init__(self, name="hex_ring_pin"):
        super().__init__(name, "hex_ring_pin")
        self.rotation = (0.0, 2.0 * 3.141592653589793)


@register_object
class AssemblyBoard(AssemblyObject):
    def __init__(self, name="assembly_board", joints="free"):
        if joints == "free":
            joints = [dict(type="free", damping="5")]
        super().__init__(name, "assembly_board", joints=joints)
