import os
import pathlib

from robosuite.models.objects import MujocoXMLObject

from libero.libero.envs.base_object import register_object


ASSET_ROOT = (
    pathlib.Path(__file__).parent.parent.parent / "assets" / "custom_objects" / "task7"
)


class LiquidStorageObject(MujocoXMLObject):
    def __init__(self, name, xml_name, joints="free"):
        if joints == "free":
            joints = [dict(type="free", damping="0.12")]
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
class LiquidBottle(LiquidStorageObject):
    def __init__(self, name="liquid_bottle", joints="free"):
        super().__init__(name, "liquid_bottle", joints=joints)


@register_object
class BottleCap(LiquidStorageObject):
    def __init__(self, name="bottle_cap", joints="free"):
        if joints == "free":
            joints = [dict(type="free", damping="1.20")]
        super().__init__(name, "bottle_cap", joints=joints)


@register_object
class BottleStorageRack(LiquidStorageObject):
    def __init__(self, name="bottle_storage_rack", joints="free"):
        if joints == "free":
            joints = [dict(type="free", damping="4.0")]
        super().__init__(name, "bottle_storage_rack", joints=joints)
