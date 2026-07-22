import os
import pathlib

from robosuite.models.objects import MujocoXMLObject

from libero.libero.envs.base_object import register_object


ASSET_ROOT = (
    pathlib.Path(__file__).parent.parent.parent / "assets" / "custom_objects" / "task6"
)


class OfficeObject(MujocoXMLObject):
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
        self.rotation = (0.0, 2.0 * 3.141592653589793)
        self.rotation_axis = "z"
        self.placement_quat_order = "xyzw"
        self.object_properties = {"vis_site_names": {}}


@register_object
class TapeRoll(OfficeObject):
    def __init__(self, name="tape_roll"):
        super().__init__(name, "tape_roll")
        self.rotation = (0.0, 0.0)


@register_object
class Eraser(OfficeObject):
    def __init__(self, name="eraser"):
        super().__init__(name, "eraser")


@register_object
class StickyNotePad(OfficeObject):
    def __init__(self, name="sticky_note_pad"):
        super().__init__(name, "sticky_note_pad")


@register_object
class BinderClip(OfficeObject):
    def __init__(self, name="binder_clip"):
        super().__init__(name, "binder_clip")
