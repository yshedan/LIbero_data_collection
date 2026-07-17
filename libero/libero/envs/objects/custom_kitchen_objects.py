import os
import pathlib

from robosuite.models.objects import MujocoXMLObject

from libero.libero.envs.base_object import register_object


ASSET_ROOT = (
    pathlib.Path(__file__).parent.parent.parent / "assets" / "custom_objects" / "task2"
)


class KitchenObject(MujocoXMLObject):
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
class KitchenBowl(KitchenObject):
    def __init__(self, name="kitchen_bowl"):
        super().__init__(name, "kitchen_bowl")
        self.rotation = (0.0, 2.0 * 3.141592653589793)


@register_object
class KitchenSpoon(KitchenObject):
    def __init__(self, name="kitchen_spoon"):
        super().__init__(name, "kitchen_spoon")
        self.rotation = (0.0, 2.0 * 3.141592653589793)


@register_object
class KitchenKnife(KitchenObject):
    def __init__(self, name="kitchen_knife"):
        super().__init__(name, "kitchen_knife")
        self.rotation = (0.0, 2.0 * 3.141592653589793)


@register_object
class BowlCabinet(KitchenObject):
    def __init__(self, name="bowl_cabinet", joints="free"):
        if joints == "free":
            joints = [dict(type="free", damping="5")]
        super().__init__(name, "bowl_cabinet", joints=joints)


@register_object
class UtensilBox(KitchenObject):
    def __init__(self, name="utensil_box", joints="free"):
        if joints == "free":
            joints = [dict(type="free", damping="5")]
        super().__init__(name, "utensil_box", joints=joints)


@register_object
class KnifeRack(KitchenObject):
    def __init__(self, name="knife_rack", joints="free"):
        if joints == "free":
            joints = [dict(type="free", damping="5")]
        super().__init__(name, "knife_rack", joints=joints)
