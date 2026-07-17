import os
import pathlib

from robosuite.models.objects import MujocoXMLObject

from libero.libero.envs.base_object import register_object


ASSET_ROOT = (
    pathlib.Path(__file__).parent.parent.parent / "assets" / "custom_objects" / "task1"
)


class MedicalObject(MujocoXMLObject):
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
        # TableRegionSampler returns xyzw quaternions, while MuJoCo free
        # joints consume wxyz. BDDLBaseDomain converts this opt-in format.
        self.placement_quat_order = "xyzw"
        self.object_properties = {"vis_site_names": {}}


@register_object
class MedicineBottle(MedicalObject):
    def __init__(self, name="medicine_bottle"):
        super().__init__(name, "medicine_bottle")
        self.rotation = (0.0, 2.0 * 3.141592653589793)


@register_object
class BandageRoll(MedicalObject):
    def __init__(self, name="bandage_roll"):
        super().__init__(name, "bandage_roll")
        self.rotation = (0.0, 2.0 * 3.141592653589793)


@register_object
class MedicalScissors(MedicalObject):
    def __init__(self, name="medical_scissors"):
        super().__init__(name, "medical_scissors")
        self.rotation = (0.0, 2.0 * 3.141592653589793)


@register_object
class CareBox(MedicalObject):
    def __init__(self, name="care_box", joints="free"):
        if joints == "free":
            joints = [dict(type="free", damping="5")]
        super().__init__(name, "care_box", joints=joints)


@register_object
class InstrumentTray(MedicalObject):
    def __init__(self, name="instrument_tray", joints="free"):
        if joints == "free":
            joints = [dict(type="free", damping="5")]
        super().__init__(name, "instrument_tray", joints=joints)


@register_object
class InstrumentBox(MedicalObject):
    def __init__(self, name="instrument_box", joints="free"):
        if joints == "free":
            joints = [dict(type="free", damping="5")]
        super().__init__(name, "instrument_box", joints=joints)
