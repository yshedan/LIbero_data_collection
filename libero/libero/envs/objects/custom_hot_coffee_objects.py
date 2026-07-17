import os
import pathlib

import numpy as np
from robosuite.models.objects import MujocoXMLObject

from libero.libero.envs.base_object import (
    register_object,
    register_visual_change_object,
)


ASSET_ROOT = (
    pathlib.Path(__file__).parent.parent.parent / "assets" / "custom_objects" / "task4"
)


class HotCoffeeObject(MujocoXMLObject):
    def __init__(self, name, xml_name, joints="free"):
        if joints == "free":
            joints = [dict(type="free", damping="5")]
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
@register_visual_change_object
class HeatingPad(HotCoffeeObject):
    def __init__(self, name="heating_pad", joints="free"):
        super().__init__(name, "heating_pad", joints=joints)
        self.object_properties["articulation"] = {
            "default_turnon_ranges": [-0.002, 0.0],
            "default_turnoff_ranges": [-0.012, -0.009],
        }
        self._power_on = True
        self._button_armed = True
        self._press_threshold = -0.0015
        self._release_threshold = -0.0002
        self._toggle_cooldown = 15
        self._cooldown_remaining = 0
        self.object_properties["vis_site_names"]["indicator"] = (
            self.naming_prefix + "indicator",
            True,
        )

    def _update_power_state(self, qpos):
        qpos = np.asarray(qpos)
        if qpos.size != 1:
            return None
        qpos = float(qpos)
        is_pressed = qpos <= self._press_threshold
        is_released = qpos >= self._release_threshold

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        if is_pressed and self._button_armed and self._cooldown_remaining == 0:
            self._power_on = not self._power_on
            self._button_armed = False
            self._cooldown_remaining = self._toggle_cooldown
        elif is_released and self._cooldown_remaining == 0:
            self._button_armed = True

        self.object_properties["vis_site_names"]["indicator"] = (
            self.naming_prefix + "indicator",
            self._power_on,
        )
        return self._power_on

    def turn_on(self, qpos):
        self._update_power_state(qpos)
        return self._power_on

    def turn_off(self, qpos):
        self.object_properties["vis_site_names"]["indicator"] = (
            self.naming_prefix + "indicator",
            self._power_on,
        )
        return not self._power_on


@register_object
class CoasterPad(HotCoffeeObject):
    def __init__(self, name="coaster_pad", joints="free"):
        super().__init__(name, "coaster_pad", joints=joints)


@register_object
class RedCoffeeMugWithCoffee(HotCoffeeObject):
    def __init__(self, name="red_coffee_mug_with_coffee", joints="free"):
        super().__init__(name, "red_coffee_mug_with_coffee", joints=joints)
        self.rotation = (-np.pi / 2, -np.pi / 2)
        self.rotation_axis = "x"
        self.placement_quat_order = "wxyz"


@register_object
class SmallWoodenTray(HotCoffeeObject):
    def __init__(self, name="small_wooden_tray", joints="free"):
        super().__init__(name, "small_wooden_tray", joints=joints)
