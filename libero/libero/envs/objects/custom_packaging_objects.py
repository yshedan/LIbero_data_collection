import os
import pathlib

import numpy as np
from robosuite.models.objects import MujocoXMLObject

from libero.libero.envs.base_object import register_object


ASSET_ROOT = (
    pathlib.Path(__file__).parent.parent.parent / "assets" / "custom_objects" / "task5"
)


class PackagingObject(MujocoXMLObject):
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
class CardboardBox(PackagingObject):
    def __init__(self, name="cardboard_box", joints="free"):
        if joints == "free":
            joints = [dict(type="free", damping="5")]
        super().__init__(name, "cardboard_box", joints=joints)
        self.object_properties["articulation"] = {
            "default_open_ranges": [-2.70, -2.35],
            "default_close_ranges": [-0.025, 0.02],
        }

    @staticmethod
    def _scalar_qpos(qpos):
        qpos = np.asarray(qpos)
        if qpos.size != 1:
            return None
        return float(qpos)

    def is_open(self, qpos):
        qpos = self._scalar_qpos(qpos)
        if qpos is None:
            return False
        return qpos < max(self.object_properties["articulation"]["default_open_ranges"])

    def is_close(self, qpos):
        qpos = self._scalar_qpos(qpos)
        if qpos is None:
            return True
        return qpos > min(self.object_properties["articulation"]["default_close_ranges"])


@register_object
class CeramicGiftBox(PackagingObject):
    def __init__(self, name="ceramic_gift_box", joints="free"):
        if joints == "free":
            joints = [dict(type="free", damping="5")]
        super().__init__(name, "ceramic_gift_box", joints=joints)
        self.object_properties["articulation"] = {
            "default_open_ranges": [-2.70, -2.35],
            "default_close_ranges": [-0.025, 0.02],
        }

    @staticmethod
    def _scalar_qpos(qpos):
        qpos = np.asarray(qpos)
        if qpos.size != 1:
            return None
        return float(qpos)

    def is_open(self, qpos):
        qpos = self._scalar_qpos(qpos)
        if qpos is None:
            return False
        return qpos < max(self.object_properties["articulation"]["default_open_ranges"])

    def is_close(self, qpos):
        qpos = self._scalar_qpos(qpos)
        if qpos is None:
            return True
        return qpos > min(self.object_properties["articulation"]["default_close_ranges"])


@register_object
class CardboardLid(PackagingObject):
    def __init__(self, name="cardboard_lid", joints="free"):
        super().__init__(name, "cardboard_lid", joints=joints)


@register_object
class ClearGlassCup(PackagingObject):
    def __init__(self, name="clear_glass_cup", joints="free"):
        super().__init__(name, "clear_glass_cup", joints=joints)


@register_object
class CottonBall(PackagingObject):
    def __init__(self, name="cotton_ball", joints="free"):
        if joints == "free":
            joints = [dict(type="free", damping="0.25")]
        super().__init__(name, "cotton_ball", joints=joints)


@register_object
class CrumpledPaper(PackagingObject):
    def __init__(self, name="crumpled_paper", joints="free"):
        if joints == "free":
            joints = [dict(type="free", damping="0.18")]
        super().__init__(name, "crumpled_paper", joints=joints)


@register_object
class LargeCrumpledPaper(PackagingObject):
    def __init__(self, name="large_crumpled_paper", joints="free"):
        if joints == "free":
            joints = [dict(type="free", damping="0.22")]
        super().__init__(name, "large_crumpled_paper", joints=joints)


@register_object
class FoamBoardFit(PackagingObject):
    def __init__(self, name="foam_board_fit", joints="free"):
        super().__init__(name, "foam_board_fit", joints=joints)


@register_object
class FoamBoardSmall(PackagingObject):
    def __init__(self, name="foam_board_small", joints="free"):
        super().__init__(name, "foam_board_small", joints=joints)


@register_object
class FoamBoardLarge(PackagingObject):
    def __init__(self, name="foam_board_large", joints="free"):
        super().__init__(name, "foam_board_large", joints=joints)
