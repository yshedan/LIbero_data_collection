import re

from libero.libero.envs.base_object import OBJECTS_DICT, VISUAL_CHANGE_OBJECTS_DICT

from .hope_objects import *
from .google_scanned_objects import *
from .articulated_objects import *
from .turbosquid_objects import *
from .custom_medical_objects import *
from .custom_kitchen_objects import *
from .custom_assembly_objects import *
from .custom_hot_coffee_objects import *
from .custom_packaging_objects import *
from .custom_office_objects import *
from .custom_liquid_storage_objects import *
from .site_object import SiteObject
from .target_zones import *


def get_object_fn(category_name):
    return OBJECTS_DICT[category_name.lower()]


def get_object_dict():
    return OBJECTS_DICT
