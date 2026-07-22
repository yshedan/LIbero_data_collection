import argparse
import os
import sys
from pathlib import Path
import datetime
import h5py
import init_path
import json
import numpy as np

# Data collection uses a desktop OpenCV window. Avoid inheriting headless or
# CoppeliaSim-specific rendering settings from the user's shell.
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("PYOPENGL_PLATFORM", None)
os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)
os.environ.setdefault("MUJOCO_GL", "glx")

REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"
for dependency_path in (
    THIRD_PARTY_ROOT / "joycon-robotics",
    THIRD_PARTY_ROOT / "meta_quest_teleop",
):
    if dependency_path.exists():
        sys.path.insert(0, str(dependency_path))

import cv2
import robosuite as suite
import time
from glob import glob
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.utils.input_utils import input2action
from robosuite.utils import transform_utils as T


import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *


FULLY_OPEN_GRIPPER_THRESHOLD = 0.039
COLLECTION_WINDOW_POSITION = (320, 220)


def move_collection_window():
    """Place the main collection window lower on the screen."""
    try:
        cv2.moveWindow(
            "offscreen render",
            COLLECTION_WINDOW_POSITION[0],
            COLLECTION_WINDOW_POSITION[1],
        )
    except cv2.error:
        pass


class JoyConRobosuiteDevice:
    """Adapter from joycon-robotics to robosuite's input2action interface."""

    def __init__(
        self,
        device="right",
        pos_sensitivity=1.0,
        rot_sensitivity=1.0,
        pos_scale=0.2,
        rot_scale=0.2,
        disable_rotation=False,
    ):
        try:
            from joyconrobotics import JoyconRobotics
        except ImportError as exc:
            raise ImportError(
                "Unable to import joyconrobotics. Install it in the libero "
                "environment with: pip install -e third_party/joycon-robotics"
            ) from exc

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.disable_rotation = disable_rotation
        self._reset_state = 0
        self._enabled = False
        self._last_pose = None
        self.rotation = np.array(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
        )

        # Use gripper values that match robosuite's expectation:
        # 0 / False -> open, 1 / True -> closed.
        self.controller = JoyconRobotics(
            device=device,
            gripper_open=0.0,
            gripper_close=1.0,
            gripper_state=0.0,
            pitch_down_double=False,
            offset_euler_rad=[0.0, -np.pi, 0.0],
            euler_reverse=[1, -1, 1],
            direction_reverse=[-1, 1, 1],
            pure_z=True,
            without_rest_init=False,
        )
        self._display_controls(device)

    @staticmethod
    def _display_controls(device):
        print("")
        print(f"JoyCon device: {device}")
        print("Control                 Command")
        print("Joystick up/down        move forward/backward")
        print("Joystick left/right     move left/right")
        print("R                       move up")
        print("Stick press             move down")
        print("ZR                      toggle gripper")
        print("Home                    return JoyCon virtual pose to origin")
        print("+                       recalibrate JoyCon orientation")
        print("Y                       discard current rollout and reset")
        print("")

    def start_control(self):
        """Reset the adapter state before a new rollout starts."""
        self._reset_state = 0
        self._enabled = True
        pose, _, _ = self.controller.get_control()
        self._last_pose = np.asarray(pose, dtype=np.float32)

    def get_controller_state(self):
        """Return robosuite-compatible device state."""
        pose, gripper, button_control = self.controller.get_control()
        pose = np.asarray(pose, dtype=np.float32)
        if self._last_pose is None:
            self._last_pose = pose.copy()

        delta = pose - self._last_pose
        self._last_pose = pose.copy()

        dpos = delta[:3] * self.pos_sensitivity * self.pos_scale
        raw_drotation = delta[3:6]
        raw_drotation = (raw_drotation + np.pi) % (2 * np.pi) - np.pi
        raw_drotation = raw_drotation * self.rot_sensitivity * self.rot_scale
        if self.disable_rotation:
            raw_drotation[:] = 0.0

        if not self.disable_rotation:
            self.rotation = self.rotation.dot(T.euler2mat(raw_drotation))

        # joycon-robotics uses A=1 for "next episode" and Y=-1 for
        # "restart episode". In LIBERO collection, only restart maps cleanly to
        # robosuite's reset signal; normal saving should still be driven by the
        # task success predicate.
        if button_control == -1:
            self._reset_state = 1

        return {
            "dpos": dpos,
            "rotation": self.rotation,
            "raw_drotation": raw_drotation,
            "grasp": bool(gripper >= 0.5),
            "reset": self._reset_state,
        }

    def close(self):
        disconnect = getattr(self.controller, "disconnnect", None)
        if disconnect is not None:
            disconnect()


class XRRobosuiteDevice:
    """Adapter from XRoboToolkit SDK controller poses to robosuite input2action."""

    R_HEADSET_TO_WORLD = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        controller="right",
        source="udp",
        udp_host="127.0.0.1",
        udp_port=50505,
        pos_sensitivity=1.0,
        rot_sensitivity=1.0,
        pos_scale=2.0,
        rot_scale=0.12,
        active_threshold=0.8,
        gripper_threshold=0.5,
        disable_rotation=False,
    ):
        if controller not in ("left", "right"):
            raise ValueError("XR controller must be 'left' or 'right'")
        if source not in ("udp", "direct"):
            raise ValueError("XR source must be 'udp' or 'direct'")

        self.source = source
        self.xrt = None
        self.socket = None
        self.latest_packet = None
        self.controller = controller
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.active_threshold = active_threshold
        self.gripper_threshold = gripper_threshold
        self.disable_rotation = disable_rotation
        self._last_pos = None
        self._last_rot = None
        self._reset_state = 0
        self.rotation = np.array(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float32,
        )

        if self.source == "direct":
            try:
                import xrobotoolkit_sdk as xrt
            except ImportError as exc:
                raise ImportError(
                    "Unable to import xrobotoolkit_sdk. The XR SDK requires "
                    "Python >= 3.10, while the LIBERO environment is often "
                    "Python 3.8. Use --xr-source udp with scripts/xr_teleop_bridge.py."
                ) from exc
            self.xrt = xrt
            self.xrt.init()
        else:
            import socket

            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind((udp_host, udp_port))
            self.socket.setblocking(False)
            print(f"Waiting for XR UDP packets on {udp_host}:{udp_port}")

        self._display_controls(controller, source)

    @staticmethod
    def _display_controls(controller, source):
        print("")
        print(f"XR controller: {controller} ({source})")
        print("Control                 Command")
        print("Grip                    hold to enable arm motion")
        print("Release Grip            clutch/recenter your hand without moving robot")
        print("Controller motion       move / rotate end effector")
        print("Trigger                 close gripper; release to open")
        print("Y / B                   discard current rollout and reset")
        print("")

    @staticmethod
    def _quat_xyzw_to_mat(quat_xyzw):
        x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64)
        norm = np.linalg.norm([x, y, z, w])
        if norm < 1e-8:
            return np.eye(3, dtype=np.float32)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _rotvec_from_mat(rot_mat):
        trace = np.trace(rot_mat)
        angle = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
        if angle < 1e-6:
            return np.zeros(3, dtype=np.float32)
        denom = 2.0 * np.sin(angle)
        axis = np.array(
            [
                rot_mat[2, 1] - rot_mat[1, 2],
                rot_mat[0, 2] - rot_mat[2, 0],
                rot_mat[1, 0] - rot_mat[0, 1],
            ],
            dtype=np.float32,
        ) / denom
        return axis * angle

    def _receive_udp_packet(self):
        if self.source != "udp":
            return
        while True:
            try:
                data, _ = self.socket.recvfrom(65536)
            except BlockingIOError:
                break
            try:
                self.latest_packet = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

    def _get_pose(self):
        if self.source == "udp":
            self._receive_udp_packet()
            if self.latest_packet is None:
                pose = np.zeros(7, dtype=np.float32)
                pose[6] = 1.0
            else:
                pose = self.latest_packet.get("pose", [0, 0, 0, 0, 0, 0, 1])
        else:
            if self.controller == "left":
                pose = self.xrt.get_left_controller_pose()
            else:
                pose = self.xrt.get_right_controller_pose()
        pose = np.asarray(pose, dtype=np.float32)
        pos = self.R_HEADSET_TO_WORLD.dot(pose[:3])
        rot = self.R_HEADSET_TO_WORLD.dot(self._quat_xyzw_to_mat(pose[3:7]))
        return pos, rot

    def _get_trigger(self):
        if self.source == "udp":
            return float((self.latest_packet or {}).get("trigger", 0.0))
        if self.controller == "left":
            return float(self.xrt.get_left_trigger())
        return float(self.xrt.get_right_trigger())

    def _get_grip(self):
        if self.source == "udp":
            return float((self.latest_packet or {}).get("grip", 0.0))
        if self.controller == "left":
            return float(self.xrt.get_left_grip())
        return float(self.xrt.get_right_grip())

    def _get_reset_pressed(self):
        if self.source == "udp":
            buttons = (self.latest_packet or {}).get("buttons", {})
            return bool(buttons.get("Y", False) or buttons.get("B", False))
        return bool(self.xrt.get_Y_button() or self.xrt.get_B_button())

    def start_control(self):
        self._reset_state = 0
        self._last_pos = None
        self._last_rot = None

    def get_controller_state(self):
        pos, rot = self._get_pose()
        active = self._get_grip() >= self.active_threshold

        if self._last_pos is None or self._last_rot is None or not active:
            self._last_pos = pos.copy()
            self._last_rot = rot.copy()
            dpos = np.zeros(3, dtype=np.float32)
            raw_drotation = np.zeros(3, dtype=np.float32)
        else:
            dpos = (pos - self._last_pos) * self.pos_sensitivity * self.pos_scale
            delta_rot_mat = rot.dot(self._last_rot.T)
            raw_drotation = self._rotvec_from_mat(delta_rot_mat)
            raw_drotation *= self.rot_sensitivity * self.rot_scale
            self._last_pos = pos.copy()
            self._last_rot = rot.copy()

        if self.disable_rotation:
            raw_drotation[:] = 0.0
        else:
            self.rotation = self.rotation.dot(T.euler2mat(raw_drotation))

        if self._get_reset_pressed():
            self._reset_state = 1

        return {
            "dpos": dpos,
            "rotation": self.rotation,
            "raw_drotation": raw_drotation,
            "grasp": self._get_trigger() >= self.gripper_threshold,
            "reset": self._reset_state,
        }

    def close(self):
        if self.socket is not None:
            self.socket.close()
        if self.xrt is not None:
            close = getattr(self.xrt, "close", None)
            if close is not None:
                close()


class MetaQuestRobosuiteDevice:
    """Adapter from meta_quest_teleop controller poses to robosuite input2action."""

    def __init__(
        self,
        controller="right",
        ip_address=None,
        apk_name="com.rail.oculus.teleop",
        pos_sensitivity=1.0,
        rot_sensitivity=1.0,
        pos_scale=0.6,
        rot_scale=0.02,
        active_threshold=0.5,
        gripper_threshold=0.5,
        disable_rotation=False,
    ):
        if controller not in ("left", "right"):
            raise ValueError("Meta Quest controller must be 'left' or 'right'")

        try:
            from meta_quest_teleop.reader import MetaQuestReader
        except ImportError as exc:
            raise ImportError(
                "Unable to import meta_quest_teleop. Install it in the libero "
                "environment with: pip install -e third_party/meta_quest_teleop"
            ) from exc

        self.controller = controller
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.active_threshold = active_threshold
        self.gripper_threshold = gripper_threshold
        self.disable_rotation = disable_rotation
        self._last_pos = None
        self._last_rot = None
        self._reset_state = 0
        self._printed_waiting = False
        self.joystick_deadzone = 0.12
        self.rotation = np.array(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float32,
        )

        self.apk_name = apk_name
        self.reader = MetaQuestReader(
            ip_address=ip_address,
            APK_name=apk_name,
            run=False,
        )
        self._start_reader_thread()
        self._display_controls(controller, ip_address)

    def _start_reader_thread(self):
        import threading

        self.reader.running = True
        self.reader.device.shell(
            f'am start -n "{self.apk_name}/{self.apk_name}.MainActivity" '
            "-a android.intent.action.MAIN -c android.intent.category.LAUNCHER"
        )
        self.reader.thread = threading.Thread(
            target=self.reader.device.shell,
            args=("logcat -T 0", self.reader.read_logcat_by_line),
            daemon=True,
        )
        self.reader.thread.start()

    @staticmethod
    def _display_controls(controller, ip_address):
        connection = "USB/ADB" if ip_address is None else f"ADB TCP {ip_address}"
        print("")
        print(f"Meta Quest controller: {controller} ({connection})")
        print("Control                 Command")
        print("Grip                    hold to enable arm motion")
        print("Controller translation  move end effector")
        print("Joystick left/right     yaw end effector")
        print("Joystick up/down        pitch end effector")
        print("Hold A/X + joystick LR  roll end effector")
        print("Trigger                 close gripper; release to open")
        print("B / Y                   discard current rollout and reset")
        print("")

    def _get_pose(self):
        transform = self.reader.get_hand_controller_transform_ros(self.controller)
        if transform is None:
            if not self._printed_waiting:
                print(
                    "Waiting for Meta Quest controller data. "
                    "Put on the headset, open the teleop app, and move the controller."
                )
                self._printed_waiting = True
            return None, None

        self._printed_waiting = False
        pos = np.asarray(transform[:3, 3], dtype=np.float32)
        rot = np.asarray(transform[:3, :3], dtype=np.float32)
        # Meta Quest ROS pose uses x=forward, y=left, z=up. In this LIBERO
        # collection setup, robosuite's horizontal action axes are ordered as
        # left/right then forward/back, so swap the table-plane axes here.
        pos = pos[[1, 0, 2]] * np.array([-1.0, -1.0, 1.0], dtype=np.float32)
        return pos, rot

    def _get_grip(self):
        try:
            return float(self.reader.get_grip_value(self.controller))
        except Exception:
            return 0.0

    def _get_trigger(self):
        try:
            return float(self.reader.get_trigger_value(self.controller))
        except Exception:
            return 0.0

    def _get_joystick(self):
        try:
            joystick = self.reader.get_joystick_value(self.controller)
        except Exception:
            return np.zeros(2, dtype=np.float32)
        joystick = np.asarray(joystick[:2], dtype=np.float32)
        joystick[np.abs(joystick) < self.joystick_deadzone] = 0.0
        return joystick

    def _get_roll_mode_pressed(self):
        return bool(
            self.reader.get_button_state("A")
            or self.reader.get_button_state("X")
        )

    def _joystick_rotation_delta(self):
        joystick_x, joystick_y = self._get_joystick()
        roll_mode = self._get_roll_mode_pressed()

        roll = joystick_x if roll_mode else 0.0
        pitch = joystick_y
        yaw = 0.0 if roll_mode else joystick_x

        return (
            np.array([roll, pitch, yaw], dtype=np.float32)
            * self.rot_sensitivity
            * self.rot_scale
        )

    def _get_reset_pressed(self):
        return bool(
            self.reader.get_button_state("B")
            or self.reader.get_button_state("Y")
            or self.reader.get_button_state("RJ")
        )

    def start_control(self):
        self._reset_state = 0
        self._last_pos = None
        self._last_rot = None

    def get_controller_state(self):
        pos, rot = self._get_pose()
        active = self._get_grip() >= self.active_threshold

        if pos is None or rot is None:
            dpos = np.zeros(3, dtype=np.float32)
            raw_drotation = np.zeros(3, dtype=np.float32)
        elif self._last_pos is None or self._last_rot is None or not active:
            self._last_pos = pos.copy()
            self._last_rot = rot.copy()
            dpos = np.zeros(3, dtype=np.float32)
            raw_drotation = np.zeros(3, dtype=np.float32)
        else:
            dpos = (pos - self._last_pos) * self.pos_sensitivity * self.pos_scale
            raw_drotation = self._joystick_rotation_delta()
            self._last_pos = pos.copy()
            self._last_rot = rot.copy()

        if self.disable_rotation:
            raw_drotation[:] = 0.0
        else:
            self.rotation = self.rotation.dot(T.euler2mat(raw_drotation))

        if self._get_reset_pressed():
            self._reset_state = 1

        return {
            "dpos": dpos,
            "rotation": self.rotation,
            "raw_drotation": raw_drotation,
            "grasp": self._get_trigger() >= self.gripper_threshold,
            "reset": self._reset_state,
        }

    def close(self):
        self.reader.running = False
        try:
            self.reader.device.shell(f"am force-stop {self.apk_name}")
        except Exception:
            pass
        thread = getattr(self.reader, "thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)


def format_task_directory(task_index):
    """Return a stable task subdirectory name, e.g. task1."""
    if task_index is None:
        return None
    text = str(task_index).strip()
    if not text:
        return None
    if text.startswith("task"):
        task_name = text
    else:
        task_name = f"task{text}"
    return "".join(
        character if character.isalnum() or character in ("_", "-") else "_"
        for character in task_name
    )


def infer_task_directory_from_bddl(bddl_file):
    """Infer a stable task folder from the first two words of a BDDL filename."""
    if not bddl_file:
        return None
    stem = os.path.splitext(os.path.basename(bddl_file))[0]
    words = [word.lower() for word in stem.split("_") if word]
    if len(words) < 2:
        return None
    return "_".join(words[:2])


def resolve_output_root(directory, task_index, bddl_file):
    """Resolve the root directory for this collection run."""
    output_root = directory
    task_directory = format_task_directory(task_index)
    if task_directory:
        return os.path.join(output_root, task_directory)

    # When collecting custom tasks into the common datasets0 folder, split
    # runs into BDDL-derived task folders such as medical_care or drawer_ring.
    if os.path.basename(os.path.normpath(output_root)) == "datasets0":
        inferred_task_directory = infer_task_directory_from_bddl(bddl_file)
        if inferred_task_directory:
            return os.path.join(output_root, inferred_task_directory)

    return output_root


def smooth_action_labels(actions, window_size=5, motion_dims=6):
    """Smooth continuous arm action labels while preserving gripper commands."""
    actions = np.asarray(actions)
    if window_size <= 1:
        return actions
    if window_size % 2 == 0:
        raise ValueError("--smooth-window-size must be odd")
    if actions.ndim != 2:
        raise ValueError(f"Expected actions with shape [T, D], got {actions.shape}")

    smoothed = actions.astype(np.float32, copy=True)
    dims_to_smooth = min(motion_dims, smoothed.shape[1])
    if dims_to_smooth <= 0:
        return smoothed.astype(actions.dtype, copy=False)

    radius = window_size // 2
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    motion = smoothed[:, :dims_to_smooth]
    padded = np.pad(motion, ((radius, radius), (0, 0)), mode="edge")
    for dim in range(dims_to_smooth):
        smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return np.clip(smoothed, -1.0, 1.0).astype(actions.dtype, copy=False)


def is_gripper_open(env, threshold=FULLY_OPEN_GRIPPER_THRESHOLD):
    """Return True only when every Panda gripper joint is open enough."""
    try:
        obs = env._get_observations(force_update=True)
        gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    if gripper_qpos.size == 0:
        return False
    return bool(np.all(np.abs(gripper_qpos) >= threshold))


def get_eef_rotation(active_robot):
    """Return the current end-effector rotation matrix."""
    controller = getattr(active_robot, "controller", None)
    ee_ori_mat = getattr(controller, "ee_ori_mat", None)
    if ee_ori_mat is not None:
        return np.asarray(ee_ori_mat, dtype=np.float32)
    return np.asarray(active_robot._hand_orn, dtype=np.float32)


def transform_keyboard_action(
    action,
    sim,
    camera_name,
    active_robot,
    *,
    control_frame,
    rotation_frame,
):
    """Map keyboard translation and rotation commands into controller axes."""
    if action is None or len(action) < 6:
        return action
    if control_frame == "libero" and rotation_frame == "libero":
        return action

    if control_frame == "camera" or rotation_frame == "camera":
        camera_id = sim.model.camera_name2id(camera_name)
        camera_rotation = sim.data.cam_xmat[camera_id].reshape(3, 3)
        camera_right = camera_rotation[:, 0]
        camera_up = camera_rotation[:, 1]
        camera_forward = -camera_rotation[:, 2]

    original_position = action[:3].copy()
    original_rotation = action[3:6].copy()

    if control_frame == "camera":
        # Keep horizontal motion parallel to the table. W/S follow the vertical
        # direction in the image, while A/D follow the horizontal direction.
        screen_right = camera_right.copy()
        screen_up = camera_up.copy()
        screen_right[2] = 0.0
        screen_up[2] = 0.0
        screen_right /= np.linalg.norm(screen_right)
        screen_up /= np.linalg.norm(screen_up)

        action[:3] = (
            original_position[1] * screen_right
            - original_position[0] * screen_up
            + original_position[2] * np.array([0.0, 0.0, 1.0])
        )
    elif control_frame == "eef":
        eef_rotation = get_eef_rotation(active_robot)
        # Keep the keyboard semantics stable while using the gripper as the
        # horizontal reference: A/D moves left-right, W/S moves forward-back,
        # and R/F always moves vertically. The Panda wrist frame has its x/y
        # axes swapped relative to the keyboard convention in common grasping
        # poses, so map the keys explicitly instead of applying the rotation
        # matrix directly.
        eef_right = -eef_rotation[:, 0].copy()
        eef_forward = -eef_rotation[:, 1].copy()
        eef_right[2] = 0.0
        eef_forward[2] = 0.0
        right_norm = np.linalg.norm(eef_right)
        forward_norm = np.linalg.norm(eef_forward)
        if right_norm < 1e-6 or forward_norm < 1e-6:
            action[:3] = -original_position
        else:
            eef_right /= right_norm
            eef_forward /= forward_norm
            action[:3] = (
                original_position[1] * eef_right
                + original_position[0] * eef_forward
                + original_position[2] * np.array([0.0, 0.0, 1.0])
            )
    elif control_frame == "world":
        action[:3] = original_position
    elif control_frame == "libero":
        action[:3] = original_position
    else:
        raise ValueError(f"Unsupported control frame: {control_frame}")

    if rotation_frame == "camera":
        action[3:6] = (
            original_rotation[0] * camera_right
            + original_rotation[1] * camera_up
            + original_rotation[2] * camera_forward
        )
    elif rotation_frame == "eef":
        eef_rotation = get_eef_rotation(active_robot)
        action[3:6] = eef_rotation @ original_rotation
    elif rotation_frame == "world":
        action[3:6] = original_rotation
    elif rotation_frame == "libero":
        action[3:6] = original_rotation
    else:
        raise ValueError(f"Unsupported rotation frame: {rotation_frame}")
    return action


def get_movable_object_names(env):
    """Return BDDL movable object instance names, excluding fixtures."""
    object_names = []
    for names in env.parsed_problem.get("objects", {}).values():
        object_names.extend(names)
    return [name for name in object_names if name in getattr(env, "obj_body_id", {})]


def camera_exists(sim, camera_name):
    """Return whether a named camera exists in the compiled MuJoCo model."""
    return camera_name in getattr(sim.model, "camera_names", ())


def set_camera_lookat(sim, camera_name, position, target):
    """Move a fixed camera and orient it so its optical axis points at target."""
    if not camera_exists(sim, camera_name):
        return False

    position = np.asarray(position, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    forward = target - position
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        return False
    forward /= forward_norm

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
    right /= right_norm
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    # MuJoCo cameras look along local -Z, with local Y as image up.
    rotation = np.column_stack([right, up, -forward])
    quat_xyzw = T.mat2quat(rotation)
    quat_wxyz = T.convert_quat(quat_xyzw, to="wxyz")

    camera_id = sim.model.camera_name2id(camera_name)
    sim.model.cam_pos[camera_id] = position
    sim.model.cam_quat[camera_id] = quat_wxyz
    sim.forward()
    return True


def configure_table_level_front_camera(
    env,
    camera_name="frontview",
    position=(0.90, 0.0, 1.08),
    target=(0.0, 0.0, 0.95),
):
    """Configure a raised front camera for live collection assistance."""
    return set_camera_lookat(env.sim, camera_name, position, target)


def render_camera_window(
    env,
    camera_name,
    window_name,
    *,
    width=320,
    height=240,
    label=None,
    window_position=None,
):
    """Render one named camera to a live OpenCV helper window."""
    if not camera_exists(env.sim, camera_name):
        return False
    try:
        image = env.sim.render(camera_name=camera_name, width=width, height=height)
    except Exception:
        return False
    image = np.flip(image, axis=0)
    image = image[..., ::-1].copy()
    if label:
        cv2.putText(
            image,
            label,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    cv2.imshow(window_name, image)
    if window_position is not None:
        cv2.moveWindow(window_name, window_position[0], window_position[1])
    return True


def render_auxiliary_views(
    env,
    *,
    enabled=True,
    wrist_camera="robot0_eye_in_hand",
    table_front_camera="frontview",
    width=320,
    height=240,
):
    """Show auxiliary live views that are not recorded into the dataset."""
    if not enabled:
        return
    wrist_ok = render_camera_window(
        env,
        wrist_camera,
        "wrist view",
        width=width,
        height=height,
        label="wrist",
        window_position=(20, 40),
    )
    front_ok = render_camera_window(
        env,
        table_front_camera,
        "table-level front view",
        width=width,
        height=height,
        label="table front",
        window_position=(40 + width, 40),
    )
    if not wrist_ok:
        pass
    if not front_ok:
        pass


def project_world_to_viewer_pixel(sim, camera_name, point, width, height):
    """Project a world point to the OpenCV viewer pixel coordinates."""
    projected = project_world_to_viewer_pixel_depth(
        sim, camera_name, point, width, height
    )
    if projected is None:
        return None
    return projected[:2]


def project_world_to_viewer_pixel_depth(sim, camera_name, point, width, height):
    """Project a world point to OpenCV viewer pixel coordinates and camera depth."""
    camera_id = sim.model.camera_name2id(camera_name)
    camera_pos = sim.data.cam_xpos[camera_id]
    camera_rot = sim.data.cam_xmat[camera_id].reshape(3, 3)
    point_camera = camera_rot.T @ (np.asarray(point) - camera_pos)
    depth = -point_camera[2]
    if depth <= 1e-6:
        return None

    fovy = np.deg2rad(float(sim.model.cam_fovy[camera_id]))
    focal = 0.5 * height / np.tan(0.5 * fovy)
    x = int(round(width * 0.5 + focal * point_camera[0] / depth))
    y = int(round(height * 0.5 - focal * point_camera[1] / depth))
    if x < -width or x > 2 * width or y < -height or y > 2 * height:
        return None
    return x, y, depth


def depth_buffer_to_metric(sim, depth_buffer):
    """Convert MuJoCo normalized depth buffer to metric camera depth."""
    extent = sim.model.stat.extent
    far = sim.model.vis.map.zfar * extent
    near = sim.model.vis.map.znear * extent
    return near / (1.0 - depth_buffer * (1.0 - near / far))


def draw_depth_aware_axis_line(
    image,
    depth_map,
    sim,
    camera_name,
    start,
    end,
    color,
    thickness,
    *,
    samples=120,
    depth_tolerance=0.01,
):
    """Draw a 3D line with approximate depth occlusion in the 2D viewer."""
    height, width = depth_map.shape
    last_pixel = None
    visible_end_pixel = None

    for alpha in np.linspace(0.0, 1.0, samples):
        point = (1.0 - alpha) * start + alpha * end
        projected = project_world_to_viewer_pixel_depth(
            sim, camera_name, point, width, height
        )
        if projected is None:
            last_pixel = None
            continue

        x, y, point_depth = projected
        if x < 0 or x >= width or y < 0 or y >= height:
            last_pixel = None
            continue

        scene_depth = depth_map[y, x]
        visible = point_depth <= scene_depth + depth_tolerance
        if not visible:
            last_pixel = None
            continue

        pixel = (x, y)
        if last_pixel is not None and pixel != last_pixel:
            cv2.line(image, last_pixel, pixel, color, thickness, cv2.LINE_AA)
        last_pixel = pixel
        visible_end_pixel = pixel

    if visible_end_pixel is not None:
        cv2.circle(image, visible_end_pixel, 3, color, -1, cv2.LINE_AA)
    return visible_end_pixel


def install_object_axis_overlay(
    env,
    camera_name,
    *,
    object_names=None,
    axis_length=0.55,
    thickness=1,
    alpha=0.28,
):
    """
    Draw object center axes only on the live OpenCV collection window.

    This intentionally patches the on-screen viewer render path instead of
    adding MJCF geoms or sites, so the guide lines are not stored in model XML,
    states, RGB observations, or videos generated later from the demo.
    """
    viewer = getattr(env, "viewer", None)
    if viewer is None or not hasattr(viewer, "render"):
        return False
    if getattr(viewer, "_object_axis_overlay_installed", False):
        return True

    sim = env.sim
    object_names = object_names or get_movable_object_names(env)
    original_render = viewer.render
    axis_specs = (
        (0, (0, 0, 255), "x"),
        (1, (0, 255, 0), "y"),
        (2, (255, 0, 0), "z"),
    )

    def render_with_object_axes():
        width = getattr(viewer, "width", 1280)
        height = getattr(viewer, "height", 800)
        active_camera = getattr(viewer, "camera_name", camera_name)

        # Match robosuite's OpenCVRenderer.render(), then add 2D guide lines
        # after rendering. This keeps the overlay out of all saved data.
        rendered = sim.render(
            camera_name=active_camera,
            height=height,
            width=width,
            depth=True,
        )
        image = rendered[0][..., ::-1]
        depth_map = depth_buffer_to_metric(sim, rendered[1])
        image = np.flip(image, axis=0).copy()
        depth_map = np.flip(depth_map, axis=0)
        overlay = image.copy()

        for object_name in object_names:
            body_id = getattr(env, "obj_body_id", {}).get(object_name)
            if body_id is None:
                continue
            center = np.asarray(sim.data.body_xpos[body_id], dtype=np.float32)
            rotation = np.asarray(
                sim.data.body_xmat[body_id], dtype=np.float32
            ).reshape(3, 3)
            center_projected = project_world_to_viewer_pixel_depth(
                sim, active_camera, center, width, height
            )
            if center_projected is not None:
                center_x, center_y, center_depth = center_projected
                if (
                    0 <= center_x < width
                    and 0 <= center_y < height
                    and center_depth <= depth_map[center_y, center_x] + 0.01
                ):
                    cv2.circle(
                        overlay,
                        (center_x, center_y),
                        3,
                        (0, 255, 255),
                        -1,
                        cv2.LINE_AA,
                    )

            for axis_index, axis_color, axis_label in axis_specs:
                axis = rotation[:, axis_index]
                start = center - 0.5 * axis_length * axis
                end = center + 0.5 * axis_length * axis
                end_px = draw_depth_aware_axis_line(
                    overlay,
                    depth_map,
                    sim,
                    active_camera,
                    start,
                    end,
                    axis_color,
                    thickness,
                )
                if end_px is not None:
                    cv2.putText(
                        overlay,
                        axis_label,
                        (end_px[0] + 3, end_px[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        axis_color,
                        1,
                        cv2.LINE_AA,
                    )

        image = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)
        cv2.imshow("offscreen render", image)
        move_collection_window()
        key = cv2.waitKey(1)
        if getattr(viewer, "keypress_callback", None):
            viewer.keypress_callback(key)

    # Keep a handle to the original render in case the caller wants to inspect
    # or restore it while debugging.
    viewer._render_without_object_axes = original_render
    viewer._object_axis_overlay_installed = True
    viewer.render = render_with_object_axes
    return True


def collect_human_trajectory(
    env,
    device,
    arm,
    env_configuration,
    problem_info,
    camera_name,
    control_frame,
    rotation_frame,
    success_hold_seconds,
    require_open_gripper_on_success,
    open_gripper_threshold,
    show_object_axis_overlay=True,
    object_axis_length=0.55,
    show_auxiliary_views=True,
    wrist_camera="robot0_eye_in_hand",
    table_front_camera="frontview",
    aux_view_width=320,
    aux_view_height=240,
    remove_directory=[],
    fixed_init_state_cache=None,
    fixed_init_settle_steps=100,
    fixed_init_state_path=None,
):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    reset_success = False
    while not reset_success:
        try:
            env.reset()
            if fixed_init_state_cache is not None and fixed_init_state_cache:
                env.reset_from_xml_string(fixed_init_state_cache["xml"])
                env.sim.reset()
                env.sim.set_state_from_flattened(fixed_init_state_cache["state"])
                env.sim.forward()
            else:
                # Let newly randomized objects settle before showing the scene or
                # recording the first action. Direct simulation steps bypass the
                # data logger, so this motion is not part of the demonstration.
                for _ in range(fixed_init_settle_steps):
                    env.sim.step()
                env.sim.forward()
                if fixed_init_state_cache is not None:
                    fixed_init_state_cache["xml"] = env.sim.model.get_xml()
                    fixed_init_state_cache["state"] = np.array(
                        env.sim.get_state().flatten()
                    )
                    if fixed_init_state_path:
                        os.makedirs(
                            os.path.dirname(os.path.abspath(fixed_init_state_path)),
                            exist_ok=True,
                        )
                        np.savez_compressed(
                            fixed_init_state_path,
                            xml=np.array(fixed_init_state_cache["xml"]),
                            state=fixed_init_state_cache["state"],
                        )
                        print(
                            "Fixed initial state saved to "
                            f"{fixed_init_state_path}"
                        )
                    print("Fixed initial state captured. Reusing it for later demos.")
            env._current_task_instance_xml = env.sim.model.get_xml()
            env._current_task_instance_state = np.array(
                env.sim.get_state().flatten()
            )
            if show_auxiliary_views:
                configure_table_level_front_camera(env, table_front_camera)
            reset_success = True
        except:
            continue

    # ID = 2 always corresponds to agentview
    env.render()
    move_collection_window()
    render_auxiliary_views(
        env,
        enabled=show_auxiliary_views,
        wrist_camera=wrist_camera,
        table_front_camera=table_front_camera,
        width=aux_view_width,
        height=aux_view_height,
    )
    if show_object_axis_overlay:
        if install_object_axis_overlay(
            env,
            camera_name,
            axis_length=object_axis_length,
        ):
            env.render()
            move_collection_window()
            render_auxiliary_views(
                env,
                enabled=show_auxiliary_views,
                wrist_camera=wrist_camera,
                table_front_camera=table_front_camera,
                width=aux_view_width,
                height=aux_view_height,
            )

    device.start_control()

    # Loop until we get a reset from the input or the task completes
    saving = True
    count = 0
    previous_goal_status = None
    previous_gripper_open = None

    while True:
        count += 1
        # Set active robot
        active_robot = (
            env.robots[0]
            if env_configuration == "bimanual"
            else env.robots[arm == "left"]
        )

        # Get the newest action
        action, grasp = input2action(
            device=device,
            robot=active_robot,
            active_arm=arm,
            env_configuration=env_configuration,
        )

        # If action is none, then this a reset so we should break
        if action is None:
            print("Break")
            saving = False
            break

        if control_frame != "libero" or rotation_frame != "libero":
            action = transform_keyboard_action(
                action,
                env.sim,
                camera_name,
                active_robot,
                control_frame=control_frame,
                rotation_frame=rotation_frame,
            )
            action = np.clip(action, -1.0, 1.0)

        # Run environment step

        env.step(action)
        env.render()
        move_collection_window()
        render_auxiliary_views(
            env,
            enabled=show_auxiliary_views,
            wrist_camera=wrist_camera,
            table_front_camera=table_front_camera,
            width=aux_view_width,
            height=aux_view_height,
        )
        goal_status = tuple(
            bool(env._eval_predicate(state))
            for state in env.parsed_problem["goal_state"]
        )
        if goal_status != previous_goal_status:
            print("Goal status:")
            for state, satisfied in zip(
                env.parsed_problem["goal_state"], goal_status
            ):
                print(f"  [{'OK' if satisfied else '--'}] {' '.join(state)}")
            previous_goal_status = goal_status
        gripper_open = is_gripper_open(env, open_gripper_threshold)
        if (
            require_open_gripper_on_success
            and all(goal_status)
            and gripper_open != previous_gripper_open
        ):
            print(
                "Open gripper success condition: "
                f"{'OK' if gripper_open else 'waiting for gripper to open'}"
            )
            previous_gripper_open = gripper_open

        # Latch success as soon as every goal is satisfied. Rechecking during
        # the trailing frames can fail when an object settles near a region
        # boundary, leaving an otherwise completed collection running.
        if all(goal_status) and (
            not require_open_gripper_on_success or gripper_open
        ):
            control_freq = int(getattr(env, "control_freq", 20))
            trailing_steps = max(0, int(round(success_hold_seconds * control_freq)))
            print(
                "All goals satisfied. Recording "
                f"{success_hold_seconds:.1f}s ({trailing_steps} control steps) "
                "before saving..."
            )
            hold_action = np.zeros_like(action)
            if hold_action.size:
                hold_action[-1] = action[-1]
            for _ in range(trailing_steps):
                env.step(hold_action)
                env.render()
                move_collection_window()
                render_auxiliary_views(
                    env,
                    enabled=show_auxiliary_views,
                    wrist_camera=wrist_camera,
                    table_front_camera=table_front_camera,
                    width=aux_view_width,
                    height=aux_view_height,
                )
            print("Task completed. Saving demonstration...")
            break

    print(count)
    # cleanup for end of data collection episodes
    if not saving:
        remove_directory.append(env.ep_directory.split("/")[-1])
    env.close()
    return saving


def find_successful_episode_directories(directory, remove_directory=None):
    """Return raw episode directory names that contain a successful rollout."""
    remove_directory = set(remove_directory or [])
    successful_directories = []

    for ep_directory in os.listdir(directory):
        if ep_directory in remove_directory:
            continue
        episode_path = os.path.join(directory, ep_directory)
        if not os.path.isdir(episode_path):
            continue

        state_paths = sorted(glob(os.path.join(episode_path, "state_*.npz")))
        if not state_paths:
            continue

        successful = False
        for state_file in state_paths:
            dic = np.load(state_file, allow_pickle=True)
            if "successful" in dic.files:
                successful = successful or bool(dic["successful"])
        if successful:
            successful_directories.append(ep_directory)

    successful_directories.sort(
        key=lambda name: os.path.getmtime(os.path.join(directory, name))
    )
    return successful_directories


def gather_demonstrations_as_hdf5(
    directory, out_dir, env_info, args, remove_directory=None, include_directories=None
):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    os.makedirs(out_dir, exist_ok=True)
    remove_directory = set(remove_directory or [])
    include_directories = (
        set(include_directories) if include_directories is not None else None
    )

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        # print(ep_directory)
        if ep_directory in remove_directory:
            # print("Skipping")
            continue
        if include_directories is not None and ep_directory not in include_directories:
            continue
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        successful = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])
            if "successful" in dic.files:
                successful = successful or bool(dic["successful"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])

        if len(states) == 0 or not successful:
            continue

        # Delete the first actions and the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        actions = np.array(actions)
        if args.smooth_actions:
            actions = smooth_action_labels(
                actions,
                window_size=args.smooth_window_size,
                motion_dims=args.smooth_motion_dims,
            )
            ep_data_grp.attrs["actions_smoothed"] = True
            ep_data_grp.attrs["smooth_window_size"] = args.smooth_window_size
            ep_data_grp.attrs["smooth_motion_dims"] = args.smooth_motion_dims
        else:
            ep_data_grp.attrs["actions_smoothed"] = False

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=actions)
        ep_data_grp.attrs["num_samples"] = len(actions)
        ep_data_grp.attrs["success"] = True

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    grp.attrs["problem_info"] = json.dumps(problem_info)
    grp.attrs["bddl_file_name"] = args.bddl_file
    with open(args.bddl_file, "r", encoding="utf-8") as bddl_file:
        grp.attrs["bddl_file_content"] = bddl_file.read()
    grp.attrs["num_demos"] = num_eps

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default="data/datasets/demonstration_data",
    )
    parser.add_argument(
        "--task-index",
        type=str,
        default=None,
        help="Optional task subdirectory under --directory, e.g. 1 saves to demonstration_data/task1.",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default=["Panda"],
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--control-frame",
        choices=("libero", "eef", "camera", "world"),
        default="libero",
        help="Interpret keyboard Cartesian commands. 'libero' keeps the original robosuite/LIBERO mapping.",
    )
    parser.add_argument(
        "--rotation-frame",
        choices=("libero", "eef", "camera", "world"),
        default="libero",
        help="Interpret keyboard rotation commands. 'libero' keeps the original robosuite/LIBERO mapping.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="OSC_POSE",
        help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="spacemouse",
        choices=("keyboard", "spacemouse", "joycon", "xr", "metaquest"),
        help="Input device used for teleoperation.",
    )
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.5,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--num-demonstration",
        type=int,
        default=50,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--fixed-init-state",
        action="store_true",
        help=(
            "Capture the first reset state and reuse it for every later "
            "demonstration so all episodes start from the same scene."
        ),
    )
    parser.add_argument(
        "--fixed-init-settle-steps",
        type=int,
        default=100,
        help=(
            "Simulation settle steps before capturing the fixed initial state. "
            "These steps are not recorded."
        ),
    )
    parser.add_argument(
        "--fixed-init-state-path",
        type=str,
        default=None,
        help=(
            "Path to a saved fixed initial state .npz. When omitted with "
            "--fixed-init-state, a default file is stored under --directory."
        ),
    )
    parser.add_argument(
        "--success-hold-seconds",
        type=float,
        default=4.0,
        help="Seconds to keep recording after all goal predicates are satisfied.",
    )
    parser.add_argument(
        "--allow-closed-gripper-on-success",
        action="store_true",
        help="Finish collection as soon as goals are satisfied, even if the gripper is closed.",
    )
    parser.add_argument(
        "--open-gripper-threshold",
        type=float,
        default=FULLY_OPEN_GRIPPER_THRESHOLD,
        help="Every gripper qpos absolute value must reach this threshold before success can finish.",
    )
    parser.add_argument(
        "--smooth-actions",
        action="store_true",
        help="Smooth continuous arm action labels before writing demo.hdf5.",
    )
    parser.add_argument(
        "--smooth-window-size",
        type=int,
        default=5,
        help="Odd moving-average window for action smoothing. 3 is light, 5 is moderate, 7 is strong.",
    )
    parser.add_argument(
        "--smooth-motion-dims",
        type=int,
        default=6,
        help="Number of leading continuous action dimensions to smooth; gripper command is preserved.",
    )
    parser.add_argument(
        "--hide-object-axis-overlay",
        action="store_true",
        help="Hide live object XYZ-axis guide lines in the collection viewer.",
    )
    parser.add_argument(
        "--object-axis-length",
        type=float,
        default=0.55,
        help="Length of live object XYZ-axis guide lines in meters.",
    )
    parser.add_argument(
        "--hide-auxiliary-views",
        action="store_true",
        help="Hide live wrist and table-level front helper windows.",
    )
    parser.add_argument(
        "--wrist-camera",
        type=str,
        default="robot0_eye_in_hand",
        help="Camera name used for the live wrist helper window.",
    )
    parser.add_argument(
        "--table-front-camera",
        type=str,
        default="frontview",
        help="Camera name reused for the live low table-level front helper window.",
    )
    parser.add_argument(
        "--aux-view-width",
        type=int,
        default=320,
        help="Width of live helper camera windows.",
    )
    parser.add_argument(
        "--aux-view-height",
        type=int,
        default=240,
        help="Height of live helper camera windows.",
    )
    parser.add_argument("--bddl-file", type=str)

    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50734)
    parser.add_argument(
        "--joycon-side",
        choices=("left", "right"),
        default="right",
        help="Which JoyCon controller to use when --device joycon.",
    )
    parser.add_argument(
        "--joycon-pos-scale",
        type=float,
        default=0.2,
        help="Extra scale applied to JoyCon position deltas before robosuite action scaling.",
    )
    parser.add_argument(
        "--joycon-rot-scale",
        type=float,
        default=0.2,
        help="Extra scale applied to JoyCon rotation deltas before robosuite action scaling.",
    )
    parser.add_argument(
        "--joycon-disable-rotation",
        action="store_true",
        help="Ignore JoyCon IMU rotation and only use translation plus gripper.",
    )
    parser.add_argument(
        "--xr-controller",
        choices=("left", "right"),
        default="right",
        help="Which XR controller to use when --device xr.",
    )
    parser.add_argument(
        "--xr-source",
        choices=("udp", "direct"),
        default="udp",
        help="Read XR data from a UDP bridge or directly from xrobotoolkit_sdk.",
    )
    parser.add_argument(
        "--xr-udp-host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind for XR UDP bridge packets.",
    )
    parser.add_argument(
        "--xr-udp-port",
        type=int,
        default=50505,
        help="UDP port to bind for XR bridge packets.",
    )
    parser.add_argument(
        "--xr-pos-scale",
        type=float,
        default=0.12,
        help="Extra scale applied to XR controller position deltas before robosuite action scaling.",
    )
    parser.add_argument(
        "--xr-rot-scale",
        type=float,
        default=0.12,
        help="Extra scale applied to XR controller rotation deltas before robosuite action scaling.",
    )
    parser.add_argument(
        "--xr-active-threshold",
        type=float,
        default=0.8,
        help="Grip value required to enable XR arm motion.",
    )
    parser.add_argument(
        "--xr-gripper-threshold",
        type=float,
        default=0.5,
        help="Trigger value required to close the gripper in XR mode.",
    )
    parser.add_argument(
        "--xr-disable-rotation",
        action="store_true",
        help="Ignore XR controller rotation and only use translation plus gripper.",
    )
    parser.add_argument(
        "--metaquest-controller",
        choices=("left", "right"),
        default="right",
        help="Which Meta Quest controller to use when --device metaquest.",
    )
    parser.add_argument(
        "--metaquest-ip-address",
        type=str,
        default=None,
        help=(
            "Optional Meta Quest ADB-over-WiFi IP. Leave unset to use the "
            "USB ADB connection."
        ),
    )
    parser.add_argument(
        "--metaquest-apk-name",
        type=str,
        default="com.rail.oculus.teleop",
        help="Meta Quest teleop APK package name.",
    )
    parser.add_argument(
        "--metaquest-pos-scale",
        type=float,
        default=0.6,
        help=(
            "Extra scale applied to Meta Quest controller position deltas "
            "before robosuite action scaling."
        ),
    )
    parser.add_argument(
        "--metaquest-rot-scale",
        type=float,
        default=0.03,
        help=(
            "Extra scale applied to Meta Quest controller rotation deltas "
            "before robosuite action scaling."
        ),
    )
    parser.add_argument(
        "--metaquest-active-threshold",
        type=float,
        default=0.5,
        help="Grip value required to enable Meta Quest arm motion.",
    )
    parser.add_argument(
        "--metaquest-gripper-threshold",
        type=float,
        default=0.5,
        help="Trigger value required to close the gripper in Meta Quest mode.",
    )
    parser.add_argument(
        "--metaquest-disable-rotation",
        action="store_true",
        help=(
            "Ignore Meta Quest controller rotation and only use translation "
            "plus gripper."
        ),
    )

    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    config = {
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    assert os.path.exists(args.bddl_file)
    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    # Check if we're using a multi-armed environment and use env_configuration argument if so

    # Create environment
    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]
    if "TwoArm" in problem_name:
        config["env_configuration"] = args.config
    print(language_instruction)
    env = TASK_MAPPING[problem_name](
        bddl_file_name=args.bddl_file,
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "data/datasets/demonstration_data/tmp/{}_ln_{}/{}".format(
        problem_name,
        language_instruction.replace(" ", "_").strip('""'),
        str(time.time()).replace(".", "_"),
    )

    env = DataCollectionWrapper(env, tmp_directory)
    if not args.hide_object_axis_overlay:
        print(
            "Live object XYZ-axis overlay is enabled. "
            "It is drawn only in the collection window and is not saved "
            "to demo.hdf5, RGB frames, or videos."
        )
    if not args.hide_auxiliary_views:
        print(
            "Live auxiliary views are enabled: wrist view and low table-level "
            "front view. They are display-only and are not saved to demo.hdf5."
        )

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity
        )
        if args.control_frame == "libero":
            print(
                "Translation controls use the original robosuite/LIBERO "
                "keyboard action mapping."
            )
        elif args.control_frame == "eef":
            print(
                "Translation controls use the current end-effector axes, so "
                "W/A/S/D/R/F move relative to the gripper frame."
            )
        elif args.control_frame == "camera":
            print(
                "Camera controls: W/S screen up/down, A/D screen left/right, "
                "R/F world up/down"
            )
        elif args.control_frame == "world":
            print("World controls: W/A/S/D/R/F move along fixed world axes.")
        if args.rotation_frame == "libero":
            print(
                "Rotation controls use the original robosuite/LIBERO "
                "keyboard action mapping."
            )
        elif args.rotation_frame == "eef":
            print(
                "Rotation controls use the current end-effector axes, so the "
                "wrist rotates about the gripper frame."
            )
        elif args.rotation_frame == "camera":
            print(
                "Camera rotations: Z/X about screen-right, T/G about screen-up, "
                "C/V about viewing axis"
            )

        def handle_keypress(key):
            if key == 27:  # OpenCV key code for ESC
                env.close()
                raise SystemExit(0)
            device.on_press(key)

        env.viewer.add_keypress_callback(handle_keypress)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            args.vendor_id,
            args.product_id,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "joycon":
        device = JoyConRobosuiteDevice(
            device=args.joycon_side,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            pos_scale=args.joycon_pos_scale,
            rot_scale=args.joycon_rot_scale,
            disable_rotation=args.joycon_disable_rotation,
        )
    elif args.device == "xr":
        device = XRRobosuiteDevice(
            controller=args.xr_controller,
            source=args.xr_source,
            udp_host=args.xr_udp_host,
            udp_port=args.xr_udp_port,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            pos_scale=args.xr_pos_scale,
            rot_scale=args.xr_rot_scale,
            active_threshold=args.xr_active_threshold,
            gripper_threshold=args.xr_gripper_threshold,
            disable_rotation=args.xr_disable_rotation,
        )
    elif args.device == "metaquest":
        device = MetaQuestRobosuiteDevice(
            controller=args.metaquest_controller,
            ip_address=args.metaquest_ip_address,
            apk_name=args.metaquest_apk_name,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            pos_scale=args.metaquest_pos_scale,
            rot_scale=args.metaquest_rot_scale,
            active_threshold=args.metaquest_active_threshold,
            gripper_threshold=args.metaquest_gripper_threshold,
            disable_rotation=args.metaquest_disable_rotation,
        )
    else:
        raise Exception(
            "Invalid device choice: choose 'keyboard', 'spacemouse', "
            "'joycon', 'xr', or 'metaquest'."
        )

    # make a new timestamped directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = resolve_output_root(args.directory, args.task_index, args.bddl_file)
    print(f"Saving demonstrations under: {output_root}")
    new_dir = os.path.join(
        output_root,
        f"{domain_name}_ln_{problem_name}_{timestamp}_"
        + language_instruction.replace(" ", "_").strip('""'),
    )

    os.makedirs(new_dir)

    # collect demonstrations

    remove_directory = []
    exported_episode_directories = set()
    fixed_init_state_cache = {} if args.fixed_init_state else None
    fixed_init_state_path = None
    if fixed_init_state_cache is not None:
        fixed_init_state_path = args.fixed_init_state_path
        if fixed_init_state_path is None:
            fixed_init_state_path = os.path.join(
                output_root,
                "fixed_init_state.npz",
            )
        if os.path.exists(fixed_init_state_path):
            fixed_data = np.load(fixed_init_state_path, allow_pickle=True)
            fixed_init_state_cache["xml"] = str(fixed_data["xml"].item())
            fixed_init_state_cache["state"] = np.asarray(fixed_data["state"])
            print(f"Loaded fixed initial state from {fixed_init_state_path}")
        else:
            print(
                "Fixed initial state is enabled. The first reset will be "
                f"saved to {fixed_init_state_path}"
            )
    i = 0
    while i < args.num_demonstration:
        print(i)
        saving = collect_human_trajectory(
            env,
            device,
            args.arm,
            args.config,
            problem_info,
            args.camera,
            args.control_frame,
            args.rotation_frame,
            args.success_hold_seconds,
            not args.allow_closed_gripper_on_success,
            args.open_gripper_threshold,
            not args.hide_object_axis_overlay,
            args.object_axis_length,
            not args.hide_auxiliary_views,
            args.wrist_camera,
            args.table_front_camera,
            args.aux_view_width,
            args.aux_view_height,
            remove_directory,
            fixed_init_state_cache,
            args.fixed_init_settle_steps,
            fixed_init_state_path,
        )
        if saving:
            successful_directories = find_successful_episode_directories(
                tmp_directory, remove_directory
            )
            new_successful_directories = [
                name
                for name in successful_directories
                if name not in exported_episode_directories
            ]
            if not new_successful_directories:
                print("No new successful episode directory found; skipping export.")
                continue

            for episode_directory in new_successful_directories:
                episode_index = len(exported_episode_directories) + 1
                episode_out_dir = os.path.join(new_dir, f"demo_{episode_index:06d}")
                gather_demonstrations_as_hdf5(
                    tmp_directory,
                    episode_out_dir,
                    env_info,
                    args,
                    remove_directory,
                    include_directories=[episode_directory],
                )
                exported_episode_directories.add(episode_directory)
                print(
                    "Saved successful trajectory "
                    f"{episode_index} to {os.path.join(episode_out_dir, 'demo.hdf5')}"
                )
                i += 1
                if i >= args.num_demonstration:
                    break

    close_device = getattr(device, "close", None)
    if close_device is not None:
        close_device()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    print("Requested demonstrations saved. Exiting collection program.")
