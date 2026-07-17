"""Bridge Unitree TeleVuer / WebXR controller data into collect_demonstration.py.

This script starts TeleVuer's WebXR server and forwards the selected controller
pose to the existing XR UDP input in scripts/collect_demonstration.py.
"""

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np


R_HEADSET_TO_WORLD = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def _add_televuer_path(path):
    if not path:
        return
    path = Path(path).expanduser().resolve()
    candidates = [
        path,
        path / "teleop" / "televuer" / "src",
        path / "src",
    ]
    for candidate in candidates:
        if (candidate / "televuer").exists():
            sys.path.insert(0, str(candidate))
            return


def _mat_to_quat_xyzw(rot):
    """Convert a 3x3 rotation matrix to a normalized xyzw quaternion."""
    rot = np.asarray(rot, dtype=np.float64)
    trace = np.trace(rot)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def _controller_value(data, controller, name, default=0.0):
    return getattr(data, f"{controller}_ctrl_{name}", default)


def _button_state(data, controller, name):
    return bool(getattr(data, f"{controller}_ctrl_{name}", False))


def _hand_value(data, controller, name, default=0.0):
    return getattr(data, f"{controller}_hand_{name}", default)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--televuer-path",
        default="/home/ysd/NewDisk/ysd/code/xr_teleoperate",
        help="Path to xr_teleoperate repo, televuer repo, or televuer/src.",
    )
    parser.add_argument("--controller", choices=("left", "right"), default="right")
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=50505)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument(
        "--use-hand-tracking",
        action="store_true",
        help="Use hand tracking instead of Quest controller tracking.",
    )
    parser.add_argument(
        "--display-mode",
        choices=("pass-through", "ego", "immersive"),
        default="pass-through",
        help="Use pass-through for controller-only teleop without image streaming.",
    )
    parser.add_argument("--cert-file", default=None)
    parser.add_argument("--key-file", default=None)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one controller data summary per second for debugging.",
    )
    args = parser.parse_args()

    _add_televuer_path(args.televuer_path)
    try:
        from televuer import TeleVuerWrapper
    except ImportError as exc:
        raise ImportError(
            "Cannot import televuer. Clone Unitree xr_teleoperate with "
            "--recurse-submodules, then install its dependencies or pass "
            "--televuer-path /path/to/xr_teleoperate."
        ) from exc

    tv = TeleVuerWrapper(
        use_hand_tracking=args.use_hand_tracking,
        binocular=True,
        img_shape=(480, 1280),
        display_fps=args.fps,
        display_mode=args.display_mode,
        zmq=False,
        webrtc=False,
        cert_file=args.cert_file,
        key_file=args.key_file,
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / max(args.fps, 1.0)
    print("")
    print("Unitree TeleVuer bridge is running.")
    print("Quest URL over WiFi:")
    print("  https://<PC_IP>:8012/?ws=wss://<PC_IP>:8012")
    print("Quest URL over USB adb reverse:")
    print("  https://127.0.0.1:8012/?ws=wss://127.0.0.1:8012")
    print(f"Forwarding {args.controller} controller to UDP {args.udp_host}:{args.udp_port}")
    print("")

    try:
        last_debug_time = 0.0
        while True:
            tele_data = tv.get_tele_data()
            wrist_pose = getattr(tele_data, f"{args.controller}_wrist_pose")
            desired_pos = np.asarray(wrist_pose[:3, 3], dtype=np.float64)
            desired_rot = np.asarray(wrist_pose[:3, :3], dtype=np.float64)

            # collect_demonstration.py's XR device applies R_HEADSET_TO_WORLD to
            # incoming poses. Send the inverse-transformed pose so the existing
            # adapter can be reused without changing the collector.
            send_pos = R_HEADSET_TO_WORLD.T @ desired_pos
            send_rot = R_HEADSET_TO_WORLD.T @ desired_rot
            quat = _mat_to_quat_xyzw(send_rot)

            if args.use_hand_tracking:
                pinch_value = float(_hand_value(tele_data, args.controller, "pinchValue", 10.0))
                squeeze_value = float(_hand_value(tele_data, args.controller, "squeezeValue", 0.0))
                # TeleVuerWrapper reports pinchValue as distance-like:
                # around 10 -> open, around 0 -> pinching.
                trigger = np.clip(1.0 - pinch_value / 10.0, 0.0, 1.0)
                a_button = False
                b_button = bool(_hand_value(tele_data, args.controller, "squeeze", False))
            else:
                trigger_value = float(_controller_value(tele_data, args.controller, "triggerValue", 10.0))
                squeeze_value = float(_controller_value(tele_data, args.controller, "squeezeValue", 0.0))
                # TeleVuerWrapper reports triggerValue as 10 -> unpressed, 0 -> fully pressed.
                trigger = np.clip(1.0 - trigger_value / 10.0, 0.0, 1.0)
                a_button = _button_state(tele_data, args.controller, "aButton")
                b_button = _button_state(tele_data, args.controller, "bButton")

            packet = {
                "pose": [*send_pos.tolist(), *quat.tolist()],
                "trigger": float(trigger),
                "grip": float(squeeze_value),
                "buttons": {
                    "A": a_button,
                    "B": b_button,
                    "Y": b_button,
                },
                "ready": bool(getattr(tele_data, "motion_data_ready", False)),
                "source": "unitree_televuer",
            }
            sock.sendto(json.dumps(packet).encode("utf-8"), (args.udp_host, args.udp_port))
            if args.verbose and time.time() - last_debug_time >= 1.0:
                print(
                    "ready={ready} pos=({x:.3f}, {y:.3f}, {z:.3f}) "
                    "trigger={trigger:.2f} grip={grip:.2f} A={a} B={b}".format(
                        ready=packet["ready"],
                        x=float(desired_pos[0]),
                        y=float(desired_pos[1]),
                        z=float(desired_pos[2]),
                        trigger=packet["trigger"],
                        grip=packet["grip"],
                        a=packet["buttons"]["A"],
                        b=packet["buttons"]["B"],
                    ),
                    flush=True,
                )
                last_debug_time = time.time()
            time.sleep(period)
    except KeyboardInterrupt:
        print("Stopping Unitree TeleVuer bridge.")
    finally:
        tv.close()
        sock.close()


if __name__ == "__main__":
    main()
