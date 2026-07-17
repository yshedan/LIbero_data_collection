#!/usr/bin/env python3
"""Send XRoboToolkit controller state to LIBERO collection over UDP.

Run this script in a Python >= 3.10 environment where xrobotoolkit_sdk is
installed and XRoboToolkit PC Service is already running.
"""

import argparse
import json
import socket
import time

import xrobotoolkit_sdk as xrt


def get_controller_packet(controller):
    if controller == "left":
        pose = xrt.get_left_controller_pose()
        trigger = xrt.get_left_trigger()
        grip = xrt.get_left_grip()
        axis = xrt.get_left_axis()
        axis_click = xrt.get_left_axis_click()
    else:
        pose = xrt.get_right_controller_pose()
        trigger = xrt.get_right_trigger()
        grip = xrt.get_right_grip()
        axis = xrt.get_right_axis()
        axis_click = xrt.get_right_axis_click()

    return {
        "pose": [float(value) for value in pose],
        "trigger": float(trigger),
        "grip": float(grip),
        "axis": [float(value) for value in axis],
        "axis_click": bool(axis_click),
        "buttons": {
            "A": bool(xrt.get_A_button()),
            "B": bool(xrt.get_B_button()),
            "X": bool(xrt.get_X_button()),
            "Y": bool(xrt.get_Y_button()),
        },
        "timestamp_ns": int(xrt.get_time_stamp_ns()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=("left", "right"), default="right")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50505)
    parser.add_argument("--rate", type=float, default=60.0)
    args = parser.parse_args()

    sleep_time = 1.0 / args.rate
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    xrt.init()
    print(
        f"Streaming {args.controller} XR controller to "
        f"{args.host}:{args.port} at {args.rate:.1f} Hz"
    )
    try:
        while True:
            packet = get_controller_packet(args.controller)
            data = json.dumps(packet, separators=(",", ":")).encode("utf-8")
            sock.sendto(data, (args.host, args.port))
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Stopping XR bridge.")
    finally:
        xrt.close()
        sock.close()


if __name__ == "__main__":
    main()
