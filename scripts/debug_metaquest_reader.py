"""Debug Meta Quest controller poses through meta_quest_teleop / oculus_reader."""

import argparse
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip-address", default=None, help="Quest IP. Omit for USB ADB.")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--apk-name", default="com.rail.oculus.teleop")
    args = parser.parse_args()

    from meta_quest_teleop.reader import MetaQuestReader

    reader = MetaQuestReader(ip_address=args.ip_address, APK_name=args.apk_name)
    deadline = time.time() + args.seconds
    count = 0
    try:
        while time.time() < deadline:
            transforms, buttons = reader.get_transformations_and_buttons()
            transforms = transforms or {}
            buttons = buttons or {}
            count += 1
            parts = []
            for key in ("l", "r"):
                mat = transforms.get(key)
                if mat is None:
                    parts.append(f"{key}=None")
                else:
                    pos = np.asarray(mat)[:3, 3]
                    parts.append(
                        f"{key}=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})"
                    )
            print(f"#{count:03d} {' '.join(parts)} buttons={buttons}", flush=True)
            time.sleep(0.2)
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
