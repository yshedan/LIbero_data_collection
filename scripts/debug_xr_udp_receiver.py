"""Print XR UDP packets sent by scripts/unitree_televuer_bridge.py."""

import argparse
import json
import socket
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50505)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.settimeout(1.0)
    print(f"Listening for XR UDP packets on {args.host}:{args.port}")
    deadline = time.time() + args.timeout
    count = 0
    while time.time() < deadline:
        try:
            data, addr = sock.recvfrom(65536)
        except socket.timeout:
            continue
        count += 1
        try:
            packet = json.loads(data.decode("utf-8"))
        except Exception:
            print(f"#{count} from {addr}: non-json packet ({len(data)} bytes)")
            continue
        pose = packet.get("pose", [])
        pos = pose[:3] if len(pose) >= 3 else []
        print(
            "#{count} from {addr}: ready={ready} pos={pos} trigger={trigger:.2f} grip={grip:.2f} buttons={buttons}".format(
                count=count,
                addr=addr,
                ready=packet.get("ready"),
                pos=[round(float(x), 3) for x in pos],
                trigger=float(packet.get("trigger", 0.0)),
                grip=float(packet.get("grip", 0.0)),
                buttons=packet.get("buttons", {}),
            ),
            flush=True,
        )
    print(f"Done. Received {count} packet(s).")


if __name__ == "__main__":
    main()
