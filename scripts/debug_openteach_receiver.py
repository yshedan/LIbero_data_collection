#!/usr/bin/env python3
"""Debug whether Open-Teach Quest APK packets reach this computer."""

import argparse
import time

import zmq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--right-port", type=int, default=8087)
    parser.add_argument("--button-port", type=int, default=8095)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    context = zmq.Context()
    sockets = []
    poller = zmq.Poller()
    for name, port in (("right_hand", args.right_port), ("button", args.button_port)):
        sock = context.socket(zmq.PULL)
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.bind(f"tcp://{args.host}:{port}")
        poller.register(sock, zmq.POLLIN)
        sockets.append((name, sock))
        print(f"Listening for {name} packets on tcp://{args.host}:{port}")

    print("Open the Quest Open-Teach app, set this computer IP, then press Stream.")
    deadline = time.time() + args.timeout
    counts = {name: 0 for name, _ in sockets}
    try:
        while time.time() < deadline:
            events = dict(poller.poll(timeout=500))
            for name, sock in sockets:
                if sock in events:
                    msg = sock.recv()
                    counts[name] += 1
                    preview = msg[:120]
                    print(f"[{name}] packet {counts[name]}: {preview!r}")
                    if all(value > 0 for value in counts.values()):
                        print("Both right hand and button streams are reaching this computer.")
                        return
        print(f"Timeout. Packet counts: {counts}")
    finally:
        for _, sock in sockets:
            sock.close()
        context.term()


if __name__ == "__main__":
    main()
