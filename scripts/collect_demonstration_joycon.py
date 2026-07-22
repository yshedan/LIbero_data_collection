"""JoyCon-specific entry point for LIBERO demonstration collection.

This wrapper uses box2ai-robotics/joycon-robotics as the input device while
reusing the main collect_demonstration.py pipeline.
"""

import argparse
import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOYCON_PATH = REPO_ROOT / "third_party" / "joycon-robotics"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "datasets0"
MAIN_SCRIPT = REPO_ROOT / "scripts" / "collect_demonstration.py"


def _has_option(argv, option):
    prefix = option + "="
    return any(arg == option or arg.startswith(prefix) for arg in argv)


def _strip_wrapper_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--joycon-path", type=Path, default=DEFAULT_JOYCON_PATH)
    wrapper_args, remaining = parser.parse_known_args(argv)
    return wrapper_args, remaining


def main():
    wrapper_args, remaining = _strip_wrapper_args(sys.argv[1:])

    joycon_path = wrapper_args.joycon_path.expanduser().resolve()
    if joycon_path.exists():
        sys.path.insert(0, str(joycon_path))
    else:
        print(
            "Warning: joycon-robotics path does not exist: "
            f"{joycon_path}. Install with: pip install -e third_party/joycon-robotics"
        )

    forwarded = list(remaining)
    if not _has_option(forwarded, "--device"):
        forwarded = ["--device", "joycon"] + forwarded
    if not _has_option(forwarded, "--directory"):
        forwarded += ["--directory", str(DEFAULT_OUTPUT_DIR)]
    if not _has_option(forwarded, "--joycon-side"):
        forwarded += ["--joycon-side", "right"]

    sys.argv = [str(MAIN_SCRIPT)] + forwarded
    runpy.run_path(str(MAIN_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
