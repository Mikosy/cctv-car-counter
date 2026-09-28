"""
CLI entry point.

    python main.py --source live
    python main.py --source path/to/test_footage.mp4
    python main.py --source path/to/test_footage.mp4 --preview

This file stays intentionally thin -- all real logic lives in pipeline.py so
a future FastAPI app can call pipeline.process() directly (e.g. from a
background task) without going through this CLI at all.
"""

import argparse

import config
import pipeline


def main():
    parser = argparse.ArgumentParser(description="Entry/exit ANPR tracker")
    parser.add_argument(
        "--source",
        required=True,
        help="'live' to use config.LIVE_SOURCE, or a path to an .mp4 file",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a debug window with detection boxes (needs a display)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print every plate detection/OCR attempt and save crops to debug_crops/",
    )
    args = parser.parse_args()

    if args.source == "live":
        source = config.LIVE_SOURCE
        label = "live"
    else:
        source = args.source
        label = args.source

    pipeline.process(source, source_label=label, show_preview=args.preview, debug=args.debug)


if __name__ == "__main__":
    main()