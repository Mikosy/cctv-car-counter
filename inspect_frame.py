"""
Saves the first frame of a video (resized the same way the pipeline does)
with a coordinate grid drawn on top, so you can read off real pixel
coordinates for config.GATE_LINE instead of guessing.

Usage:
    python inspect_frame.py traffic.mp4
    python inspect_frame.py traffic.mp4 --frame 50   # a later frame, if the
                                                       # first one is empty
    python inspect_frame.py traffic.mp4 --out frame.jpg

Open the saved image, find two points that form a line across the road the
vehicles must cross (e.g. where you want the "gate"), and copy those (x, y)
pairs into GATE_LINE in config.py.
"""

import argparse

import cv2

import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Path to the video file")
    parser.add_argument("--frame", type=int, default=0, help="Frame number to grab")
    parser.add_argument("--out", default="inspect_frame.jpg", help="Output image path")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {args.source}")

    for _ in range(args.frame + 1):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Video has fewer than {args.frame + 1} frames")
    cap.release()

    # Match the pipeline's resize so the coordinates you read off here are
    # the same ones GATE_LINE needs to use.
    if config.INFERENCE_FRAME_WIDTH is not None:
        h, w = frame.shape[:2]
        scale = config.INFERENCE_FRAME_WIDTH / w
        frame = cv2.resize(frame, (config.INFERENCE_FRAME_WIDTH, int(h * scale)))

    h, w = frame.shape[:2]
    step = 100
    for x in range(0, w, step):
        cv2.line(frame, (x, 0), (x, h), (0, 255, 0), 1)
        cv2.putText(frame, str(x), (x + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    for y in range(0, h, step):
        cv2.line(frame, (0, y), (w, y), (0, 255, 0), 1)
        cv2.putText(frame, str(y), (2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.imwrite(args.out, frame)
    print(f"Saved {args.out} ({w}x{h}). Open it, pick two (x, y) points across")
    print("the road/gate, and set them as GATE_LINE in config.py.")


if __name__ == "__main__":
    main()
