"""
The core loop. This is the module the FastAPI layer will eventually call
into (e.g. as a background task per camera) -- it has no CLI or web
dependencies of its own, just: give it a source, it logs events to the DB.

Both live and offline calls end up here, unchanged:
    pipeline.process("rtsp://camera-ip/stream1", source_label="live")
    pipeline.process("test_footage.mp4", source_label="test_footage.mp4")
"""

import os
import time
from dataclasses import dataclass, field

import cv2

import config
from db import init_db, get_session, VehicleEvent
from vehicle_tracker import VehicleTracker
from plate_reader import detect_plate_crop, read_plate_text, PlateVoter
from gate_line import side_of_line, box_leading_edge, direction_from_crossing


@dataclass
class TrackState:
    voter: PlateVoter = field(default_factory=PlateVoter)
    last_side: float = None       # previous smoothed value, feeds the EMA calc
    confirmed_side: float = None  # last "settled" side, used for crossing confirmation
    pending_side: float = None    # candidate new side awaiting confirmation
    pending_count: int = 0
    last_seen_frame: int = 0
    logged: bool = False


def _resize_if_needed(frame):
    if config.INFERENCE_FRAME_WIDTH is None:
        return frame
    h, w = frame.shape[:2]
    if w == config.INFERENCE_FRAME_WIDTH:
        return frame
    scale = config.INFERENCE_FRAME_WIDTH / w
    return cv2.resize(frame, (config.INFERENCE_FRAME_WIDTH, int(h * scale)))


def process(source, source_label: str = None, show_preview: bool = False, debug: bool = False):
    """
    Runs the full detect -> track -> read plate -> log pipeline on `source`.

    source: an RTSP/HTTP URL, a camera index (int), or a path to an .mp4 file
            -- cv2.VideoCapture handles all three the same way.
    source_label: what to record in the DB "source" column (defaults to the
                  source itself, stringified).
    show_preview: if True, opens a debug window with boxes drawn (only makes
                  sense when you have a display -- leave False on a server).
    debug: if True, prints every plate detection/OCR attempt and saves the
           cropped images to debug_crops/, so you can see exactly what the
           plate model and OCR are looking at.
    """
    source_label = source_label or str(source)
    init_db()
    os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
    if debug:
        os.makedirs("debug_crops", exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    tracker = VehicleTracker()
    tracks: dict[int, TrackState] = {}
    frame_idx = 0
    seen_track_ids = set()
    events_logged = 0
    session_summary = []  # one dict per logged event, printed at the end

    print(f"[pipeline] processing '{source_label}' -- Ctrl+C to stop")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # End of file for offline mode; for a live RTSP drop, you may
                # instead want to retry the connection here rather than exit.
                print("[pipeline] stream ended")
                break

            frame_idx += 1
            raw_frame = frame  # full original resolution, kept for plate OCR
            frame = _resize_if_needed(frame)  # smaller/faster, used for tracking
            # How much smaller the tracking frame is than the original --
            # used to map a detected box back to full-res coordinates.
            downscale = frame.shape[1] / raw_frame.shape[1]

            vehicles = tracker.update(frame)

            for vehicle in vehicles:
                seen_track_ids.add(vehicle.track_id)
                state = tracks.setdefault(vehicle.track_id, TrackState())
                state.last_seen_frame = frame_idx

                center = box_leading_edge(vehicle.box)
                raw_side = side_of_line(center)
                if state.last_side is None:
                    curr_side = raw_side  # nothing to smooth against yet
                else:
                    curr_side = (
                        config.GATE_SMOOTHING_ALPHA * state.last_side
                        + (1 - config.GATE_SMOOTHING_ALPHA) * raw_side
                    )

                if debug and state.last_side is None:
                    print(
                        f"[debug-line] frame={frame_idx} track={vehicle.track_id} "
                        f"center={center} first_seen_side={curr_side:.1f}"
                    )

                # Periodically attempt a plate read for this track.
                if frame_idx % config.PLATE_DETECT_EVERY_N_FRAMES == 0 and not state.logged:
                    x1, y1, x2, y2 = vehicle.box
                    # Map the box from tracking-frame coords back to the
                    # original full-res frame -- tracking doesn't need much
                    # resolution, but OCR does. Cropping from raw_frame
                    # instead of the (possibly much smaller) tracking frame
                    # gives OCR far more actual pixels of the plate to work
                    # with, especially on a high-res source like a phone video.
                    ox1, oy1, ox2, oy2 = (int(v / downscale) for v in (x1, y1, x2, y2))
                    crop = raw_frame[max(oy1, 0):oy2, max(ox1, 0):ox2]
                    plate_crop = detect_plate_crop(crop)
                    text, conf = read_plate_text(plate_crop)
                    if text:
                        state.voter.add(text, conf)

                    if debug:
                        found_plate = plate_crop is not None and plate_crop.size > 0
                        print(
                            f"[debug] frame={frame_idx} track={vehicle.track_id} "
                            f"vehicle_box={vehicle.box} plate_found={found_plate} "
                            f"ocr_text={text!r} ocr_conf={conf:.2f}"
                        )
                        if found_plate:
                            cv2.imwrite(
                                f"debug_crops/f{frame_idx}_t{vehicle.track_id}_plate.jpg",
                                plate_crop,
                            )
                        else:
                            cv2.imwrite(
                                f"debug_crops/f{frame_idx}_t{vehicle.track_id}_vehicle.jpg",
                                crop,
                            )

                # Crossing confirmation: don't trust a single flipped frame --
                # require the new side to persist for CONFIRM_FRAMES
                # consecutive frames before it counts as a real crossing.
                # This is what catches a taxi's box briefly shrinking for a
                # frame or two (occlusion, a turn, a detection blip) without
                # either missing a genuine crossing or logging a false one.
                if state.confirmed_side is None:
                    state.confirmed_side = curr_side  # first-ever reading is the baseline
                elif not state.logged:
                    same_side = (curr_side > 0) == (state.confirmed_side > 0)
                    if same_side:
                        state.pending_side = None
                        state.pending_count = 0
                    else:
                        if state.pending_side is not None and (curr_side > 0) == (state.pending_side > 0):
                            state.pending_count += 1
                        else:
                            state.pending_side = curr_side
                            state.pending_count = 1

                        if debug:
                            print(
                                f"[debug-line] frame={frame_idx} track={vehicle.track_id} "
                                f"center={center} raw_side={raw_side:.1f} "
                                f"curr_side(smoothed)={curr_side:.1f} "
                                f"pending {state.pending_count}/{config.CONFIRM_FRAMES}"
                            )

                        if state.pending_count >= config.CONFIRM_FRAMES:
                            direction = direction_from_crossing(state.confirmed_side, curr_side)
                            if direction:
                                logged_info = _log_event(vehicle, state, direction, raw_frame, source_label)
                                session_summary.append(logged_info)
                                state.logged = True
                                events_logged += 1
                            state.confirmed_side = curr_side
                            state.pending_side = None
                            state.pending_count = 0

                state.last_side = curr_side

            _evict_stale_tracks(tracks, frame_idx)

            if show_preview:
                try:
                    cv2.imshow("ANPR debug", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error as e:
                    print(f"[pipeline] preview window unavailable ({e}); continuing headless")
                    show_preview = False

    finally:
        cap.release()
        if show_preview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        print(
            f"[pipeline] done -- {frame_idx} frames processed, "
            f"{len(seen_track_ids)} vehicle track(s) seen, "
            f"{events_logged} event(s) logged"
        )
        if session_summary:
            print("[pipeline] summary:")
            for info in session_summary:
                valid_tag = " [format-verified]" if info.get("format_valid") else ""
                print(
                    f"  - {info['direction']:5s} | vehicle_type: {info['vehicle_type']:10s} "
                    f"| plate_number: {info['plate_number']:12s} | conf: {info['confidence']:.2f}"
                    f"{valid_tag}"
                )
        if seen_track_ids and events_logged == 0:
            print(
                "[pipeline] vehicles were detected but none crossed GATE_LINE -- "
                "your line coordinates in config.py likely don't match this "
                "video. Run 'python inspect_frame.py traffic.mp4' to pick "
                "correct coordinates."
            )
        elif not seen_track_ids:
            print(
                "[pipeline] no vehicles were detected at all in this video -- "
                "check the footage actually shows vehicles clearly, or lower "
                "DETECTION_CONFIDENCE_THRESHOLD in config.py."
            )


def _log_event(vehicle, state: TrackState, direction: str, frame, source_label: str):
    plate_text, confidence, votes = state.voter.best()
    format_valid = state.voter.best_is_format_valid()
    if plate_text is None:
        plate_text = "UNVERIFIED" if not config.ENABLE_PLATE_DETECTION else "UNREADABLE"
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    snapshot_path = os.path.join(
        config.SNAPSHOT_DIR, f"{timestamp_str}_{plate_text}_{direction}.jpg"
    )
    cv2.imwrite(snapshot_path, frame)

    session = get_session()
    try:
        event = VehicleEvent(
            plate_number=plate_text,
            vehicle_type=vehicle.class_name,
            direction=direction,
            confidence=confidence,
            track_id=vehicle.track_id,
            snapshot_path=snapshot_path,
            source=source_label,
        )
        session.add(event)
        session.commit()
        valid_flag = " [format-verified]" if format_valid else ""
        print(f"[pipeline] logged {direction}: {plate_text} (conf={confidence:.2f}){valid_flag}")
    finally:
        session.close()

    return {
        "direction": direction,
        "vehicle_type": vehicle.class_name,
        "plate_number": plate_text,
        "confidence": confidence,
        "format_valid": format_valid,
    }


def _evict_stale_tracks(tracks: dict, frame_idx: int):
    stale = [
        tid for tid, s in tracks.items()
        if frame_idx - s.last_seen_frame > config.TRACK_TIMEOUT_FRAMES
    ]
    for tid in stale:
        del tracks[tid]