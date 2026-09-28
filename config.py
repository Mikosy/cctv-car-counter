"""
Central configuration. Every other module reads from here, including the
future FastAPI layer -- so tuning the gate line, thresholds, or DB location
only ever happens in one place.
"""

# --- Video sources -----------------------------------------------------
# "live" mode reads from this RTSP/HTTP URL (or an int like 0 for a USB cam).
LIVE_SOURCE = "rtsp://username:password@camera-ip:554/stream1"

# --- Models --------------------------------------------------------------
# General-purpose COCO-pretrained model, detects vehicles out of the box.
VEHICLE_MODEL_PATH = "yolov8n.pt"

# COCO class ids that count as "a vehicle" for this project.
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# You need a plate-specific model here -- COCO does NOT detect plates.
# Fine-tune a small YOLOv8 model on a license-plate dataset (several open
# datasets exist) and point this at the resulting .pt file. Until you do,
# plate detection will not work -- see README.md.
PLATE_MODEL_PATH = "models/plate_yolov8.pt"

# Set False to smoke-test vehicle tracking + line crossing without any plate
# model in place yet -- crossings get logged as "UNVERIFIED" so you can
# confirm the rest of the pipeline works before plugging in real plate
# detection. Set True once models/plate_yolov8.pt exists.
# ENABLE_PLATE_DETECTION = False
ENABLE_PLATE_DETECTION = True

# --- Detection cadence / performance -------------------------------------
# Run the (expensive) plate detector + OCR only every N frames per track;
# vehicle tracking runs every frame regardless (it's cheap).
PLATE_DETECT_EVERY_N_FRAMES = 3

# Resize incoming frames to this width before inference. Smaller = faster.
# Set to None to disable resizing.
INFERENCE_FRAME_WIDTH = 960

# Drop a track's in-memory state if it hasn't been seen for this many frames
# (handles cars that leave the frame without crossing the line, e.g. parking).
TRACK_TIMEOUT_FRAMES = 150

# --- Gate line -------------------------------------------------------------
# Two points (x, y) in pixel space (relative to INFERENCE_FRAME_WIDTH),
# defining the virtual line vehicles must cross to be logged.
# Draw this by eye on one saved frame from your camera -- see README.md.
# GATE_LINE = ((0, 375), (960, 375))
GATE_LINE = ((0, 300), (960, 300))

# Which side of the line counts as "entry" vs "exit". "positive" / "negative"
# refers to the sign of the cross product used in pipeline.py -- flip this
# if entries and exits come out swapped for your camera orientation.
POSITIVE_SIDE_MEANS = "exit"  # or "entry"

# How much to smooth the gate-line side calculation over time (0-1). Higher
# = more resistant to a single noisy/glitchy detection frame flipping the
# result, but slower to register a genuinely fast crossing. This is what
# stops a one-frame box-detection glitch from being logged as a real
# entry/exit -- see pipeline.py for how it's applied.
GATE_SMOOTHING_ALPHA = 0.6

# How many consecutive frames the (smoothed) side must stay flipped before a
# crossing is confirmed and logged. This is the debounce that stops a brief
# 1-2 frame detection glitch or occlusion from being logged as a real
# entry/exit. Higher = more resistant to false positives, but slower to
# register genuinely fast crossings -- tune against your actual camera's fps.
CONFIRM_FRAMES = 3

# --- OCR / voting ----------------------------------------------------------
MIN_PLATE_CONFIDENCE = 0.35        # discard OCR reads below this
MIN_VOTES_TO_LOG = 2               # need at least this many agreeing reads
DETECTION_CONFIDENCE_THRESHOLD = 0.5

# Nigerian plates follow LLL-DDD-LL: 3 letters (LGA code), 3 digits, 2
# letters -- e.g. "ABC-123-DE", cleaned to "ABC123DE". When enabled, a
# reading that matches this shape is preferred over a higher-confidence
# reading that doesn't even look like a real plate (garbage text a generic
# OCR model sometimes produces with surprisingly high confidence). Falls
# back to the best reading of any shape if nothing ever matches for a given
# vehicle, so a real plate is never dropped just because OCR mangled it.
ENFORCE_PLATE_FORMAT = True
PLATE_FORMAT_REGEX = r"^[A-Z]{3}\d{3}[A-Z]{2}$"

# --- Storage ----------------------------------------------------------------
DB_URL = "sqlite:///anpr.db"       # swap for a postgres:// URL later if needed
SNAPSHOT_DIR = "snapshots"         # cropped plate + full frame saved per event