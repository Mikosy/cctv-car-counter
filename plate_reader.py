"""
Plate detection (YOLOv8, fine-tuned model) + OCR (EasyOCR) + a small voting
scheme so one noisy frame can't ruin an otherwise-clear reading. A vehicle
crossing the gate is seen for dozens of frames; we read the plate on a
subset of them and let the most consistent reading win.
"""

import re

import cv2
import easyocr
from ultralytics import YOLO

import config

# Loaded lazily, and only if plate detection is enabled -- this is what lets
# you run the pipeline in smoke-test mode (config.ENABLE_PLATE_DETECTION =
# False) before models/plate_yolov8.pt exists.
_plate_model = None
_ocr_reader = None


def _get_plate_model():
    global _plate_model
    if _plate_model is None:
        _plate_model = YOLO(config.PLATE_MODEL_PATH)
    return _plate_model


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader

# Very loose plate-shaped cleanup: uppercase, strip anything that isn't
# alphanumeric. Tighten this to your country's actual plate format once you
# know it (e.g. Nigerian plates are typically ABC-123-DE) to reject junk reads.
_CLEAN_RE = re.compile(r"[^A-Z0-9]")


def _clean_plate_text(raw: str) -> str:
    return _CLEAN_RE.sub("", raw.upper())


_PLATE_FORMAT_RE = re.compile(config.PLATE_FORMAT_REGEX)


def matches_plate_format(text: str) -> bool:
    """True if `text` has the shape of a real Nigerian plate (LLL-DDD-LL)."""
    return bool(text) and bool(_PLATE_FORMAT_RE.match(text))


def detect_plate_crop(vehicle_crop):
    """Returns the tightest plate crop found inside a vehicle crop, or None."""
    if not config.ENABLE_PLATE_DETECTION:
        return None
    if vehicle_crop.size == 0:
        return None

    results = _get_plate_model()(vehicle_crop, conf=config.DETECTION_CONFIDENCE_THRESHOLD, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    # If more than one "plate" is found (shouldn't usually happen), take the
    # highest-confidence one.
    best = boxes[boxes.conf.argmax()]
    x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().astype(int)
    return vehicle_crop[y1:y2, x1:x2]


def read_plate_text(plate_crop):
    """Returns (text, confidence) or (None, 0.0) if nothing readable."""
    if not config.ENABLE_PLATE_DETECTION:
        return None, 0.0
    if plate_crop is None or plate_crop.size == 0:
        return None, 0.0

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    results = _get_ocr_reader().readtext(gray)
    if not results:
        return None, 0.0

    # Concatenate all detected text fragments (plates sometimes split into
    # two OCR boxes), weight confidence by the lowest of the fragments.
    text = "".join(r[1] for r in results)
    confidence = min(r[2] for r in results)
    return _clean_plate_text(text), confidence


class PlateVoter:
    """
    Tracks OCR readings for one tracked vehicle and picks a winner.

    Deliberately NOT majority-vote-by-exact-string: OCR noise means the same
    physical plate rarely produces an identical string twice across frames
    (e.g. '171TX', 'WS17ZTX', 'WANT1TX' for one real plate). Exact-match
    voting tends to crown whatever noisy string happens to repeat, even if a
    single much cleaner high-confidence read exists elsewhere. Instead, we
    just keep the single highest-confidence reading seen -- a 0.9+ confidence
    read is almost always a cleaner frame than a pile of 0.1-0.3 reads,
    regardless of whether its exact text repeats.
    """

    def __init__(self):
        self._best_text = None
        self._best_confidence = 0.0
        self._reading_count = 0  # how many attempts cleared MIN_PLATE_CONFIDENCE
        self._best_valid_text = None       # best reading that also matches the plate format
        self._best_valid_confidence = 0.0

    def add(self, text: str, confidence: float):
        if not text or confidence < config.MIN_PLATE_CONFIDENCE:
            return
        self._reading_count += 1
        if confidence > self._best_confidence:
            self._best_confidence = confidence
            self._best_text = text
        if config.ENFORCE_PLATE_FORMAT and matches_plate_format(text):
            if confidence > self._best_valid_confidence:
                self._best_valid_confidence = confidence
                self._best_valid_text = text

    def best(self):
        """
        Returns (plate_text, confidence, reading_count) or (None, 0, 0).
        Prefers a format-valid reading over a higher-confidence one that
        doesn't match a real plate's shape; falls back to the best reading
        of any shape if nothing ever matched the format.
        """
        if config.ENFORCE_PLATE_FORMAT and self._best_valid_text is not None:
            return self._best_valid_text, self._best_valid_confidence, self._reading_count
        return self._best_text, self._best_confidence, self._reading_count

    def best_is_format_valid(self) -> bool:
        """True if the winning reading from best() passed format validation."""
        return config.ENFORCE_PLATE_FORMAT and self._best_valid_text is not None

    def has_enough_votes(self) -> bool:
        return self._reading_count >= config.MIN_VOTES_TO_LOG