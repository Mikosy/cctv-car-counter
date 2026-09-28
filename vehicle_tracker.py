"""
Wraps Ultralytics YOLOv8 detection + ByteTrack tracking. Emits one record
per tracked vehicle per frame, with a stable track_id that persists across
frames as long as the vehicle stays (roughly) visible.
"""

from dataclasses import dataclass

from ultralytics import YOLO

import config


@dataclass
class TrackedVehicle:
    track_id: int
    box: tuple          # (x1, y1, x2, y2) in frame pixel coords
    class_name: str
    confidence: float


class VehicleTracker:
    def __init__(self, model_path: str = config.VEHICLE_MODEL_PATH):
        self.model = YOLO(model_path)

    def update(self, frame) -> list[TrackedVehicle]:
        """
        Run detection + tracking on a single frame. Call this once per frame,
        in order -- persist=True keeps track identities alive between calls.
        """
        results = self.model.track(
            frame,
            persist=True,
            classes=list(config.VEHICLE_CLASS_IDS.keys()),
            conf=config.DETECTION_CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        vehicles = []
        result = results[0]
        if result.boxes is None or result.boxes.id is None:
            return vehicles  # nothing tracked this frame (e.g. warm-up frame)

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for box, track_id, cls_id, conf in zip(boxes, ids, classes, confs):
            vehicles.append(
                TrackedVehicle(
                    track_id=int(track_id),
                    box=tuple(box.astype(int)),
                    class_name=config.VEHICLE_CLASS_IDS.get(int(cls_id), "vehicle"),
                    confidence=float(conf),
                )
            )
        return vehicles
