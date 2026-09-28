# Entry/exit ANPR tracker

Detects vehicles crossing a gate, reads their plates, and logs entry/exit
events to a database. Same pipeline code runs on a live RTSP feed and on
offline .mp4 files -- only the `--source` argument changes.

## Setup

```bash
pip install -r requirements.txt --break-system-packages   # if on a managed Python env
```

`yolov8n.pt` (vehicle detection) downloads automatically on first run.

### You still need to provide

1. **A plate detection model** at `models/plate_yolov8.pt`. The general
   vehicle model does NOT detect plates -- you need a model fine-tuned on
   license plates specifically. Options, roughly fastest to best:
   - Run `python download_plate_model.py` to grab a pretrained generic
     plate detector as a baseline (needs `pip install huggingface_hub`).
     It works but wasn't trained on Nigerian plates -- expect it to
     underperform there.
   - Fine-tune a small YOLOv8 model yourself on an open license-plate
     dataset (search Roboflow Universe for "license plate detection" --
     several ready-to-use datasets exist you can train on directly), then
     ideally fine-tune again on real Nigerian plate samples once you have
     some, since font, format, and state stickers differ from US/EU plates.
   - Or, to test everything else first: set `ENABLE_PLATE_DETECTION = False`
     in `config.py`. The pipeline will log every gate crossing as
     `"UNVERIFIED"` instead of a real plate, letting you confirm vehicle
     tracking and entry/exit direction work correctly before plate
     detection is even in the picture.

2. **The gate line coordinates** in `config.py` (`GATE_LINE`). Grab one
   frame from your camera, open it in any image viewer, and note two pixel
   points that form a line across the gate the cars must cross. Adjust
   `POSITIVE_SIDE_MEANS` if entries/exits come out swapped.

3. **Your RTSP URL** in `config.py` (`LIVE_SOURCE`), if using live mode.

## Running

```bash
# Offline, against a test video file
python main.py --source test_footage.mp4 --preview

# Live, against the camera configured in config.py
python main.py --source live
```

Drop the `--preview` flag when running headless on a server.

## Where things are logged

- Every logged event goes into the `vehicle_events` table (SQLite file
  `anpr.db` by default -- see `db.py` / `config.DB_URL`).
- A snapshot image is saved to `snapshots/` for every logged event, named
  with timestamp + plate + direction, so you can manually audit any
  low-confidence reads.

## Tuning accuracy

- `ENFORCE_PLATE_FORMAT` in `config.py` (on by default) rejects OCR readings
  that don't match the shape of a real Nigerian plate (3 letters, 3 digits,
  2 letters -- `PLATE_FORMAT_REGEX`) when picking the winning reading for a
  vehicle, even if a malformed reading had higher raw OCR confidence. Falls
  back to the best reading of any shape if nothing ever matched the format,
  so a real plate is never dropped just because OCR mangled it on every
  frame. Logged/summary output tags a winning reading `[format-verified]`
  when it passed this check.
- `MIN_VOTES_TO_LOG` / `MIN_PLATE_CONFIDENCE` in `config.py` control how
  many agreeing OCR reads are required before an event is logged. Raise
  these if you're seeing wrong plates logged; lower them if legitimate
  vehicles aren't getting logged at all.
- The `_clean_plate_text` regex in `plate_reader.py` is intentionally loose.
  Once you know your exact plate format, tighten it (or add a format
  validation step) to reject obviously-malformed OCR output before it's
  even counted as a vote.

## Performance on CPU-only hardware

If this runs too slowly on your laptop:
- Lower `INFERENCE_FRAME_WIDTH` in `config.py`.
- Raise `PLATE_DETECT_EVERY_N_FRAMES` (plate OCR is the expensive step, not
  vehicle tracking).
- Export both YOLO models to OpenVINO format for a meaningful CPU speedup:
  `model.export(format="openvino")`, then point `VEHICLE_MODEL_PATH` /
  `PLATE_MODEL_PATH` at the exported `.xml` files.

## Wrapping in FastAPI later

`db.py` and `pipeline.py` have no CLI or web dependencies -- a future
FastAPI app can `import db` to query `VehicleEvent` rows directly for a
`GET /events` endpoint, and `import pipeline` to kick off `pipeline.process()`
as a background task per camera. `main.py` is just a thin CLI wrapper around
the same `pipeline.process()` call and won't need to change.