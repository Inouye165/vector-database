# BBox Crop Tester (Standalone Tool)

Local-only helper to detect people/animals in a folder of photos, draw bounding boxes, save cropped detections, and write a manifest for later use. This lives under `tools/bbox_crop_tester` and does not touch the main app.

## Layout

- `app.py` – Tkinter desktop UI entry point
- `detector.py` – reusable detection + cropping functions
- `models.py` – typed result models and shared types
- `io_utils.py` – file/path helpers, manifests, logging
- `requirements.txt` – dependencies for this standalone tool
- `output/annotated/` – annotated full images with boxes
- `output/crops/` – cropped detections
- `output/logs/` – run logs and errors

## Install (Windows / local)

From the repo root:

```bash
cd tools/bbox_crop_tester
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

The first run will download the YOLO model weights (cached afterward).

### Dependency pinning + lockfile strategy

- `requirements.in` contains top-level dependencies.
- `requirements.txt` is pinned to validated versions for normal installs.
- `requirements.lock.txt` is the reproducible lock snapshot used for deterministic installs.

Install from lock snapshot:

```bash
pip install -r requirements.lock.txt
```

Update workflow:

1. Edit `requirements.in` when you want to add/remove top-level deps.
2. Resolve and validate versions in a fresh venv.
3. Update both `requirements.txt` and `requirements.lock.txt`.
4. Run tests before shipping.

## Run the UI

From `tools/bbox_crop_tester` (with the venv activated):

```bash
python app.py
```

This opens a Tkinter window with:

- Folder picker
- "All images" vs "First N images"
- N input
- Confidence threshold input
- Profile selector (`fast`, `balanced`, `high_recall`)
- Resume checkbox for crash/cancel recovery
- Cancel button for long runs
- Run button
- Status/progress text
- Summary counts and scrollable list of crop filenames

## Core reusable APIs

The reusable entry points are available directly from this package.

### Import from another project

From the repository root (or with this folder on your `PYTHONPATH`):

```python
from pathlib import Path
from tools.bbox_crop_tester import (
    detect_and_crop_folder,
    detect_and_crop_image,
)
from tools.bbox_crop_tester.detector import DetectionBackend

backend = DetectionBackend(model_path="yolov8m.pt")
batch_result = detect_and_crop_folder(
    input_dir=Path("/path/to/images"),
    output_dir=Path("tools/bbox_crop_tester"),
    max_images=100,
    confidence_threshold=0.35,
    backend=backend,
)
```

- `detect_and_crop_folder(...) -> BatchDetectionResult`
- `detect_and_crop_image(...) -> ImageDetectionResult`

Results contain per-image detections, crop/annotated paths, bounding boxes, labels, confidences, timestamps, and run ids.

### Run the UI function programmatically

```python
from tools.bbox_crop_tester.app import run_app

run_app()
```

## Outputs and manifests

All outputs are written under `tools/bbox_crop_tester/output`:

- `output/annotated/` – annotated copies of input images
- `output/crops/` – one image per detection
- `output/logs/` – log files per run
- `output/manifest.csv` – flat tabular manifest
- `output/manifest.json` – same data as JSON list

Each manifest row includes:

- source image path
- crop file path
- annotated image path
- detected label
- confidence
- bounding box (x1, y1, x2, y2), width, height
- timestamp
- run id

## Manual test checklist

1. Activate the venv and run the UI (`python app.py`).
2. Click "Choose Folder" and select a folder of photos (jpg/jpeg/png/webp/bmp).
3. Select "All images" or "First N images" and set N.
4. Adjust the confidence threshold (e.g. `0.35`).
5. Click "Run scan" and wait for status to change to "Done."
6. Verify the summary line shows images scanned, detections, and crops.
7. Confirm each row shows the source image, color-coded boxes in the preview, and per-box details on the right.
8. Check `output/annotated/` and `output/crops/` for saved files.
9. Open `output/manifest.csv` and `output/manifest.json` and confirm metadata is present.
10. Intentionally add a bad/unsupported image in the folder and confirm the run still completes and an error is logged under `output/logs/`.

## Automated tests

Run the standalone tests from `tools/bbox_crop_tester`:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Crash recovery / resume

For long scans, the app writes checkpoint state after each image to:

- `output/state/resume_state.json`

If a crash/cancel happens:

1. Reopen the app.
2. Keep the same input folder.
3. Ensure "Resume after crash/cancel" is checked.
4. Run scan again.

The batch resumes from the next unprocessed image.

## Quality checks (lint/type/security/audit)

Install dev tools:

```bash
pip install -r requirements-dev.txt
```

Run checks:

```bash
ruff check .
mypy .
bandit -r .
pip-audit -r requirements.lock.txt
```

Current coverage includes:

- bbox IoU and duplicate-detection helper logic
- invalid input and confidence validation for batch scan
- filename sanitization
- manifest CSV/JSON generation

## Security and robustness notes

- The app is local-only and does not expose any network endpoints.
- Image loading applies EXIF orientation consistently to avoid mismatch between detection and display.
- `load_image` rejects invalid image files and sets a max pixel guard to reduce decompression-bomb risk.
- Output filenames are sanitized and uniquely generated to avoid unsafe characters and silent overwrites.
- Batch processing is fail-soft: one bad image does not terminate the full run.
- Logging is written per run under `output/logs/` as JSON lines for easier parsing.
- Detection pipeline supports optional TTA horizontal-flip pass in `high_recall` profile.

Example log line:

```json
{"timestamp":"2026-03-31T15:00:00+00:00","level":"INFO","logger":"bbox_crop_tester","message":"Scan completed","event":"scan_completed","run_id":"...","image_path":"C:/photos"}
```

## Known limitations

- Detection quality depends on model choice (`yolov8n.pt` is fast but can miss hard cases). For higher recall, use a larger model (for example `yolov8m.pt`) via `DetectionBackend`.
- This is not a sandboxed or multi-user service; it assumes trusted local usage.

