from __future__ import annotations

import csv
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from .models import (
        SUPPORTED_IMAGE_EXTENSIONS,
        BatchDetectionResult,
        Detection,
        ImageDetectionResult,
    )
except ImportError:
    from models import (
        SUPPORTED_IMAGE_EXTENSIONS,
        BatchDetectionResult,
        Detection,
        ImageDetectionResult,
    )


LOGGER_NAME = "bbox_crop_tester"


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("run_id", "image_path", "event"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=True)


def setup_logging(base_dir: Path) -> logging.Logger:
    logs_dir = base_dir / "output" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)

    # Replace handlers so each launch gets its own run log file.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(JsonLogFormatter())
    logger.addHandler(fh)

    return logger


def generate_run_id() -> str:
    return uuid.uuid4().hex


def is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def iter_images(directory: Path) -> Iterable[Path]:
    if not directory.is_dir():
        return []
    yield from sorted(
        (p for p in directory.iterdir() if is_supported_image(p)),
        key=lambda p: p.name,
    )


def safe_filename(stem: str, label: str, suffix: str, counter: int) -> str:
    safe_stem = "".join(c if c.isalnum() or c in "-._" else "_" for c in stem)
    safe_label = "".join(c if c.isalnum() or c in "-._" else "_" for c in label)
    unique = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{counter:04d}"
    return f"{safe_stem}__{safe_label}__{unique}{suffix}"


def ensure_output_subdirs(base_dir: Path) -> Tuple[Path, Path, Path]:
    annotated_dir = base_dir / "output" / "annotated"
    crops_dir = base_dir / "output" / "crops"
    logs_dir = base_dir / "output" / "logs"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return annotated_dir, crops_dir, logs_dir


def state_file_path(base_dir: Path) -> Path:
    state_dir = base_dir / "output" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "resume_state.json"


def save_resume_state(base_dir: Path, payload: dict[str, Any]) -> None:
    path = state_file_path(base_dir)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(temp_path, path)


def load_resume_state(base_dir: Path) -> dict[str, Any] | None:
    path = state_file_path(base_dir)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clear_resume_state(base_dir: Path) -> None:
    path = state_file_path(base_dir)
    if path.exists():
        path.unlink()


def save_manifest(batch: BatchDetectionResult, base_dir: Path) -> None:
    manifest_dir = base_dir / "output"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    csv_path = manifest_dir / "manifest.csv"
    json_path = manifest_dir / "manifest.json"
    meta_path = manifest_dir / "run_metadata.json"

    rows = []
    for img_res in batch.image_results:
        for det in img_res.detections:
            rows.append(
                {
                    "run_id": det.run_id or batch.run_id,
                    "timestamp": det.timestamp.isoformat(),
                    "source_image_path": str(det.source_image),
                    "crop_file_path": str(det.crop_path) if det.crop_path else "",
                    "annotated_image_path": (
                        str(det.annotated_image_path)
                        if det.annotated_image_path
                        else ""
                    ),
                    "label": det.label,
                    "confidence": det.confidence,
                    "x1": det.bbox.x1,
                    "y1": det.bbox.y1,
                    "x2": det.bbox.x2,
                    "y2": det.bbox.y2,
                    "width": det.bbox.width,
                    "height": det.bbox.height,
                }
            )

    fieldnames = [
        "run_id",
        "timestamp",
        "source_image_path",
        "crop_file_path",
        "annotated_image_path",
        "label",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    metadata = {
        "run_id": batch.run_id,
        "input_dir": str(batch.input_dir),
        "output_dir": str(batch.output_dir),
        "images_scanned": batch.images_scanned,
        "total_detections": batch.total_detections,
        "total_crops": batch.total_crops,
        "errors": batch.errors,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_image(path: Path) -> Image.Image:
    # Guard against image bombs while still allowing reasonably large photos.
    Image.MAX_IMAGE_PIXELS = 120_000_000
    try:
        with Image.open(path) as img:
            # Keep orientation consistent for detection, crops, and UI previews.
            return ImageOps.exif_transpose(img).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unsupported or invalid image file: {path}") from exc

