from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set


SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: BoundingBox
    source_image: Path
    crop_path: Optional[Path] = None
    annotated_image_path: Optional[Path] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    run_id: str = ""


@dataclass
class ImageDetectionResult:
    source_image: Path
    detections: List[Detection] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def successful(self) -> bool:
        return self.error is None


@dataclass
class BatchDetectionResult:
    input_dir: Path
    output_dir: Path
    run_id: str
    images_scanned: int
    total_detections: int
    total_crops: int
    image_results: List[ImageDetectionResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

