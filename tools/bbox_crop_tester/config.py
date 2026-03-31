from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ProfileName = Literal["fast", "balanced", "high_recall"]


@dataclass(frozen=True)
class ScanProfile:
    name: ProfileName
    model_path: str
    confidence_threshold: float
    enable_person_second_pass: bool
    enable_tta_flip: bool


PROFILES: dict[ProfileName, ScanProfile] = {
    "fast": ScanProfile(
        name="fast",
        model_path="yolov8n.pt",
        confidence_threshold=0.35,
        enable_person_second_pass=False,
        enable_tta_flip=False,
    ),
    "balanced": ScanProfile(
        name="balanced",
        model_path="yolov8s.pt",
        confidence_threshold=0.25,
        enable_person_second_pass=True,
        enable_tta_flip=False,
    ),
    "high_recall": ScanProfile(
        name="high_recall",
        model_path="yolov8m.pt",
        confidence_threshold=0.2,
        enable_person_second_pass=True,
        enable_tta_flip=True,
    ),
}

