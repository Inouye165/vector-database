from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable, Iterable, List, Optional, Protocol, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

try:
    from .io_utils import (
        clear_resume_state,
        ensure_output_subdirs,
        generate_run_id,
        iter_images,
        load_resume_state,
        load_image,
        save_resume_state,
        safe_filename,
    )
    from .models import (
        BatchDetectionResult,
        BoundingBox,
        Detection,
        ImageDetectionResult,
    )
except ImportError:
    from io_utils import (
        clear_resume_state,
        ensure_output_subdirs,
        generate_run_id,
        iter_images,
        load_resume_state,
        load_image,
        save_resume_state,
        safe_filename,
    )
    from models import (
        BatchDetectionResult,
        BoundingBox,
        Detection,
        ImageDetectionResult,
    )


DEFAULT_CLASSES: Set[str] = {
    "person",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}


@dataclass
class DetectionBackend:
    model_path: str = "yolov8n.pt"

    def __post_init__(self) -> None:
        try:
            self._model = YOLO(self.model_path)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to initialize detection backend: {exc}") from exc

    @property
    def model(self) -> YOLO:
        return self._model


class DetectorBackendProtocol(Protocol):
    @property
    def model(self) -> YOLO: ...


def _filter_classes(
    model: YOLO, include_classes: Optional[Set[str]]
) -> Tuple[Set[int], Set[str]]:
    names = model.names
    label_to_id = {str(v).lower(): k for k, v in names.items()}

    if include_classes is None:
        include_classes = DEFAULT_CLASSES

    include_lower = {c.lower() for c in include_classes}
    supported_ids: Set[int] = set()
    supported_labels: Set[str] = set()
    for label in include_lower:
        if label in label_to_id:
            cid = label_to_id[label]
            supported_ids.add(cid)
            supported_labels.add(label)
    return supported_ids, supported_labels


def _run_detection_on_image(
    backend: DetectorBackendProtocol,
    image: Image.Image,
    confidence_threshold: float,
    include_classes: Optional[Set[str]],
    enable_person_second_pass: bool = True,
    enable_tta_flip: bool = False,
) -> Iterable[Tuple[BoundingBox, str, float]]:
    class_ids, supported_labels = _filter_classes(backend.model, include_classes)
    if not class_ids:
        return []

    np_image = np.array(image)
    try:
        results = backend.model.predict(
            np_image,
            conf=confidence_threshold,
            classes=sorted(class_ids),
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Model inference failed: {exc}") from exc

    detections: List[Tuple[BoundingBox, str, float]] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in class_ids:
                continue
            score = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            label = str(result.names.get(cls_id, str(cls_id)))
            detections.append((BoundingBox(x1, y1, x2, y2), label, score))

    # Optional second pass for person-only detections to better split/group people photos.
    if enable_person_second_pass and "person" in supported_labels:
        person_cls_id = next(
            (cid for cid, name in backend.model.names.items() if str(name).lower() == "person"),
            None,
        )
        if person_cls_id is not None:
            try:
                person_results = backend.model.predict(
                    np_image,
                    conf=max(0.15, confidence_threshold - 0.12),
                    classes=[int(person_cls_id)],
                    verbose=False,
                )
            except Exception:
                person_results = []
            for result in person_results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    score = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    candidate = BoundingBox(x1, y1, x2, y2)
                    if _is_duplicate_detection(candidate, "person", detections):
                        continue
                    detections.append((candidate, "person", score))

    # Optional test-time augmentation pass using horizontal flip.
    if enable_tta_flip:
        flipped = np.fliplr(np_image)
        try:
            tta_results = backend.model.predict(
                flipped,
                conf=max(0.1, confidence_threshold - 0.05),
                classes=sorted(class_ids),
                verbose=False,
            )
        except Exception:
            tta_results = []
        img_w = image.size[0]
        for result in tta_results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                label = str(result.names.get(cls_id, str(cls_id)))
                score = float(box.conf[0])
                fx1, fy1, fx2, fy2 = [int(v) for v in box.xyxy[0].tolist()]
                x1 = img_w - fx2
                x2 = img_w - fx1
                candidate = BoundingBox(x1, fy1, x2, fy2)
                if _is_duplicate_detection(candidate, label, detections, iou_threshold=0.6):
                    continue
                detections.append((candidate, label, score))

    return detections


def _is_duplicate_detection(
    candidate_bbox: BoundingBox,
    candidate_label: str,
    existing: List[Tuple[BoundingBox, str, float]],
    iou_threshold: float = 0.7,
) -> bool:
    for bbox, label, _score in existing:
        if label != candidate_label:
            continue
        if _bbox_iou(candidate_bbox, bbox) >= iou_threshold:
            return True
    return False


def _bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1)
    area_b = max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def detect_and_crop_image(
    image_path: Path,
    output_dir: Path,
    max_crops_counter: List[int] | None = None,
    include_classes: Optional[Set[str]] = None,
    confidence_threshold: float = 0.35,
    save_annotated: bool = True,
    save_crops: bool = True,
    backend: Optional[DetectorBackendProtocol] = None,
    run_id: str | None = None,
    enable_person_second_pass: bool = True,
    enable_tta_flip: bool = False,
) -> ImageDetectionResult:
    if confidence_threshold < 0.0 or confidence_threshold > 1.0:
        return ImageDetectionResult(
            source_image=image_path,
            error="confidence_threshold must be between 0.0 and 1.0",
        )

    if backend is None:
        try:
            backend = DetectionBackend()
        except Exception as exc:  # noqa: BLE001
            return ImageDetectionResult(
                source_image=image_path,
                error=f"Failed to initialize detection backend: {exc}",
            )

    annotated_dir, crops_dir, _logs_dir = ensure_output_subdirs(output_dir)
    result = ImageDetectionResult(source_image=image_path)

    try:
        image = load_image(image_path)
    except Exception as exc:  # noqa: BLE001
        result.error = f"Failed to load image: {exc}"
        return result

    width, height = image.size
    try:
        raw_detections = _run_detection_on_image(
            backend=backend,
            image=image,
            confidence_threshold=confidence_threshold,
            include_classes=include_classes,
            enable_person_second_pass=enable_person_second_pass,
            enable_tta_flip=enable_tta_flip,
        )
    except Exception as exc:  # noqa: BLE001
        result.error = f"Detection inference failed: {exc}"
        return result

    if not raw_detections:
        return result

    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    stem = image_path.stem
    effective_run_id = run_id or generate_run_id()
    crop_counter = 0 if max_crops_counter is None else max_crops_counter[0]

    for bbox, label, score in raw_detections:
        x1 = max(0, min(bbox.x1, width - 1))
        y1 = max(0, min(bbox.y1, height - 1))
        x2 = max(0, min(bbox.x2, width))
        y2 = max(0, min(bbox.y2, height))

        if x2 <= x1 or y2 <= y1:
            continue

        clean_bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        detection = Detection(
            label=label,
            confidence=score,
            bbox=clean_bbox,
            source_image=image_path,
            run_id=effective_run_id,
        )

        if save_crops:
            crop_counter += 1
            filename = safe_filename(stem=stem, label=label, suffix=".jpg", counter=crop_counter)
            crop = image.crop(
                (clean_bbox.x1, clean_bbox.y1, clean_bbox.x2, clean_bbox.y2)
            )
            crop_path = crops_dir / filename
            idx = 1
            while crop_path.exists():
                filename = safe_filename(
                    stem=f"{stem}_{idx}",
                    label=label,
                    suffix=".jpg",
                    counter=crop_counter,
                )
                crop_path = crops_dir / filename
                idx += 1
            crop.save(crop_path, format="JPEG", quality=95)
            detection.crop_path = crop_path

        draw.rectangle([clean_bbox.x1, clean_bbox.y1, clean_bbox.x2, clean_bbox.y2], outline="red", width=3)
        draw.text((clean_bbox.x1 + 2, clean_bbox.y1 + 2), f"{label} {score:.2f}", fill="red")

        result.detections.append(detection)

    if save_annotated and result.detections:
        filename = safe_filename(
            stem=stem,
            label="annotated",
            suffix=".jpg",
            counter=0,
        )
        annotated_path = annotated_dir / filename
        idx = 1
        while annotated_path.exists():
            filename = safe_filename(
                stem=f"{stem}_{idx}",
                label="annotated",
                suffix=".jpg",
                counter=0,
            )
            annotated_path = annotated_dir / filename
            idx += 1
        annotated_image.save(annotated_path, format="JPEG", quality=95)
        for det in result.detections:
            det.annotated_image_path = annotated_path

    if max_crops_counter is not None:
        max_crops_counter[0] = crop_counter

    return result


def detect_and_crop_folder(
    input_dir: Path,
    output_dir: Path,
    max_images: Optional[int] = None,
    include_classes: Optional[Set[str]] = None,
    confidence_threshold: float = 0.35,
    save_annotated: bool = True,
    save_crops: bool = True,
    backend: Optional[DetectorBackendProtocol] = None,
    progress_callback: Optional[Callable[[int, int, Path], None]] = None,
    cancel_event: Optional[Event] = None,
    resume_from_last: bool = False,
    enable_person_second_pass: bool = True,
    enable_tta_flip: bool = False,
) -> BatchDetectionResult:
    run_id = generate_run_id()
    if not input_dir.exists() or not input_dir.is_dir():
        return BatchDetectionResult(
            input_dir=input_dir,
            output_dir=output_dir,
            run_id=run_id,
            images_scanned=0,
            total_detections=0,
            total_crops=0,
            image_results=[],
            errors=[f"Input directory does not exist or is not a directory: {input_dir}"],
        )
    if max_images is not None and max_images <= 0:
        return BatchDetectionResult(
            input_dir=input_dir,
            output_dir=output_dir,
            run_id=run_id,
            images_scanned=0,
            total_detections=0,
            total_crops=0,
            image_results=[],
            errors=["max_images must be greater than 0 when provided"],
        )
    if confidence_threshold < 0.0 or confidence_threshold > 1.0:
        return BatchDetectionResult(
            input_dir=input_dir,
            output_dir=output_dir,
            run_id=run_id,
            images_scanned=0,
            total_detections=0,
            total_crops=0,
            image_results=[],
            errors=["confidence_threshold must be between 0.0 and 1.0"],
        )

    if backend is None:
        try:
            backend = DetectionBackend()
        except Exception as exc:  # noqa: BLE001
            return BatchDetectionResult(
                input_dir=input_dir,
                output_dir=output_dir,
                run_id=run_id,
                images_scanned=0,
                total_detections=0,
                total_crops=0,
                image_results=[],
                errors=[f"Failed to initialize detection backend: {exc}"],
            )

    images = list(iter_images(input_dir))
    if max_images is not None:
        images = images[:max_images]

    start_index = 0
    if resume_from_last:
        state = load_resume_state(output_dir)
        if state:
            prior_input = state.get("input_dir")
            if prior_input == str(input_dir):
                start_index = max(0, int(state.get("last_completed_index", -1)) + 1)

    image_results: List[ImageDetectionResult] = []
    errors: List[str] = []
    total_detections = 0
    total_crops = 0
    crops_counter = [0]

    for idx, image_path in enumerate(images[start_index:], start=start_index):
        if cancel_event is not None and cancel_event.is_set():
            errors.append("Scan cancelled by user.")
            break
        res = detect_and_crop_image(
            image_path=image_path,
            output_dir=output_dir,
            max_crops_counter=crops_counter,
            include_classes=include_classes,
            confidence_threshold=confidence_threshold,
            save_annotated=save_annotated,
            save_crops=save_crops,
            backend=backend,
            run_id=run_id,
            enable_person_second_pass=enable_person_second_pass,
            enable_tta_flip=enable_tta_flip,
        )
        image_results.append(res)
        if not res.successful:
            errors.append(f"{image_path}: {res.error}")
        else:
            total_detections += len(res.detections)
        save_resume_state(
            output_dir,
            {
                "run_id": run_id,
                "input_dir": str(input_dir),
                "last_completed_index": idx,
                "last_completed_image": str(image_path),
            },
        )
        if progress_callback is not None:
            progress_callback(idx + 1, len(images), image_path)

    total_crops = crops_counter[0]

    batch = BatchDetectionResult(
        input_dir=input_dir,
        output_dir=output_dir,
        run_id=run_id,
        images_scanned=len(image_results),
        total_detections=total_detections,
        total_crops=total_crops,
        image_results=image_results,
        errors=errors,
    )

    # Clear state only on complete successful finish.
    if not errors:
        clear_resume_state(output_dir)

    return batch

