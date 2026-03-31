from __future__ import annotations

import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable, Iterable, List, Optional, Protocol, Set, Tuple
import time
import psutil
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
    "person", "bird", "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe",
}


@dataclass
class DetectionConfig:
    """Configuration for detection parameters with sensible defaults."""
    confidence_threshold: float = 0.35
    include_classes: Optional[Set[str]] = None
    enable_person_second_pass: bool = True
    enable_tta_flip: bool = False
    max_image_size: int = 2048  # Prevent memory issues with huge images
    batch_size: int = 1  # For future batch processing
    memory_threshold_mb: float = 1024  # Memory usage threshold
    enable_nms: bool = True  # Non-maximum suppression
    nms_iou_threshold: float = 0.45


class MemoryManager:
    """Monitor and manage memory usage during processing."""
    
    def __init__(self, threshold_mb: float = 1024):
        self.threshold_mb = threshold_mb
        self.process = psutil.Process()
    
    def check_memory(self) -> bool:
        """Check if memory usage exceeds threshold."""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        return memory_mb > self.threshold_mb
    
    def force_gc(self) -> None:
        """Force garbage collection."""
        gc.collect()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024


@dataclass
class DetectionBackend:
    """Enhanced detection backend with caching and memory management."""
    model_path: str = "yolov8n.pt"
    config: DetectionConfig = None

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = DetectionConfig()
        
        try:
            self._model = YOLO(self.model_path)
            self._class_cache = {}  # Cache for class filtering
            self.memory_manager = MemoryManager(self.config.memory_threshold_mb)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize detection backend: {exc}") from exc

    @property
    def model(self) -> YOLO:
        return self._model


class DetectorBackendProtocol(Protocol):
    @property
    def model(self) -> YOLO: ...


def _filter_classes_cached(
    model: YOLO, include_classes: Optional[Set[str]], cache: dict
) -> Tuple[Set[int], Set[str]]:
    """Cached version of class filtering for performance."""
    cache_key = tuple(sorted(include_classes)) if include_classes else "default"
    
    if cache_key in cache:
        return cache[cache_key]
    
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
    
    result = (supported_ids, supported_labels)
    cache[cache_key] = result
    return result


def _preprocess_image(
    image: Image.Image, max_size: int = 2048
) -> Tuple[Image.Image, Tuple[int, int]]:
    """Preprocess image for optimal detection performance."""
    original_size = image.size
    
    # Resize if image is too large to prevent memory issues
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return image, original_size


def _scale_bounding_boxes(
    detections: List[Tuple[BoundingBox, str, float]],
    original_size: Tuple[int, int],
    processed_size: Tuple[int, int],
) -> List[Tuple[BoundingBox, str, float]]:
    """Scale bounding boxes back to original image size."""
    if original_size == processed_size:
        return detections
    
    scale_x = original_size[0] / processed_size[0]
    scale_y = original_size[1] / processed_size[1]
    
    scaled_detections = []
    for bbox, label, score in detections:
        scaled_bbox = BoundingBox(
            x1=int(bbox.x1 * scale_x),
            y1=int(bbox.y1 * scale_y),
            x2=int(bbox.x2 * scale_x),
            y2=int(bbox.y2 * scale_y),
        )
        scaled_detections.append((scaled_bbox, label, score))
    
    return scaled_detections


def _run_detection_with_memory_management(
    backend: DetectionBackend,
    image: Image.Image,
    config: DetectionConfig,
) -> Iterable[Tuple[BoundingBox, str, float]]:
    """Enhanced detection with memory management and error recovery."""
    
    # Check memory before processing
    if backend.memory_manager.check_memory():
        backend.memory_manager.force_gc()
    
    # Preprocess image
    processed_image, original_size = _preprocess_image(image, config.max_image_size)
    
    try:
        # Get cached class filters
        class_ids, supported_labels = _filter_classes_cached(
            backend.model, config.include_classes, backend._class_cache
        )
        
        if not class_ids:
            return []
        
        np_image = np.array(processed_image)
        
        # Main detection pass
        detections = _run_single_detection_pass(
            backend, np_image, config.confidence_threshold, class_ids
        )
        
        # Person second pass if enabled
        if config.enable_person_second_pass and "person" in supported_labels:
            person_detections = _run_person_second_pass(
                backend, np_image, config.confidence_threshold
            )
            detections.extend(person_detections)
        
        # TTA pass if enabled
        if config.enable_tta_flip:
            tta_detections = _run_tta_flip_pass(
                backend, np_image, processed_image.size, config.confidence_threshold, class_ids
            )
            detections.extend(tta_detections)
        
        # Apply NMS if enabled
        if config.enable_nms and len(detections) > 1:
            detections = _apply_non_maximum_suppression(detections, config.nms_iou_threshold)
        
        # Scale bounding boxes back to original size
        detections = _scale_bounding_boxes(detections, original_size, processed_image.size)
        
        return detections
        
    except Exception as exc:
        raise RuntimeError(f"Model inference failed: {exc}") from exc
    finally:
        # Clean up memory
        del np_image
        if processed_image != image:
            processed_image.close()
        backend.memory_manager.force_gc()


def _run_single_detection_pass(
    backend: DetectionBackend,
    np_image: np.ndarray,
    confidence_threshold: float,
    class_ids: Set[int],
) -> List[Tuple[BoundingBox, str, float]]:
    """Run a single detection pass."""
    try:
        results = backend.model.predict(
            np_image,
            conf=confidence_threshold,
            classes=sorted(class_ids),
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Detection pass failed: {exc}") from exc
    
    detections = []
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
    
    return detections


def _run_person_second_pass(
    backend: DetectionBackend,
    np_image: np.ndarray,
    confidence_threshold: float,
) -> List[Tuple[BoundingBox, str, float]]:
    """Run person-specific second detection pass."""
    person_cls_id = next(
        (cid for cid, name in backend.model.names.items() if str(name).lower() == "person"),
        None,
    )
    
    if person_cls_id is None:
        return []
    
    try:
        person_results = backend.model.predict(
            np_image,
            conf=max(0.15, confidence_threshold - 0.12),
            classes=[int(person_cls_id)],
            verbose=False,
        )
    except Exception:
        return []
    
    detections = []
    for result in person_results:
        boxes = result.boxes
        if boxes is None:
            continue
            
        for box in boxes:
            score = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append((BoundingBox(x1, y1, x2, y2), "person", score))
    
    return detections


def _run_tta_flip_pass(
    backend: DetectionBackend,
    np_image: np.ndarray,
    image_size: Tuple[int, int],
    confidence_threshold: float,
    class_ids: Set[int],
) -> List[Tuple[BoundingBox, str, float]]:
    """Run test-time augmentation with horizontal flip."""
    flipped = np.fliplr(np_image)
    
    try:
        tta_results = backend.model.predict(
            flipped,
            conf=max(0.1, confidence_threshold - 0.05),
            classes=sorted(class_ids),
            verbose=False,
        )
    except Exception:
        return []
    
    detections = []
    img_w = image_size[0]
    
    for result in tta_results:
        boxes = result.boxes
        if boxes is None:
            continue
            
        for box in boxes:
            cls_id = int(box.cls[0])
            label = str(result.names.get(cls_id, str(cls_id)))
            score = float(box.conf[0])
            fx1, fy1, fx2, fy2 = [int(v) for v in box.xyxy[0].tolist()]
            
            # Flip coordinates back
            x1 = img_w - fx2
            x2 = img_w - fx1
            candidate = BoundingBox(x1, fy1, x2, fy2)
            
            # Check for duplicates with higher threshold for TTA
            if _is_duplicate_detection(candidate, label, detections, iou_threshold=0.6):
                continue
                
            detections.append((candidate, label, score))
    
    return detections


def _apply_non_maximum_suppression(
    detections: List[Tuple[BoundingBox, str, float]],
    iou_threshold: float = 0.45,
) -> List[Tuple[BoundingBox, str, float]]:
    """Apply non-maximum suppression to reduce duplicate detections."""
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence (highest first)
    detections.sort(key=lambda x: x[2], reverse=True)
    
    suppressed = []
    for i, (bbox, label, score) in enumerate(detections):
        if i > 0 and _is_duplicate_detection(bbox, label, detections[:i], iou_threshold):
            continue
        suppressed.append((bbox, label, score))
    
    return suppressed


def _is_duplicate_detection(
    candidate_bbox: BoundingBox,
    candidate_label: str,
    existing: List[Tuple[BoundingBox, str, float]],
    iou_threshold: float = 0.7,
) -> bool:
    """Check if detection is duplicate of existing ones."""
    for bbox, label, _score in existing:
        if label != candidate_label:
            continue
        if _bbox_iou(candidate_bbox, bbox) >= iou_threshold:
            return True
    return False


def _bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    """Calculate Intersection over Union for two bounding boxes."""
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


def detect_and_crop_image_improved(
    image_path: Path,
    output_dir: Path,
    max_crops_counter: List[int] | None = None,
    config: Optional[DetectionConfig] = None,
    save_annotated: bool = True,
    save_crops: bool = True,
    backend: Optional[DetectorBackendProtocol] = None,
    run_id: str | None = None,
) -> ImageDetectionResult:
    """Improved detection with better error handling and performance."""
    
    if config is None:
        config = DetectionConfig()
    
    if not (0.0 <= config.confidence_threshold <= 1.0):
        return ImageDetectionResult(
            source_image=image_path,
            error="confidence_threshold must be between 0.0 and 1.0",
        )

    if backend is None:
        try:
            backend = DetectionBackend(config=config)
        except Exception as exc:
            return ImageDetectionResult(
                source_image=image_path,
                error=f"Failed to initialize detection backend: {exc}",
            )

    annotated_dir, crops_dir, _logs_dir = ensure_output_subdirs(output_dir)
    result = ImageDetectionResult(source_image=image_path)

    try:
        image = load_image(image_path)
    except Exception as exc:
        result.error = f"Failed to load image: {exc}"
        return result

    width, height = image.size
    
    try:
        raw_detections = _run_detection_with_memory_management(
            backend=backend,
            image=image,
            config=config,
        )
    except Exception as exc:
        result.error = f"Detection inference failed: {exc}"
        return result

    if not raw_detections:
        return result

    # Process detections and save crops
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    stem = image_path.stem
    effective_run_id = run_id or generate_run_id()
    crop_counter = 0 if max_crops_counter is None else max_crops_counter[0]

    for bbox, label, score in raw_detections:
        # Validate and clamp bounding box coordinates
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
            crop = image.crop((clean_bbox.x1, clean_bbox.y1, clean_bbox.x2, clean_bbox.y2))
            
            # Ensure crop has minimum size
            if crop.size[0] < 10 or crop.size[1] < 10:
                continue
            
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
            
            # Save with optimized quality
            crop.save(crop_path, format="JPEG", quality=95, optimize=True)
            detection.crop_path = crop_path

        # Draw annotations
        draw.rectangle([clean_bbox.x1, clean_bbox.y1, clean_bbox.x2, clean_bbox.y2], 
                      outline="red", width=3)
        draw.text((clean_bbox.x1 + 2, clean_bbox.y1 + 2), 
                 f"{label} {score:.2f}", fill="red")

        result.detections.append(detection)

    # Save annotated image
    if save_annotated and result.detections:
        filename = safe_filename(stem=stem, label="annotated", suffix=".jpg", counter=0)
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
        
        annotated_image.save(annotated_path, format="JPEG", quality=95, optimize=True)
        for det in result.detections:
            det.annotated_image_path = annotated_path

    if max_crops_counter is not None:
        max_crops_counter[0] = crop_counter

    return result


def detect_and_crop_folder_parallel(
    input_dir: Path,
    output_dir: Path,
    max_images: Optional[int] = None,
    config: Optional[DetectionConfig] = None,
    save_annotated: bool = True,
    save_crops: bool = True,
    backend: Optional[DetectorBackendProtocol] = None,
    progress_callback: Optional[Callable[[int, int, Path], None]] = None,
    cancel_event: Optional[Event] = None,
    resume_from_last: bool = False,
    max_workers: int = 2,  # Conservative for memory usage
) -> BatchDetectionResult:
    """Parallel processing version with improved resource management."""
    
    if config is None:
        config = DetectionConfig()
    
    run_id = generate_run_id()
    
    # Validation
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
    
    if not (0.0 <= config.confidence_threshold <= 1.0):
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

    # Initialize backend
    if backend is None:
        try:
            backend = DetectionBackend(config=config)
        except Exception as exc:
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

    # Get images
    images = list(iter_images(input_dir))
    if max_images is not None:
        images = images[:max_images]

    # Handle resume
    start_index = 0
    if resume_from_last:
        state = load_resume_state(output_dir)
        if state and state.get("input_dir") == str(input_dir):
            start_index = max(0, int(state.get("last_completed_index", -1)) + 1)

    # Process images
    image_results: List[ImageDetectionResult] = []
    errors: List[str] = []
    total_detections = 0
    total_crops = 0
    crops_counter = [0]
    memory_manager = backend.memory_manager

    # Use sequential processing for now (can be upgraded to parallel later)
    for idx, image_path in enumerate(images[start_index:], start=start_index):
        if cancel_event is not None and cancel_event.is_set():
            errors.append("Scan cancelled by user.")
            break
        
        # Check memory usage
        if memory_manager.check_memory():
            memory_manager.force_gc()
            if memory_manager.check_memory():  # Still too high
                errors.append(f"Memory threshold exceeded at image {idx + 1}")
                break
        
        res = detect_and_crop_image_improved(
            image_path=image_path,
            output_dir=output_dir,
            max_crops_counter=crops_counter,
            config=config,
            save_annotated=save_annotated,
            save_crops=save_crops,
            backend=backend,
            run_id=run_id,
        )
        
        image_results.append(res)
        if not res.successful:
            errors.append(f"{image_path}: {res.error}")
        else:
            total_detections += len(res.detections)
        
        # Save progress state
        save_resume_state(
            output_dir,
            {
                "run_id": run_id,
                "input_dir": str(input_dir),
                "last_completed_index": idx,
                "last_completed_image": str(image_path),
                "memory_usage_mb": memory_manager.get_memory_usage(),
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

    # Clear state only on complete successful finish
    if not errors:
        clear_resume_state(output_dir)

    return batch
