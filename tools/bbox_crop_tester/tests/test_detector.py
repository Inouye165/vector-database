from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from threading import Event

from detector import (
    _bbox_iou,
    _is_duplicate_detection,
    detect_and_crop_folder,
    detect_and_crop_image,
)
from models import BoundingBox
from PIL import Image


class DetectorUtilsTests(unittest.TestCase):
    def test_bbox_iou_overlap(self) -> None:
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(5, 5, 15, 15)
        iou = _bbox_iou(a, b)
        self.assertGreater(iou, 0.1)
        self.assertLess(iou, 0.2)

    def test_duplicate_detection_true(self) -> None:
        existing = [(BoundingBox(0, 0, 100, 100), "person", 0.9)]
        candidate = BoundingBox(5, 5, 98, 98)
        self.assertTrue(_is_duplicate_detection(candidate, "person", existing))

    def test_duplicate_detection_false_different_label(self) -> None:
        existing = [(BoundingBox(0, 0, 100, 100), "bird", 0.9)]
        candidate = BoundingBox(5, 5, 98, 98)
        self.assertFalse(_is_duplicate_detection(candidate, "person", existing))

    def test_detect_and_crop_folder_invalid_input_dir(self) -> None:
        with TemporaryDirectory() as td:
            output_dir = Path(td)
            result = detect_and_crop_folder(
                input_dir=Path(td) / "missing",
                output_dir=output_dir,
            )
            self.assertEqual(result.images_scanned, 0)
            self.assertGreater(len(result.errors), 0)

    def test_detect_and_crop_folder_invalid_confidence(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td)
            result = detect_and_crop_folder(
                input_dir=p,
                output_dir=p,
                confidence_threshold=1.5,
            )
            self.assertEqual(result.images_scanned, 0)
            self.assertIn("confidence_threshold", result.errors[0])

    def test_detect_and_crop_image_handles_inference_failure(self) -> None:
        with TemporaryDirectory() as td:
            base = Path(td)
            image_path = base / "test.jpg"
            Image.new("RGB", (64, 64), color="white").save(image_path)

            with patch("detector._run_detection_on_image", side_effect=RuntimeError("boom")):
                result = detect_and_crop_image(
                    image_path=image_path,
                    output_dir=base,
                    confidence_threshold=0.25,
                    backend=object(),
                )
            self.assertFalse(result.successful)
            self.assertIn("Detection inference failed", result.error or "")

    def test_detect_and_crop_image_saves_crop_with_mocked_detection(self) -> None:
        with TemporaryDirectory() as td:
            base = Path(td)
            image_path = base / "test2.jpg"
            Image.new("RGB", (120, 100), color="white").save(image_path)

            with patch(
                "detector._run_detection_on_image",
                return_value=[(BoundingBox(10, 10, 60, 60), "person", 0.9)],
            ):
                result = detect_and_crop_image(
                    image_path=image_path,
                    output_dir=base,
                    confidence_threshold=0.25,
                    backend=object(),
                )
            self.assertTrue(result.successful)
            self.assertEqual(len(result.detections), 1)
            self.assertIsNotNone(result.detections[0].crop_path)
            self.assertTrue(result.detections[0].crop_path.exists())

    def test_detect_and_crop_folder_cancel_event(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td)
            Image.new("RGB", (50, 50), color="white").save(p / "a.jpg")
            Image.new("RGB", (50, 50), color="white").save(p / "b.jpg")
            cancel_event = Event()
            cancel_event.set()
            result = detect_and_crop_folder(
                input_dir=p,
                output_dir=p,
                cancel_event=cancel_event,
                backend=object(),
            )
            self.assertIn("cancelled", " ".join(result.errors).lower())


if __name__ == "__main__":
    unittest.main()

