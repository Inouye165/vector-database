from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import unittest

from io_utils import (
    clear_resume_state,
    load_resume_state,
    save_manifest,
    save_resume_state,
    safe_filename,
)
from models import BatchDetectionResult, BoundingBox, Detection, ImageDetectionResult


class IoUtilsTests(unittest.TestCase):
    def test_safe_filename_sanitizes(self) -> None:
        name = safe_filename("img:/bad name", "person*1", ".jpg", 3)
        self.assertIn("__", name)
        self.assertTrue(name.endswith(".jpg"))
        self.assertNotIn(":", name)
        self.assertNotIn("*", name)
        self.assertNotIn(" ", name)

    def test_save_manifest_writes_csv_json(self) -> None:
        with TemporaryDirectory() as td:
            base = Path(td)
            detection = Detection(
                label="person",
                confidence=0.9,
                bbox=BoundingBox(1, 2, 10, 12),
                source_image=Path("source.jpg"),
                crop_path=Path("crop.jpg"),
                annotated_image_path=Path("annotated.jpg"),
                run_id="run123",
            )
            img_result = ImageDetectionResult(
                source_image=Path("source.jpg"),
                detections=[detection],
            )
            batch = BatchDetectionResult(
                input_dir=base,
                output_dir=base,
                run_id="run123",
                images_scanned=1,
                total_detections=1,
                total_crops=1,
                image_results=[img_result],
                errors=[],
            )
            save_manifest(batch, base)

            csv_path = base / "output" / "manifest.csv"
            json_path = base / "output" / "manifest.json"
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())

            with csv_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["label"], "person")
            self.assertEqual(rows[0]["x1"], "1")

    def test_resume_state_roundtrip(self) -> None:
        with TemporaryDirectory() as td:
            base = Path(td)
            payload = {"last_completed_index": 3, "input_dir": "C:/photos"}
            save_resume_state(base, payload)
            loaded = load_resume_state(base)
            self.assertEqual(loaded["last_completed_index"], 3)
            clear_resume_state(base)
            self.assertIsNone(load_resume_state(base))


if __name__ == "__main__":
    unittest.main()

