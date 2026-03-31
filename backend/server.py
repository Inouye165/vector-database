"""Image vector database backend – FastAPI + ChromaDB + OpenCLIP.

Indexes photos into a vector database using CLIP embeddings so users
can search for images with natural language queries like
"show me images of a dog at the park".

Supports rich metadata (EXIF date, GPS location, tags, comments) and
metadata-aware search so location, date, and keywords influence retrieval.
"""

import io
import os
import hashlib
import json
import re
import shutil
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import open_clip
import PIL.Image
import PIL.ExifTags
import pillow_heif
import torch
import chromadb
from dotenv import load_dotenv

# Optional: BLIP image captioning (install `transformers` for this feature)
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# ---------------------------------------------------------------------------
# Config – load from .env, fall back to sensible defaults
# ---------------------------------------------------------------------------
load_dotenv()

PHOTOS_DIR = Path(os.getenv(
    "PHOTOS_DIR",
    str(Path(__file__).resolve().parent.parent / "photos"),
))
CHROMA_DIR = Path(os.getenv(
    "CHROMA_DIR",
    str(Path(__file__).resolve().parent.parent / "chroma_db"),
))
MAX_INDEX_IMAGES = int(os.getenv("MAX_INDEX_IMAGES", "5000"))
INDEX_BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", "25"))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp",
                    ".heic", ".heif", ".webp", ".tiff", ".tif"}
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
VECTOR_DB_DEVICE = os.getenv("VECTOR_DB_DEVICE", "").strip().lower()
VECTOR_DB_CPU_THREADS = max(0, int(os.getenv("VECTOR_DB_CPU_THREADS", "0")))
INDEX_THROTTLE_MS = max(0, int(os.getenv("INDEX_THROTTLE_MS", "0")))
INDEX_BATCH_COOLDOWN_MS = max(0, int(os.getenv("INDEX_BATCH_COOLDOWN_MS", "0")))

# -- Search-quality knobs --------------------------------------------------
ENABLE_CAPTIONING = os.getenv("ENABLE_CAPTIONING", "true").lower() == "true"
DEFAULT_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.15"))
CAPTION_WEIGHT = float(os.getenv("CAPTION_WEIGHT", "0.4"))
CAPTION_MODEL_NAME = os.getenv(
    "CAPTION_MODEL", "Salesforce/blip-image-captioning-base",
)

# ---------------------------------------------------------------------------
# Register HEIF/HEIC opener with Pillow
# ---------------------------------------------------------------------------
pillow_heif.register_heif_opener()

# ---------------------------------------------------------------------------
# Runtime state (populated at startup)
# ---------------------------------------------------------------------------
runtime = {
    "model": None,
    "preprocess": None,
    "tokenizer": None,
    "device": None,
    "chroma_collection": None,
    "chroma_client": None,
    "caption_model": None,
    "caption_processor": None,
    "caption_collection": None,
    "path_index": {},
    "interrupted_checkpoint": None,
    "last_index_summary": {},
    "rebuild_status": {
        "is_running": False,
        "last_trigger": None,
        "last_started_at": None,
        "last_completed_at": None,
        "last_error": None,
    },
    "progress": {
        "phase": "idle",
        "current": 0,
        "total": 0,
        "detail": "",
    },
}
index_lock = threading.Lock()
_runtime_limits_configured = False


def _now_iso() -> str:
    """Return a compact local timestamp for status payloads."""
    return datetime.now().isoformat(timespec="seconds")


def _new_index_summary(reset_db: bool, trigger: str) -> dict:
    """Create an empty per-run indexing summary."""
    return {
        "status": "running",
        "trigger": trigger,
        "reset_db": reset_db,
        "started_at": _now_iso(),
        "completed_at": None,
        "total_discovered": 0,
        "existing_indexed_images": 0,
        "new_candidates": 0,
        "sampled_images": 0,
        "kept_images": 0,
        "filtered_images": 0,
        "skipped_images": 0,
        "indexed_images": 0,
        "batches_committed": 0,
        "resumed_from": 0,
        "collection_count": 0,
        "reason_counts": {},
        "error": None,
    }


def _increment_reason(summary: dict, reason: str, amount: int = 1) -> None:
    """Increment a summary reason counter."""
    summary["reason_counts"][reason] = summary["reason_counts"].get(reason, 0) + amount


# ---------------------------------------------------------------------------
# Durable indexing checkpoint – survives process restarts
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = CHROMA_DIR / ".index_checkpoint.json"


def _load_checkpoint() -> dict | None:
    """Load the checkpoint file if it exists and is valid JSON."""
    if not CHECKPOINT_PATH.exists():
        return None
    try:
        data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("version") == 1:
            return data
        print(f"Ignoring unrecognised checkpoint version: {data.get('version')}")
        return None
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Ignoring corrupt checkpoint file: {exc}")
        return None


def _save_checkpoint(trigger: str, started_at: str, total_sampled: int,
                     committed_count: int) -> None:
    """Persist current indexing progress to disk after each batch commit."""
    data = {
        "version": 1,
        "trigger": trigger,
        "started_at": started_at,
        "total_sampled": total_sampled,
        "committed_count": committed_count,
        "last_batch_at": _now_iso(),
    }
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(CHECKPOINT_PATH)


def _delete_checkpoint() -> None:
    """Remove the checkpoint file after a successful indexing run."""
    try:
        CHECKPOINT_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def _delete_database_dir() -> None:
    """Delete the persisted Chroma database directory if it exists."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)


def _reset_chroma_store() -> None:
    """Reset the active Chroma store without deleting open files in-process."""
    _delete_checkpoint()
    runtime["interrupted_checkpoint"] = None
    client = runtime["chroma_client"]
    runtime["chroma_collection"] = None
    runtime["caption_collection"] = None
    runtime["path_index"] = {}

    if client is not None:
        for name in ("photos", "photo_captions"):
            try:
                client.delete_collection(name)
            except Exception:
                pass
        return

    _delete_database_dir()


def _init_chroma_collection(reset_db: bool = False) -> None:
    """Initialize or recreate the Chroma client and collection."""
    if reset_db:
        _reset_chroma_store()

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    if runtime["chroma_client"] is None:
        runtime["chroma_client"] = chromadb.PersistentClient(path=str(CHROMA_DIR))
    runtime["chroma_collection"] = runtime["chroma_client"].get_or_create_collection(
        name="photos",
        metadata={"hnsw:space": "cosine"},
    )
    runtime["caption_collection"] = runtime["chroma_client"].get_or_create_collection(
        name="photo_captions",
        metadata={"hnsw:space": "cosine"},
    )


def _stats_payload() -> dict:
    """Build the stats payload used by status and rebuild endpoints."""
    collection = runtime["chroma_collection"]
    caption_col = runtime["caption_collection"]
    indexed_images = collection.count() if collection else 0
    captioned_images = caption_col.count() if caption_col else 0
    return {
        "indexed_images": indexed_images,
        "captioned_images": captioned_images,
        "captioning_enabled": ENABLE_CAPTIONING and runtime["caption_model"] is not None,
        "photos_dir": str(PHOTOS_DIR),
        "max_index_images": MAX_INDEX_IMAGES,
        "default_threshold": DEFAULT_THRESHOLD,
        "runtime_limits": {
            "device_override": VECTOR_DB_DEVICE or None,
            "cpu_threads": VECTOR_DB_CPU_THREADS,
            "index_throttle_ms": INDEX_THROTTLE_MS,
            "index_batch_cooldown_ms": INDEX_BATCH_COOLDOWN_MS,
        },
        "rebuild_status": runtime["rebuild_status"],
        "last_index_summary": runtime["last_index_summary"],
        "interrupted_checkpoint": runtime.get("interrupted_checkpoint"),
    }


def _configure_runtime_limits() -> None:
    """Apply optional runtime limits before loading heavyweight models."""
    global _runtime_limits_configured

    if _runtime_limits_configured:
        return

    if VECTOR_DB_CPU_THREADS > 0:
        torch.set_num_threads(VECTOR_DB_CPU_THREADS)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    _runtime_limits_configured = True


def _get_device() -> str:
    """Select the best available compute device."""
    if VECTOR_DB_DEVICE == "cpu":
        return "cpu"
    if VECTOR_DB_DEVICE == "cuda" and torch.cuda.is_available():
        return "cuda"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_image(path: Path) -> PIL.Image.Image:
    """Load an image file (including HEIC) and return as RGB PIL Image."""
    with PIL.Image.open(path) as img:
        return img.convert("RGB")


# ---------------------------------------------------------------------------
# Content-hash ID – deduplicates identical images across folders
# ---------------------------------------------------------------------------
def _file_id(path: Path) -> str:
    """Deterministic ID based on file content (first 64 KB + file size)."""
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(size.to_bytes(8, "big"))
    with open(path, "rb") as f:
        h.update(f.read(65_536))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------
_EXIF_TAG_MAP = {v: k for k, v in PIL.ExifTags.TAGS.items()}
_GPS_TAG_MAP = {v: k for k, v in PIL.ExifTags.GPSTAGS.items()}


def _dms_to_decimal(dms, ref: str) -> float | None:
    """Convert EXIF GPS DMS tuple to decimal degrees."""
    try:
        degrees, minutes, seconds = [float(x) for x in dms]
        dec = degrees + minutes / 60.0 + seconds / 3600.0
        if ref in ("S", "W"):
            dec = -dec
        return round(dec, 6)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _normalize_exif_date(raw: str) -> str:
    """Normalise EXIF date strings like '2025:02:14 10:30:00' to ISO."""
    if not raw:
        return ""
    return re.sub(r"^(\d{4}):(\d{2}):(\d{2})", r"\1-\2-\3", raw.strip())


def _extract_metadata(path: Path) -> dict:
    """Extract rich metadata from an image file."""
    rel = path.relative_to(PHOTOS_DIR)
    stat = path.stat()
    meta: dict = {
        "filename": path.name,
        "path": str(path),
        "relative_path": str(rel),
        "folder": str(rel.parent) if str(rel.parent) != "." else "",
        "file_size": stat.st_size,
        "date_modified": datetime.fromtimestamp(
            stat.st_mtime
        ).isoformat(timespec="seconds"),
    }

    try:
        with PIL.Image.open(path) as img:
            meta["width"] = img.width
            meta["height"] = img.height

            exif = img.getexif()
            if exif:
                for tag_name in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                    tag_id = _EXIF_TAG_MAP.get(tag_name)
                    if tag_id and tag_id in exif:
                        raw = exif[tag_id]
                        if raw:
                            meta["date_taken"] = _normalize_exif_date(str(raw))
                        break

                make = str(exif.get(_EXIF_TAG_MAP.get("Make", -1), ""))
                model_name = str(exif.get(_EXIF_TAG_MAP.get("Model", -1), ""))
                cam = f"{make} {model_name}".strip()
                if cam:
                    meta["camera"] = cam

                try:
                    gps_ifd = exif.get_ifd(PIL.ExifTags.IFD.GPSInfo)
                except (AttributeError, KeyError):
                    gps_ifd = {}
                if gps_ifd:
                    lat_tag = _GPS_TAG_MAP.get("GPSLatitude")
                    lat_ref_tag = _GPS_TAG_MAP.get("GPSLatitudeRef")
                    lon_tag = _GPS_TAG_MAP.get("GPSLongitude")
                    lon_ref_tag = _GPS_TAG_MAP.get("GPSLongitudeRef")

                    lat_dms = gps_ifd.get(lat_tag) if lat_tag else None
                    lat_ref = gps_ifd.get(lat_ref_tag) if lat_ref_tag else None
                    lon_dms = gps_ifd.get(lon_tag) if lon_tag else None
                    lon_ref = gps_ifd.get(lon_ref_tag) if lon_ref_tag else None

                    if lat_dms and lat_ref:
                        lat = _dms_to_decimal(lat_dms, lat_ref)
                        if lat is not None:
                            meta["gps_lat"] = lat
                    if lon_dms and lon_ref:
                        lon = _dms_to_decimal(lon_dms, lon_ref)
                        if lon is not None:
                            meta["gps_lon"] = lon

            info = img.info or {}
            if "keywords" in info:
                kw = info["keywords"]
                if isinstance(kw, (list, tuple)):
                    tags = ", ".join(str(k) for k in kw if k)
                else:
                    tags = str(kw)
                if tags:
                    meta["tags"] = tags[:500]
            if "comment" in info:
                c = str(info["comment"])[:500]
                if c:
                    meta["comment"] = c
            elif "description" in info:
                c = str(info["description"])[:500]
                if c:
                    meta["comment"] = c

    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        pass

    meta["best_date"] = meta.get("date_taken", "") or meta.get("date_modified", "")
    return meta


def _metadata_text(meta: dict) -> str:
    """Build a short text summary of metadata for embedding fusion."""
    parts = []
    if meta.get("folder"):
        parts.append(f"folder: {meta['folder']}")
    if meta.get("date_taken"):
        parts.append(f"date: {meta['date_taken']}")
    elif meta.get("date_modified"):
        parts.append(f"date: {meta['date_modified']}")
    if meta.get("tags"):
        parts.append(f"tags: {meta['tags']}")
    if meta.get("comment"):
        parts.append(f"comment: {meta['comment']}")
    if meta.get("caption"):
        parts.append(f"description: {meta['caption']}")
    if meta.get("camera"):
        parts.append(f"camera: {meta['camera']}")
    if meta.get("gps_lat") is not None and meta.get("gps_lon") is not None:
        parts.append(f"location: {meta['gps_lat']}, {meta['gps_lon']}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Vision captioning
# ---------------------------------------------------------------------------
def _generate_caption(img: PIL.Image.Image) -> str:
    """Generate a natural-language caption for *img* using BLIP."""
    if runtime["caption_model"] is None:
        return ""
    processor = runtime["caption_processor"]
    model = runtime["caption_model"]
    inputs = processor(img, return_tensors="pt").to(runtime["device"])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True).strip()
    del inputs
    del out
    return caption


def _sleep_if_needed(delay_ms: int) -> None:
    """Sleep for a short configured interval to yield shared resources."""
    if delay_ms > 0:
        time.sleep(delay_ms / 1000)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def _embed_image(img: PIL.Image.Image) -> list[float]:
    """Return a normalised CLIP embedding for a PIL Image."""
    tensor = runtime["preprocess"](img).unsqueeze(0).to(runtime["device"])
    with torch.no_grad():
        features = runtime["model"].encode_image(tensor)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten().tolist()


def _embed_text(text: str) -> list[float]:
    """Return a normalised CLIP embedding for a text query."""
    tokens = runtime["tokenizer"]([text]).to(runtime["device"])
    with torch.no_grad():
        features = runtime["model"].encode_text(tokens)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten().tolist()


def _fused_embedding(img: PIL.Image.Image, meta_text: str,
                     image_weight: float = 0.85) -> list[float]:
    """Combine image + metadata text embeddings with weighted fusion."""
    img_emb = np.array(_embed_image(img))
    if meta_text:
        txt_emb = np.array(_embed_text(meta_text))
        combined = image_weight * img_emb + (1.0 - image_weight) * txt_emb
        combined = combined / np.linalg.norm(combined)
        return combined.tolist()
    return img_emb.tolist()


# ---------------------------------------------------------------------------
# Image discovery & sampling
# ---------------------------------------------------------------------------
def _discover_images() -> list[Path]:
    """Recursively discover all image files under PHOTOS_DIR."""
    return sorted(
        p for p in PHOTOS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _sample_images(all_files: list[Path], max_count: int) -> list[Path]:
    """Select up to max_count images: every-third, round-robin across folders.

    Deduplicates by content hash so copied images are only included once.
    """
    folder_buckets: dict[str, list[Path]] = defaultdict(list)
    for p in all_files:
        rel_folder = str(p.relative_to(PHOTOS_DIR).parent)
        folder_buckets[rel_folder].append(p)

    # Every-third-image selection within each folder
    stepped: dict[str, list[Path]] = {}
    for folder, files in sorted(folder_buckets.items()):
        stepped[folder] = files[::3]

    # Round-robin across folders until we hit max_count
    selected: list[Path] = []
    seen_hashes: set[str] = set()
    folder_iters = {
        folder: iter(files) for folder, files in sorted(stepped.items())
    }

    while folder_iters and len(selected) < max_count:
        exhausted = []
        for folder, it in list(folder_iters.items()):
            if len(selected) >= max_count:
                break
            try:
                candidate = next(it)
                fid = _file_id(candidate)
                if fid not in seen_hashes:
                    seen_hashes.add(fid)
                    selected.append(candidate)
            except StopIteration:
                exhausted.append(folder)
        for folder in exhausted:
            del folder_iters[folder]

    return selected


def _filter_new_candidates(all_files: list[Path], existing_ids: set[str],
                           progress: dict) -> list[Path]:
    """Return files not yet indexed while reporting progress for resume scans."""
    progress.update(
        phase="resuming",
        current=0,
        total=len(all_files),
        detail=(
            f"Checking {len(existing_ids)} indexed images against "
            f"{len(all_files)} files…"
        ),
    )

    new_candidates = []
    for idx, fpath in enumerate(all_files, start=1):
        if _file_id(fpath) not in existing_ids:
            new_candidates.append(fpath)
        if idx % 25 == 0 or idx == len(all_files):
            progress["current"] = idx
            progress["detail"] = (
                f"Checked {idx} of {len(all_files)} files; "
                f"found {len(new_candidates)} new candidates"
            )

    return new_candidates


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def _index_photos(reset_db: bool = False, trigger: str = "manual") -> dict:
    """Scan PHOTOS_DIR, sample images, extract metadata, and upsert."""
    summary = _new_index_summary(reset_db=reset_db, trigger=trigger)
    collection = runtime["chroma_collection"]
    progress = runtime["progress"]
    progress.update(phase="discovering", current=0, total=0,
                    detail="Scanning photo directories\u2026")

    if not PHOTOS_DIR.exists():
        message = f"Photos directory not found: {PHOTOS_DIR}"
        print(message)
        summary["status"] = "failed"
        summary["error"] = message
        summary["completed_at"] = _now_iso()
        _increment_reason(summary, "missing_source_dir")
        return summary

    existing_ids = set(collection.get()["ids"])
    summary["existing_indexed_images"] = len(existing_ids)

    all_files = _discover_images()
    summary["total_discovered"] = len(all_files)
    progress.update(phase="sampling", current=0, total=len(all_files),
                    detail=f"Found {len(all_files)} images, selecting sample\u2026")
    print(f"Found {len(all_files)} images across {PHOTOS_DIR}")

    if existing_ids:
        new_candidates = _filter_new_candidates(all_files, existing_ids, progress)
        summary["new_candidates"] = len(new_candidates)
        if not new_candidates:
            print(f"All {len(existing_ids)} images already indexed.")
            _rebuild_path_index()
            _delete_checkpoint()
            runtime["interrupted_checkpoint"] = None
            summary["status"] = "completed"
            summary["completed_at"] = _now_iso()
            summary["collection_count"] = collection.count()
            return summary
        progress.update(
            phase="sampling",
            current=0,
            total=len(new_candidates),
            detail=(
                f"Found {len(new_candidates)} new candidates, selecting sample\u2026"
            ),
        )
        sampled = _sample_images(
            new_candidates,
            max(0, MAX_INDEX_IMAGES - len(existing_ids)),
        )
    else:
        summary["new_candidates"] = len(all_files)
        sampled = _sample_images(all_files, MAX_INDEX_IMAGES)

    summary["sampled_images"] = len(sampled)

    if not sampled:
        print("No new images to index.")
        _rebuild_path_index()
        _delete_checkpoint()
        runtime["interrupted_checkpoint"] = None
        summary["status"] = "completed"
        summary["completed_at"] = _now_iso()
        summary["collection_count"] = collection.count()
        return summary

    summary["resumed_from"] = len(existing_ids)
    print(f"Indexing {len(sampled)} new images …")
    progress.update(phase="indexing", current=0, total=len(sampled), detail="")
    batch_ids, batch_embs, batch_metas = [], [], []
    batch_cap_ids, batch_cap_embs, batch_cap_metas = [], [], []
    total_indexed = 0
    total_captioned = 0
    run_start = summary["started_at"]

    for i, fpath in enumerate(sampled):
        img = None
        try:
            img = _load_image(fpath)
            meta = _extract_metadata(fpath)
            # Auto-caption with BLIP when available
            caption = _generate_caption(img)
            if caption:
                meta["caption"] = caption
            meta_text = _metadata_text(meta)
            emb = _fused_embedding(img, meta_text)
            fid = _file_id(fpath)
            batch_ids.append(fid)
            batch_embs.append(emb)
            batch_metas.append(meta)
            # Store caption embedding in the captions collection
            if caption:
                batch_cap_ids.append(fid)
                batch_cap_embs.append(_embed_text(caption))
                batch_cap_metas.append(meta)
            progress["current"] = i + 1
            progress["detail"] = fpath.name
            if (i + 1) % 10 == 0 or i + 1 == len(sampled):
                print(f"  [{i + 1}/{len(sampled)}]")
        except (MemoryError, OSError, RuntimeError, TypeError, ValueError) as exc:
            summary["skipped_images"] += 1
            _increment_reason(summary, "load_or_embed_error")
            print(f"  Skipped {fpath.name}: {exc}")
            continue
        finally:
            if img is not None:
                close_image = getattr(img, "close", None)
                if callable(close_image):
                    close_image()
            _sleep_if_needed(INDEX_THROTTLE_MS)

        # Commit batch when full or at end of sample list
        if len(batch_ids) >= INDEX_BATCH_SIZE or i == len(sampled) - 1:
            if batch_ids:
                collection.upsert(
                    ids=batch_ids, embeddings=batch_embs,
                    metadatas=batch_metas,
                )
                for fid, meta in zip(batch_ids, batch_metas):
                    if meta and meta.get("path"):
                        runtime["path_index"][fid] = Path(meta["path"])
                if batch_cap_ids:
                    runtime["caption_collection"].upsert(
                        ids=batch_cap_ids,
                        embeddings=batch_cap_embs,
                        metadatas=batch_cap_metas,
                    )
                total_indexed += len(batch_ids)
                total_captioned += len(batch_cap_ids)
                summary["batches_committed"] += 1
                _save_checkpoint(
                    trigger=trigger, started_at=run_start,
                    total_sampled=len(sampled),
                    committed_count=total_indexed,
                )
                print(f"  Batch {summary['batches_committed']} committed "
                      f"({total_indexed}/{len(sampled)} images)")
                batch_ids, batch_embs, batch_metas = [], [], []
                batch_cap_ids, batch_cap_embs, batch_cap_metas = [], [], []
                _sleep_if_needed(INDEX_BATCH_COOLDOWN_MS)

    if total_indexed:
        print(f"Indexed {total_indexed} images ({total_captioned} captioned) "
              f"in {summary['batches_committed']} batches into ChromaDB.")

    _rebuild_path_index()
    _delete_checkpoint()
    runtime["interrupted_checkpoint"] = None
    summary["status"] = "completed"
    summary["completed_at"] = _now_iso()
    summary["indexed_images"] = total_indexed
    summary["kept_images"] = total_indexed
    summary["collection_count"] = collection.count()
    return summary


def _run_indexing(reset_db: bool = False, trigger: str = "manual") -> dict:
    """Run a full indexing pass with status tracking and optional reset."""
    with index_lock:
        rebuild_status = runtime["rebuild_status"]
        rebuild_status["is_running"] = True
        rebuild_status["last_trigger"] = trigger
        rebuild_status["last_started_at"] = _now_iso()
        rebuild_status["last_error"] = None

        try:
            _init_chroma_collection(reset_db=reset_db)
            summary = _index_photos(reset_db=reset_db, trigger=trigger)
            runtime["last_index_summary"] = summary
            rebuild_status["last_completed_at"] = summary.get("completed_at")
            rebuild_status["last_error"] = summary.get("error")
            return summary
        except (OSError, RuntimeError, ValueError) as exc:
            runtime["last_index_summary"] = {
                "status": "failed",
                "trigger": trigger,
                "reset_db": reset_db,
                "started_at": rebuild_status["last_started_at"],
                "completed_at": _now_iso(),
                "error": str(exc),
                "reason_counts": {"unexpected_error": 1},
            }
            rebuild_status["last_completed_at"] = runtime["last_index_summary"]["completed_at"]
            rebuild_status["last_error"] = str(exc)
            raise
        finally:
            rebuild_status["is_running"] = False
            runtime["progress"].update(phase="idle", current=0, total=0, detail="")


def _rebuild_path_index():
    """Populate the in-memory path index from ChromaDB metadata."""
    collection = runtime["chroma_collection"]
    all_meta = collection.get(include=["metadatas"])
    runtime["path_index"] = {}
    for fid, meta in zip(all_meta["ids"], all_meta["metadatas"]):
        if meta and meta.get("path"):
            runtime["path_index"][fid] = Path(meta["path"])


# ---------------------------------------------------------------------------
# App lifespan – load model & index on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    _configure_runtime_limits()
    # Load CLIP model
    runtime["device"] = _get_device()
    print(f"Loading CLIP model ({CLIP_MODEL}) on {runtime['device']} …")
    runtime["model"], _, runtime["preprocess"] = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=runtime["device"],
    )
    runtime["tokenizer"] = open_clip.get_tokenizer(CLIP_MODEL)
    runtime["model"].eval()
    print("CLIP model loaded.")

    # Load BLIP captioning model (optional)
    if ENABLE_CAPTIONING and _HAS_TRANSFORMERS:
        print(f"Loading BLIP caption model ({CAPTION_MODEL_NAME}) \u2026")
        runtime["caption_processor"] = BlipProcessor.from_pretrained(
            CAPTION_MODEL_NAME,
        )
        runtime["caption_model"] = BlipForConditionalGeneration.from_pretrained(
            CAPTION_MODEL_NAME,
        ).to(runtime["device"])
        runtime["caption_model"].eval()
        print("BLIP caption model loaded.")
    elif ENABLE_CAPTIONING:
        print("Captioning enabled but `transformers` not installed \u2013 skipping.")

    _init_chroma_collection()
    _rebuild_path_index()
    count = runtime["chroma_collection"].count()
    cap_count = runtime["caption_collection"].count()

    ckpt = _load_checkpoint()
    if ckpt:
        runtime["interrupted_checkpoint"] = ckpt
        committed = ckpt.get("committed_count", 0)
        total = ckpt.get("total_sampled", 0)
        print(
            f"Interrupted indexing detected: {committed}/{total} images "
            f"committed (started {ckpt.get('started_at', '?')}). "
            "Use Continue Indexing in the UI to resume."
        )

    print(
        f"Loaded {count} indexed images ({cap_count} captioned). "
        "Use Reset DB to reindex."
    )

    yield  # app runs


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Image Vector Search", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/search")
def search_images(
    q: str = Query(..., min_length=1),
    n: int = Query(20, ge=1, le=100),
    folder: str = Query(None, description="Exact folder filter"),
    date_from: str = Query(None, description="ISO date lower bound"),
    date_to: str = Query(None, description="ISO date upper bound"),
    threshold: float = Query(
        None, ge=0.0, le=1.0,
        description="Minimum similarity score (0\u20131)",
    ),
):
    """Search with hybrid image + caption matching and threshold filtering."""
    min_score = threshold if threshold is not None else DEFAULT_THRESHOLD
    query_emb = _embed_text(q)

    # Build optional ChromaDB where filter (only equality filters)
    where_clauses = []
    if folder:
        where_clauses.append({"folder": {"$eq": folder}})

    where = None
    if len(where_clauses) == 1:
        where = where_clauses[0]
    elif len(where_clauses) > 1:
        where = {"$and": where_clauses}

    collection = runtime["chroma_collection"]
    caption_col = runtime["caption_collection"]
    count = collection.count()
    if count == 0:
        return {"query": q, "results": [], "threshold": min_score}

    # Over-fetch so we have candidates for hybrid merge + threshold filter
    fetch_n = min(max(n * 4, 80), count)

    # -- Image-embedding search -------------------------------------------
    img_results = collection.query(
        query_embeddings=[query_emb],
        n_results=fetch_n,
        where=where,
    )
    img_scores: dict[str, float] = {}
    img_meta: dict[str, dict] = {}
    for fid, meta, dist in zip(
        img_results["ids"][0],
        img_results["metadatas"][0],
        img_results["distances"][0],
    ):
        img_scores[fid] = 1.0 - dist
        img_meta[fid] = meta

    # -- Caption-embedding search (when available) ------------------------
    cap_scores: dict[str, float] = {}
    cap_count = caption_col.count() if caption_col else 0
    if cap_count > 0:
        cap_fetch = min(fetch_n, cap_count)
        cap_results = caption_col.query(
            query_embeddings=[query_emb],
            n_results=cap_fetch,
            where=where,
        )
        for fid, meta, dist in zip(
            cap_results["ids"][0],
            cap_results["metadatas"][0],
            cap_results["distances"][0],
        ):
            cap_scores[fid] = 1.0 - dist
            if fid not in img_meta:
                img_meta[fid] = meta

    # -- Merge scores -----------------------------------------------------
    all_ids = set(img_scores) | set(cap_scores)
    w_img = 1.0 - CAPTION_WEIGHT
    w_cap = CAPTION_WEIGHT
    scored: list[tuple[str, float, float, float]] = []
    for fid in all_ids:
        i_sc = img_scores.get(fid, 0.0)
        c_sc = cap_scores.get(fid, 0.0)
        if cap_scores and fid in cap_scores:
            combined = w_img * i_sc + w_cap * c_sc
        else:
            combined = i_sc
        scored.append((fid, combined, i_sc, c_sc))
    scored.sort(key=lambda x: x[1], reverse=True)

    # -- Build result items with threshold + date filtering ---------------
    items = []
    for fid, combined, i_sc, c_sc in scored:
        if combined < min_score:
            continue
        meta = img_meta.get(fid, {})
        best = meta.get("best_date", "") or meta.get("date_modified", "")
        if date_from and best < date_from:
            continue
        if date_to and best > date_to + "T99":
            continue

        item = {
            "id": fid,
            "filename": meta.get("filename", ""),
            "score": round(combined, 4),
            "image_score": round(i_sc, 4),
            "caption_score": round(c_sc, 4),
            "caption": meta.get("caption", ""),
            "url": f"/api/photo/{fid}",
            "folder": meta.get("folder", ""),
            "relative_path": meta.get("relative_path", ""),
            "date_taken": meta.get("date_taken", ""),
            "date_modified": meta.get("date_modified", ""),
            "tags": meta.get("tags", ""),
            "comment": meta.get("comment", ""),
        }
        if meta.get("width"):
            item["width"] = meta["width"]
            item["height"] = meta.get("height", 0)
        if meta.get("gps_lat") is not None:
            item["gps_lat"] = meta["gps_lat"]
            item["gps_lon"] = meta.get("gps_lon")
        if meta.get("camera"):
            item["camera"] = meta["camera"]
        items.append(item)
        if len(items) >= n:
            break

    return {"query": q, "results": items, "threshold": min_score}


@app.get("/api/photo/{photo_id}")
def get_photo(photo_id: str):
    """Serve a photo by its content-hash ID (safe for recursive folders)."""
    filepath = runtime["path_index"].get(photo_id)
    if filepath is None or not filepath.exists() or not filepath.is_file():
        return Response(status_code=404, content="Not found")

    # Verify the resolved path is inside PHOTOS_DIR
    try:
        filepath.resolve().relative_to(PHOTOS_DIR.resolve())
    except ValueError:
        return Response(status_code=403, content="Forbidden")

    suffix = filepath.suffix.lower()
    if suffix in (".heic", ".heif"):
        img = _load_image(filepath)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return Response(content=buf.getvalue(), media_type="image/jpeg")

    content_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif",
        ".webp": "image/webp", ".bmp": "image/bmp",
        ".tiff": "image/tiff", ".tif": "image/tiff",
    }
    media_type = content_types.get(suffix, "application/octet-stream")
    return Response(content=filepath.read_bytes(), media_type=media_type)


@app.get("/api/progress")
def get_progress():
    """Return current indexing progress."""
    p = runtime["progress"]
    total = p["total"]
    current = p["current"]
    percent = round(current / total * 100) if total > 0 else 0
    return {
        "phase": p["phase"],
        "current": current,
        "total": total,
        "percent": percent,
        "detail": p["detail"],
        "is_running": p["phase"] != "idle",
    }


@app.get("/api/stats")
def stats():
    """Return collection stats."""
    return _stats_payload()


@app.get("/api/search-settings")
def search_settings():
    """Return current search-quality settings."""
    caption_col = runtime["caption_collection"]
    return {
        "captioning_enabled": ENABLE_CAPTIONING and runtime["caption_model"] is not None,
        "captioned_images": caption_col.count() if caption_col else 0,
        "default_threshold": DEFAULT_THRESHOLD,
        "caption_weight": CAPTION_WEIGHT,
        "caption_model": CAPTION_MODEL_NAME if runtime["caption_model"] else None,
    }


@app.post("/api/reindex")
def reindex():
    """Re-scan photos directory and index new images (resumes interrupted runs)."""
    if runtime["progress"]["phase"] != "idle":
        return {"status": "already_running", "message": "Indexing already in progress."}

    runtime["progress"].update(
        phase="starting",
        current=0,
        total=0,
        detail="Preparing reindex…",
    )

    def _bg_reindex():
        try:
            _run_indexing(reset_db=False, trigger="reindex")
        except Exception as exc:
            print(f"Background reindex failed: {exc}")

    threading.Thread(target=_bg_reindex, daemon=True).start()
    return {"status": "started", "message": "Reindex started in background."}


@app.post("/api/reset")
def reset_database():
    """Delete the persisted database and rebuild it from the source folder."""
    if runtime["progress"]["phase"] != "idle":
        return {"status": "already_running", "message": "Indexing already in progress."}

    runtime["progress"].update(
        phase="starting",
        current=0,
        total=0,
        detail="Preparing rebuild…",
    )

    def _bg_rebuild():
        try:
            _run_indexing(reset_db=True, trigger="reset")
        except Exception as exc:
            print(f"Background rebuild failed: {exc}")

    threading.Thread(target=_bg_rebuild, daemon=True).start()
    return {"status": "started", "message": "Rebuild started in background."}
