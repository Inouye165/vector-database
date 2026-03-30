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
import re
import shutil
import threading
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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp",
                    ".heic", ".heif", ".webp", ".tiff", ".tif"}
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

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
    "path_index": {},
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
        "collection_count": 0,
        "reason_counts": {},
        "error": None,
    }


def _increment_reason(summary: dict, reason: str, amount: int = 1) -> None:
    """Increment a summary reason counter."""
    summary["reason_counts"][reason] = summary["reason_counts"].get(reason, 0) + amount


def _delete_database_dir() -> None:
    """Delete the persisted Chroma database directory if it exists."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)


def _reset_chroma_store() -> None:
    """Reset the active Chroma store without deleting open files in-process."""
    client = runtime["chroma_client"]
    runtime["chroma_collection"] = None
    runtime["path_index"] = {}

    if client is not None:
        try:
            client.delete_collection("photos")
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


def _stats_payload() -> dict:
    """Build the stats payload used by status and rebuild endpoints."""
    collection = runtime["chroma_collection"]
    indexed_images = collection.count() if collection else 0
    return {
        "indexed_images": indexed_images,
        "photos_dir": str(PHOTOS_DIR),
        "max_index_images": MAX_INDEX_IMAGES,
        "rebuild_status": runtime["rebuild_status"],
        "last_index_summary": runtime["last_index_summary"],
    }


def _get_device() -> str:
    """Select the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_image(path: Path) -> PIL.Image.Image:
    """Load an image file (including HEIC) and return as RGB PIL Image."""
    img = PIL.Image.open(path)
    img = img.convert("RGB")
    return img


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
        img = PIL.Image.open(path)
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
    if meta.get("camera"):
        parts.append(f"camera: {meta['camera']}")
    if meta.get("gps_lat") is not None and meta.get("gps_lon") is not None:
        parts.append(f"location: {meta['gps_lat']}, {meta['gps_lon']}")
    return "; ".join(parts)


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
        new_candidates = [
            f for f in all_files if _file_id(f) not in existing_ids
        ]
        summary["new_candidates"] = len(new_candidates)
        if not new_candidates:
            print(f"All {len(existing_ids)} images already indexed.")
            _rebuild_path_index()
            summary["status"] = "completed"
            summary["completed_at"] = _now_iso()
            summary["collection_count"] = collection.count()
            return summary
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
        summary["status"] = "completed"
        summary["completed_at"] = _now_iso()
        summary["collection_count"] = collection.count()
        return summary

    print(f"Indexing {len(sampled)} new images …")
    progress.update(phase="indexing", current=0, total=len(sampled), detail="")
    ids, embeddings, metadatas = [], [], []
    for i, fpath in enumerate(sampled):
        try:
            img = _load_image(fpath)
            meta = _extract_metadata(fpath)
            meta_text = _metadata_text(meta)
            emb = _fused_embedding(img, meta_text)
            fid = _file_id(fpath)
            ids.append(fid)
            embeddings.append(emb)
            metadatas.append(meta)
            progress["current"] = i + 1
            progress["detail"] = fpath.name
            if (i + 1) % 10 == 0 or i + 1 == len(sampled):
                print(f"  [{i + 1}/{len(sampled)}]")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            summary["skipped_images"] += 1
            _increment_reason(summary, "load_or_embed_error")
            print(f"  Skipped {fpath.name}: {exc}")

    if ids:
        progress.update(phase="saving", current=0, total=0,
                        detail="Writing to database\u2026")
        collection.upsert(
            ids=ids, embeddings=embeddings, metadatas=metadatas,
        )
        print(f"Indexed {len(ids)} images into ChromaDB.")

    _rebuild_path_index()
    summary["status"] = "completed"
    summary["completed_at"] = _now_iso()
    summary["indexed_images"] = len(ids)
    summary["kept_images"] = len(ids)
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
    # Load CLIP model
    runtime["device"] = _get_device()
    print(f"Loading CLIP model ({CLIP_MODEL}) on {runtime['device']} …")
    runtime["model"], _, runtime["preprocess"] = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=runtime["device"],
    )
    runtime["tokenizer"] = open_clip.get_tokenizer(CLIP_MODEL)
    runtime["model"].eval()
    print("CLIP model loaded.")

    _init_chroma_collection()
    _rebuild_path_index()
    count = runtime["chroma_collection"].count()
    print(f"Loaded {count} existing indexed images. Use Reset DB to reindex.")

    yield  # app runs


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Image Vector Search", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
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
):
    """Search images by natural-language query and return ranked results."""
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
    count = collection.count()
    if count == 0:
        return {"query": q, "results": []}

    # Over-fetch when date filtering so we have enough after trimming
    fetch_n = min(n * 4, count) if (date_from or date_to) else min(n, count)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=fetch_n,
        where=where,
    )

    items = []
    for fid, meta, dist in zip(
        results["ids"][0], results["metadatas"][0], results["distances"][0],
    ):
        # Post-query date filter (ChromaDB $gte/$lte needs numeric types)
        best = meta.get("best_date", "") or meta.get("date_modified", "")
        if date_from and best < date_from:
            continue
        if date_to and best > date_to + "T99":
            continue

        item = {
            "id": fid,
            "filename": meta.get("filename", ""),
            "score": round(1 - dist, 4),
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

    return {"query": q, "results": items}


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


@app.get("/api/reindex")
def reindex():
    """Re-scan photos directory and index new images."""
    _run_indexing(reset_db=False, trigger="reindex")
    return _stats_payload()


@app.post("/api/reset")
def reset_database():
    """Delete the persisted database and rebuild it from the source folder."""
    if runtime["progress"]["phase"] != "idle":
        return {"status": "already_running", "message": "Indexing already in progress."}

    def _bg_rebuild():
        try:
            _run_indexing(reset_db=True, trigger="reset")
        except Exception as exc:
            print(f"Background rebuild failed: {exc}")

    threading.Thread(target=_bg_rebuild, daemon=True).start()
    return {"status": "started", "message": "Rebuild started in background."}
