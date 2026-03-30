"""Image vector database backend – FastAPI + ChromaDB + OpenCLIP.

Indexes photos into a vector database using CLIP embeddings so users
can search for images with natural language queries like
"show me images of a dog at the park".
"""

import io
import os
import hashlib
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import open_clip
import PIL.Image
import pillow_heif
import torch
import chromadb
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PHOTOS_DIR = Path(__file__).resolve().parent.parent / "photos"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp",
                    ".heic", ".heif", ".webp", ".tiff", ".tif"}
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# ---------------------------------------------------------------------------
# Register HEIF/HEIC opener with Pillow
# ---------------------------------------------------------------------------
pillow_heif.register_heif_opener()

# ---------------------------------------------------------------------------
# Globals (populated at startup)
# ---------------------------------------------------------------------------
model = None
preprocess = None
tokenizer = None
device = None
chroma_collection = None


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_image(path: Path) -> PIL.Image.Image:
    """Load an image file (including HEIC) and return as RGB PIL Image."""
    img = PIL.Image.open(path)
    img = img.convert("RGB")
    return img


def _embed_image(img: PIL.Image.Image) -> list[float]:
    """Return a normalised CLIP embedding for a PIL Image."""
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten().tolist()


def _embed_text(text: str) -> list[float]:
    """Return a normalised CLIP embedding for a text query."""
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten().tolist()


def _file_id(path: Path) -> str:
    """Deterministic ID for a file based on its path."""
    return hashlib.sha256(str(path).encode()).hexdigest()[:16]


def _index_photos():
    """Walk the photos dir and upsert any new images into ChromaDB."""
    if not PHOTOS_DIR.exists():
        print(f"Photos directory not found: {PHOTOS_DIR}")
        return

    existing_ids = set(chroma_collection.get()["ids"])
    files = [
        p for p in sorted(PHOTOS_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    new_files = [f for f in files if _file_id(f) not in existing_ids]

    if not new_files:
        print(f"All {len(existing_ids)} images already indexed.")
        return

    print(f"Indexing {len(new_files)} new images …")
    ids, embeddings, metadatas = [], [], []
    for i, fpath in enumerate(new_files):
        try:
            img = _load_image(fpath)
            emb = _embed_image(img)
            ids.append(_file_id(fpath))
            embeddings.append(emb)
            metadatas.append({"filename": fpath.name, "path": str(fpath)})
            if (i + 1) % 10 == 0 or i + 1 == len(new_files):
                print(f"  [{i+1}/{len(new_files)}]")
        except Exception as exc:
            print(f"  Skipped {fpath.name}: {exc}")

    if ids:
        chroma_collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
        print(f"Indexed {len(ids)} images into ChromaDB.")


# ---------------------------------------------------------------------------
# App lifespan – load model & index on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocess, tokenizer, device, chroma_collection

    # Load CLIP model
    device = _get_device()
    print(f"Loading CLIP model ({CLIP_MODEL}) on {device} …")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=device
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.eval()
    print("CLIP model loaded.")

    # Init ChromaDB
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = client.get_or_create_collection(
        name="photos",
        metadata={"hnsw:space": "cosine"},
    )

    # Index images
    _index_photos()

    yield  # app runs

    # Cleanup (nothing needed)


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
def search_images(q: str = Query(..., min_length=1), n: int = Query(20, ge=1, le=100)):
    """Search images by natural-language query and return ranked results."""
    query_emb = _embed_text(q)
    results = chroma_collection.query(
        query_embeddings=[query_emb],
        n_results=min(n, chroma_collection.count()),
    )
    items = []
    for fid, meta, dist in zip(
        results["ids"][0], results["metadatas"][0], results["distances"][0]
    ):
        items.append({
            "id": fid,
            "filename": meta["filename"],
            "score": round(1 - dist, 4),  # cosine similarity
            "url": f"/api/photo/{meta['filename']}",
        })
    return {"query": q, "results": items}


@app.get("/api/photo/{filename}")
def get_photo(filename: str):
    """Serve a photo file (with HEIC→JPEG conversion for browser display)."""
    # Sanitize filename to prevent path traversal
    safe_name = Path(filename).name
    filepath = PHOTOS_DIR / safe_name
    if not filepath.exists() or not filepath.is_file():
        return Response(status_code=404, content="Not found")

    # Verify the resolved path is inside PHOTOS_DIR
    try:
        filepath.resolve().relative_to(PHOTOS_DIR.resolve())
    except ValueError:
        return Response(status_code=403, content="Forbidden")

    suffix = filepath.suffix.lower()
    if suffix in (".heic", ".heif"):
        # Convert HEIC to JPEG for browser compatibility
        img = _load_image(filepath)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return Response(content=buf.getvalue(), media_type="image/jpeg")

    # Serve other formats directly
    content_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif",
        ".webp": "image/webp", ".bmp": "image/bmp",
        ".tiff": "image/tiff", ".tif": "image/tiff",
    }
    media_type = content_types.get(suffix, "application/octet-stream")
    return Response(content=filepath.read_bytes(), media_type=media_type)


@app.get("/api/stats")
def stats():
    """Return collection stats."""
    return {
        "indexed_images": chroma_collection.count(),
        "photos_dir": str(PHOTOS_DIR),
    }


@app.get("/api/reindex")
def reindex():
    """Re-scan photos directory and index new images."""
    _index_photos()
    return {"indexed_images": chroma_collection.count()}
