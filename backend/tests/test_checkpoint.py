"""Tests for resumable indexing checkpoint and batched upsert support."""

# pyright: reportPrivateUsage=false

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server


class _FakeCollection:
    """Minimal stand-in for a Chroma collection."""

    def __init__(self, count: int = 0, ids: list | None = None):
        self._count = count
        self._ids = ids or []
        self._upserted: list[dict] = []

    def count(self) -> int:
        return self._count

    def get(self, include=None):  # noqa: ARG002
        return {"ids": list(self._ids), "metadatas": [{}] * len(self._ids)}

    def upsert(self, ids=None, embeddings=None, metadatas=None):
        self._upserted.append({
            "ids": ids, "embeddings": embeddings, "metadatas": metadatas,
        })
        self._ids.extend(ids)
        self._count += len(ids)


# -----------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------
def test_save_and_load_checkpoint(tmp_path, monkeypatch):
    """Checkpoint round-trip: save then load returns same data."""
    monkeypatch.setattr(server, "CHECKPOINT_PATH", tmp_path / ".ckpt.json")
    monkeypatch.setattr(server, "CHROMA_DIR", tmp_path)

    server._save_checkpoint(
        trigger="reset", started_at="2026-03-31T10:00:00",
        total_sampled=200, committed_count=50,
    )

    loaded = server._load_checkpoint()
    assert loaded is not None
    assert loaded["version"] == 1
    assert loaded["trigger"] == "reset"
    assert loaded["total_sampled"] == 200
    assert loaded["committed_count"] == 50
    assert loaded["started_at"] == "2026-03-31T10:00:00"
    assert "last_batch_at" in loaded


def test_load_checkpoint_returns_none_when_missing(tmp_path, monkeypatch):
    """No checkpoint file means None."""
    monkeypatch.setattr(server, "CHECKPOINT_PATH", tmp_path / ".ckpt.json")
    assert server._load_checkpoint() is None


def test_load_checkpoint_handles_corrupt_json(tmp_path, monkeypatch):
    """Corrupt JSON gracefully returns None."""
    ckpt = tmp_path / ".ckpt.json"
    ckpt.write_text("NOT VALID JSON{{{", encoding="utf-8")
    monkeypatch.setattr(server, "CHECKPOINT_PATH", ckpt)
    assert server._load_checkpoint() is None


def test_load_checkpoint_rejects_wrong_version(tmp_path, monkeypatch):
    """Unknown version returns None."""
    ckpt = tmp_path / ".ckpt.json"
    ckpt.write_text(json.dumps({"version": 99}), encoding="utf-8")
    monkeypatch.setattr(server, "CHECKPOINT_PATH", ckpt)
    assert server._load_checkpoint() is None


def test_delete_checkpoint_removes_file(tmp_path, monkeypatch):
    """_delete_checkpoint removes the file."""
    ckpt = tmp_path / ".ckpt.json"
    ckpt.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(server, "CHECKPOINT_PATH", ckpt)
    server._delete_checkpoint()
    assert not ckpt.exists()


def test_delete_checkpoint_noop_when_missing(tmp_path, monkeypatch):
    """_delete_checkpoint is a no-op when the file doesn't exist."""
    monkeypatch.setattr(server, "CHECKPOINT_PATH", tmp_path / ".ckpt.json")
    server._delete_checkpoint()  # should not raise


def test_save_checkpoint_uses_atomic_rename(tmp_path, monkeypatch):
    """Checkpoint write uses a .tmp intermediate for atomic replace."""
    monkeypatch.setattr(server, "CHECKPOINT_PATH", tmp_path / ".ckpt.json")
    monkeypatch.setattr(server, "CHROMA_DIR", tmp_path)

    server._save_checkpoint(
        trigger="reindex", started_at="2026-03-31T11:00:00",
        total_sampled=100, committed_count=25,
    )

    # .tmp should be cleaned up by replace()
    assert not (tmp_path / ".ckpt.tmp").exists()
    assert (tmp_path / ".ckpt.json").exists()


# -----------------------------------------------------------------------
# Summary shape
# -----------------------------------------------------------------------
def test_new_index_summary_has_resume_fields():
    """Summary includes batches_committed and resumed_from."""
    summary = server._new_index_summary(reset_db=False, trigger="reindex")
    assert "batches_committed" in summary
    assert "resumed_from" in summary
    assert summary["batches_committed"] == 0
    assert summary["resumed_from"] == 0


# -----------------------------------------------------------------------
# Stats payload
# -----------------------------------------------------------------------
def test_stats_payload_includes_interrupted_checkpoint(monkeypatch):
    """Stats payload exposes interrupted_checkpoint from runtime."""
    ckpt_data = {
        "version": 1, "trigger": "reset",
        "started_at": "2026-03-31T10:00:00",
        "total_sampled": 200, "committed_count": 100,
        "last_batch_at": "2026-03-31T10:05:00",
    }
    monkeypatch.setitem(server.runtime, "interrupted_checkpoint", ckpt_data)
    monkeypatch.setitem(server.runtime, "chroma_collection",
                        _FakeCollection(count=100))
    monkeypatch.setitem(server.runtime, "caption_collection",
                        _FakeCollection(count=50))
    monkeypatch.setitem(server.runtime, "caption_model", None)
    monkeypatch.setitem(server.runtime, "last_index_summary", {})
    monkeypatch.setitem(server.runtime, "rebuild_status", {
        "is_running": False, "last_trigger": None,
        "last_started_at": None, "last_completed_at": None, "last_error": None,
    })

    payload = server._stats_payload()
    assert payload["interrupted_checkpoint"] == ckpt_data


def test_stats_payload_no_interrupted_checkpoint(monkeypatch):
    """Stats payload returns None when no interrupted checkpoint."""
    monkeypatch.setitem(server.runtime, "interrupted_checkpoint", None)
    monkeypatch.setitem(server.runtime, "chroma_collection",
                        _FakeCollection(count=0))
    monkeypatch.setitem(server.runtime, "caption_collection",
                        _FakeCollection(count=0))
    monkeypatch.setitem(server.runtime, "caption_model", None)
    monkeypatch.setitem(server.runtime, "last_index_summary", {})
    monkeypatch.setitem(server.runtime, "rebuild_status", {
        "is_running": False, "last_trigger": None,
        "last_started_at": None, "last_completed_at": None, "last_error": None,
    })

    payload = server._stats_payload()
    assert payload["interrupted_checkpoint"] is None


# -----------------------------------------------------------------------
# Reset clears checkpoint
# -----------------------------------------------------------------------
def test_reset_chroma_store_deletes_checkpoint(tmp_path, monkeypatch):
    """_reset_chroma_store removes checkpoint and clears runtime flag."""
    ckpt = tmp_path / ".ckpt.json"
    ckpt.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(server, "CHECKPOINT_PATH", ckpt)
    monkeypatch.setitem(server.runtime, "interrupted_checkpoint", {"some": "data"})
    monkeypatch.setitem(server.runtime, "chroma_client", None)
    monkeypatch.setitem(server.runtime, "chroma_collection", None)
    monkeypatch.setitem(server.runtime, "caption_collection", None)
    monkeypatch.setitem(server.runtime, "path_index", {})
    monkeypatch.setattr(server, "CHROMA_DIR", tmp_path)

    server._reset_chroma_store()

    assert not ckpt.exists()
    assert server.runtime["interrupted_checkpoint"] is None


# -----------------------------------------------------------------------
# Reindex endpoint
# -----------------------------------------------------------------------
def test_reindex_returns_already_running_when_busy(monkeypatch):
    """Reindex rejects requests when indexing is already in progress."""
    monkeypatch.setitem(server.runtime, "progress", {
        "phase": "indexing", "current": 10, "total": 50, "detail": "",
    })

    payload = server.reindex()

    assert payload["status"] == "already_running"


def test_reindex_starts_background_thread(monkeypatch):
    """Reindex starts a background thread when idle."""
    monkeypatch.setitem(server.runtime, "progress", {
        "phase": "idle", "current": 0, "total": 0, "detail": "",
    })

    payload = server.reindex()

    assert payload["status"] == "started"
    assert "message" in payload
    assert server.runtime["progress"]["phase"] == "starting"
    assert server.runtime["progress"]["detail"] == "Preparing reindex…"


def test_reset_starts_background_thread_with_starting_phase(monkeypatch):
    """Reset marks progress immediately so the UI doesn't miss the launch race."""
    monkeypatch.setitem(server.runtime, "progress", {
        "phase": "idle", "current": 0, "total": 0, "detail": "",
    })

    payload = server.reset_database()

    assert payload["status"] == "started"
    assert server.runtime["progress"]["phase"] == "starting"
    assert server.runtime["progress"]["detail"] == "Preparing rebuild…"


def test_index_photos_updates_path_index_after_each_batch(tmp_path, monkeypatch):
    """Committed photos become servable before the full indexing pass finishes."""
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake")

    collection = _FakeCollection(count=0)
    checkpoint_snapshots = []

    monkeypatch.setattr(server, "PHOTOS_DIR", tmp_path)
    monkeypatch.setattr(server, "MAX_INDEX_IMAGES", 1)
    monkeypatch.setattr(server, "INDEX_BATCH_SIZE", 1)
    monkeypatch.setitem(server.runtime, "chroma_collection", collection)
    monkeypatch.setitem(server.runtime, "caption_collection", _FakeCollection(count=0))
    monkeypatch.setitem(server.runtime, "path_index", {})
    monkeypatch.setitem(server.runtime, "progress", {
        "phase": "idle", "current": 0, "total": 0, "detail": "",
    })

    monkeypatch.setattr(server, "_discover_images", lambda: [image_path])
    monkeypatch.setattr(server, "_sample_images", lambda files, limit: files[:limit])
    monkeypatch.setattr(server, "_load_image", lambda _path: object())
    monkeypatch.setattr(server, "_extract_metadata", lambda path: {
        "filename": path.name,
        "path": str(path),
        "relative_path": path.name,
        "folder": "",
        "date_modified": "2026-03-31T09:30:00",
    })
    monkeypatch.setattr(server, "_generate_caption", lambda _img: "")
    monkeypatch.setattr(server, "_metadata_text", lambda _meta: "sample")
    monkeypatch.setattr(server, "_fused_embedding", lambda _img, _text: [0.1, 0.2, 0.3])
    monkeypatch.setattr(server, "_file_id", lambda _path: "photo-1")
    monkeypatch.setattr(server, "_delete_checkpoint", lambda: None)
    monkeypatch.setattr(server, "_rebuild_path_index", lambda: None)

    def fake_save_checkpoint(**_kwargs):
        checkpoint_snapshots.append(dict(server.runtime["path_index"]))

    monkeypatch.setattr(server, "_save_checkpoint", fake_save_checkpoint)

    summary = server._index_photos(reset_db=False, trigger="reindex")

    assert summary["status"] == "completed"
    assert checkpoint_snapshots
    assert checkpoint_snapshots[0]["photo-1"] == image_path
    assert server.runtime["path_index"]["photo-1"] == image_path


def test_filter_new_candidates_reports_resume_progress(tmp_path):
    """Resume scans should report progress while checking already indexed files."""
    files = [tmp_path / f"img-{idx}.jpg" for idx in range(3)]
    progress = {"phase": "idle", "current": 0, "total": 0, "detail": ""}
    ids = {"keep-1"}
    by_path = {
        files[0]: "keep-1",
        files[1]: "new-2",
        files[2]: "new-3",
    }

    original_file_id = server._file_id
    try:
        server._file_id = lambda path: by_path[path]
        new_candidates = server._filter_new_candidates(files, ids, progress)
    finally:
        server._file_id = original_file_id

    assert new_candidates == files[1:]
    assert progress["phase"] == "resuming"
    assert progress["current"] == 3
    assert progress["total"] == 3
    assert "found 2 new candidates" in progress["detail"]
