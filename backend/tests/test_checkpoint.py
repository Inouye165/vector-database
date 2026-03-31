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
