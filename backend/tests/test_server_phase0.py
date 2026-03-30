"""Regression tests for phase 0 reset/rebuild support."""

# pyright: reportPrivateUsage=false

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server


class FakeCollection:
    """Simple stand-in for a Chroma collection in tests."""

    def __init__(self, count: int):
        self._count = count

    def count(self) -> int:
        return self._count


def test_increment_reason_accumulates_counts():
    summary = server._new_index_summary(reset_db=True, trigger="reset")

    server._increment_reason(summary, "load_or_embed_error")
    server._increment_reason(summary, "load_or_embed_error", 2)

    assert summary["reason_counts"] == {"load_or_embed_error": 3}


def test_delete_database_dir_removes_existing_directory(tmp_path, monkeypatch):
    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    (chroma_dir / "index.bin").write_text("data", encoding="utf-8")
    monkeypatch.setattr(server, "CHROMA_DIR", chroma_dir)

    server._delete_database_dir()

    assert not chroma_dir.exists()


def test_stats_payload_without_collection(monkeypatch):
    monkeypatch.setitem(server.runtime, "chroma_collection", None)
    monkeypatch.setitem(server.runtime, "last_index_summary", {"status": "completed"})
    monkeypatch.setitem(
        server.runtime,
        "rebuild_status",
        {
            "is_running": False,
            "last_trigger": "startup",
            "last_started_at": "2026-03-30T10:00:00",
            "last_completed_at": "2026-03-30T10:01:00",
            "last_error": None,
        },
    )

    payload = server._stats_payload()

    assert payload["indexed_images"] == 0
    assert payload["photos_dir"] == str(server.PHOTOS_DIR)
    assert payload["last_index_summary"]["status"] == "completed"


def test_reset_database_starts_background_rebuild(monkeypatch):
    monkeypatch.setitem(server.runtime, "chroma_collection", FakeCollection(count=12))
    monkeypatch.setitem(server.runtime, "progress", {
        "phase": "idle", "current": 0, "total": 0, "detail": "",
    })

    payload = server.reset_database()

    assert payload["status"] == "started"
    assert "message" in payload