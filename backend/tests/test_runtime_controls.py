"""Tests for runtime controls that reduce main-app resource contention."""

# pyright: reportMissingImports=false, reportPrivateUsage=false

import importlib.util
import sys
from pathlib import Path
from typing import Any, cast


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SERVER_SPEC = importlib.util.spec_from_file_location("server", ROOT / "server.py")
assert SERVER_SPEC is not None and SERVER_SPEC.loader is not None
server = importlib.util.module_from_spec(SERVER_SPEC)
SERVER_SPEC.loader.exec_module(server)
server_api = cast(Any, server)


def test_get_device_respects_cpu_override(monkeypatch):
    """Explicit cpu override wins even if CUDA is reported available."""
    monkeypatch.setattr(server, "VECTOR_DB_DEVICE", "cpu")
    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: True)

    assert server_api._get_device() == "cpu"


def test_stats_payload_reports_runtime_limits(monkeypatch):
    """Stats expose the current runtime throttling configuration."""

    class _FakeCollection:
        def __init__(self, count: int):
            self._count = count

        def count(self) -> int:
            return self._count

    monkeypatch.setattr(server, "VECTOR_DB_DEVICE", "cpu")
    monkeypatch.setattr(server, "VECTOR_DB_CPU_THREADS", 2)
    monkeypatch.setattr(server, "INDEX_THROTTLE_MS", 15)
    monkeypatch.setattr(server, "INDEX_BATCH_COOLDOWN_MS", 100)
    monkeypatch.setitem(server.runtime, "chroma_collection", _FakeCollection(count=1))
    monkeypatch.setitem(server.runtime, "caption_collection", _FakeCollection(count=0))
    monkeypatch.setitem(server.runtime, "caption_model", None)
    monkeypatch.setitem(server.runtime, "last_index_summary", {})
    monkeypatch.setitem(server.runtime, "rebuild_status", {
        "is_running": False,
        "last_trigger": None,
        "last_started_at": None,
        "last_completed_at": None,
        "last_error": None,
    })

    payload = server_api._stats_payload()

    assert payload["runtime_limits"] == {
        "device_override": "cpu",
        "cpu_threads": 2,
        "index_throttle_ms": 15,
        "index_batch_cooldown_ms": 100,
    }


def test_configure_runtime_limits_applies_thread_limit(monkeypatch):
    """Configured thread limits are forwarded to torch exactly once."""
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(server, "VECTOR_DB_CPU_THREADS", 3)
    monkeypatch.setattr(server, "_runtime_limits_configured", False)
    monkeypatch.setattr(server.torch, "set_num_threads", lambda value: calls.append(("threads", value)))
    monkeypatch.setattr(server.torch, "set_num_interop_threads", lambda value: calls.append(("interop", value)))

    server_api._configure_runtime_limits()
    server_api._configure_runtime_limits()

    assert calls == [("threads", 3), ("interop", 1)]