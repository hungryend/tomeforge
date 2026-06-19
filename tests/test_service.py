"""HTTP sidecar tests. The conversion engine itself is stubbed (no Calibre/PDF
needed) so we exercise the job lifecycle: submit → poll → download → delete.
Needs the `service` extra (fastapi) + httpx for TestClient — importorskip both so
the rest of the suite still runs without them."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from tomeforge import service  # noqa: E402
from tomeforge.converter import ConversionError, ConversionResult  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "calibre_available", lambda: True)
    # Each test gets an isolated job registry + scratch dir.
    monkeypatch.setattr(service, "_jobs", {})
    return TestClient(service.create_app(base_dir=tmp_path / "jobs"))


def _stub_convert_ok(monkeypatch):
    """convert() that writes a fake EPUB to `out` and reports the heuristic engine."""

    def fake(src, out=None, *, work_dir=None, **kw):
        Path(out).write_bytes(b"PK\x03\x04 fake-epub")
        return ConversionResult(Path(out), engine="heuristic", scanned=False)

    monkeypatch.setattr(service, "convert", fake)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["calibre"] is True


def test_convert_lifecycle(client, monkeypatch):
    _stub_convert_ok(monkeypatch)

    r = client.post("/convert", files={"file": ("book.pdf", b"%PDF-1.4 ...", "application/pdf")})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # Thread is daemonized; poll until it settles (stub is ~instant).
    for _ in range(50):
        status = client.get(f"/jobs/{job_id}").json()
        if status["status"] in ("done", "failed"):
            break
    assert status["status"] == "done"
    assert status["engine"] == "heuristic"

    result = client.get(f"/jobs/{job_id}/result")
    assert result.status_code == 200
    assert result.content.startswith(b"PK")

    assert client.delete(f"/jobs/{job_id}").status_code == 200
    assert client.get(f"/jobs/{job_id}").status_code == 404


def test_convert_failure_is_recorded(client, monkeypatch):
    def boom(src, out=None, *, work_dir=None, **kw):
        raise ConversionError("no text in this PDF")

    monkeypatch.setattr(service, "convert", boom)

    job_id = client.post("/convert", files={"file": ("x.pdf", b"%PDF", "application/pdf")}).json()[
        "job_id"
    ]
    for _ in range(50):
        status = client.get(f"/jobs/{job_id}").json()
        if status["status"] in ("done", "failed"):
            break
    assert status["status"] == "failed"
    assert "no text" in status["error"]
    # Result is 409 (not ready / failed), not a stale file.
    assert client.get(f"/jobs/{job_id}/result").status_code == 409


def test_bad_ocr_mode_is_422(client):
    r = client.post(
        "/convert",
        files={"file": ("x.pdf", b"%PDF", "application/pdf")},
        data={"ocr": "sometimes"},
    )
    assert r.status_code == 422


def test_unknown_job_is_404(client):
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/jobs/nope/result").status_code == 404
    assert client.delete("/jobs/nope").status_code == 404
