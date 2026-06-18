"""Optional HTTP service wrapper around :func:`tomeforge.convert`.

Run it as a sidecar so another app can offload PDF→EPUB conversion (incl. the
heavyweight PyMuPDF + optional Ollama-OCR steps) instead of bundling them itself::

    tomeforge serve --host 0.0.0.0 --port 8400
    # or, in Docker, the image's default CMD

The conversion engine is unchanged — this only adds a thin job API:

    POST   /convert            multipart file + options  -> {"job_id": ...}
    GET    /jobs/{id}          -> {status, phase, engine, scanned, error}
    GET    /jobs/{id}/result   -> the EPUB (FileResponse) once status == "done"
    DELETE /jobs/{id}          -> drop the job's temp files
    GET    /healthz            -> liveness + whether Calibre is on PATH

Jobs run in background threads with their temp files under a single base dir;
``phase`` surfaces ``OCR page N/M`` by watching the resumable ``pages/`` cache.
The registry is in-memory — the sidecar is a stateless worker, not a store, so a
restart drops in-flight jobs (the caller re-submits). FastAPI/uvicorn are the
optional ``service`` extra; everything here is import-light so the CLI/library
keep working without them installed.
"""

import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from tomeforge import __version__
from tomeforge.converter import ConversionError, calibre_available, convert

# Status values mirrored by callers — keep the strings stable.
QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"


@dataclass
class _Job:
    id: str
    dir: Path  # per-job scratch: holds the upload, work/, and out.epub
    status: str = QUEUED
    phase: str | None = None
    engine: str | None = None
    scanned: bool = False
    error: str | None = None
    total_pages: int = 0  # PDF page count, for "OCR page N/M"
    output: Path | None = None

    @property
    def work_dir(self) -> Path:
        return self.dir / "work"

    def public(self) -> dict:
        return {
            "job_id": self.id,
            "status": self.status,
            "phase": self.effective_phase(),
            "engine": self.engine,
            "scanned": self.scanned,
            "error": self.error,
        }

    def effective_phase(self) -> str | None:
        """Live phase. While running, derive OCR progress from the pages cache
        (pdf2md writes ``work/pages/page-NNN.md`` per transcribed page)."""
        if self.status != RUNNING:
            return self.phase
        pages_dir = self.work_dir / "pages"
        try:
            done = sum(1 for _ in pages_dir.glob("page-*.md")) if pages_dir.is_dir() else 0
        except OSError:
            done = 0
        if done:
            total = self.total_pages
            return f"OCR page {min(done, total) if total else done}/{total or '?'}"
        return self.phase or "Converting…"


_jobs: dict[str, _Job] = {}
_lock = threading.Lock()


def _pdf_page_count(path: Path) -> int:
    """Page count for OCR progress; best-effort (0 if PyMuPDF/file unavailable)."""
    if path.suffix.lower() != ".pdf":
        return 0
    try:
        import fitz  # PyMuPDF
    except Exception:
        return 0
    try:
        doc = fitz.open(str(path))
        try:
            return doc.page_count
        finally:
            doc.close()
    except Exception:
        return 0


def _run(job: _Job, src: Path, opts: dict) -> None:
    """Worker body: run a single conversion, recording the outcome on the job."""
    out = job.dir / "out.epub"
    try:
        with _lock:
            job.status = RUNNING
            job.phase = "Converting…"
        result = convert(src, out, work_dir=job.work_dir, quiet=True, **opts)
        with _lock:
            job.status = DONE
            job.phase = None
            job.engine = result.engine
            job.scanned = result.scanned
            job.output = Path(result.output)
    except ConversionError as e:
        with _lock:
            job.status, job.phase, job.error = FAILED, None, str(e)
    except Exception as e:  # never leave a job stuck in "running"
        with _lock:
            job.status, job.phase, job.error = FAILED, None, f"{type(e).__name__}: {e}"


def create_app(base_dir: str | Path | None = None):
    """Build the FastAPI app. Imported lazily so the CLI/library don't require the
    `service` extra (fastapi/uvicorn/python-multipart)."""
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.responses import FileResponse
    except ModuleNotFoundError as e:  # pragma: no cover - import-guard
        raise SystemExit(
            "the HTTP service needs the 'service' extra: pip install 'tomeforge[service]'"
        ) from e

    base = Path(base_dir) if base_dir else Path(tempfile.gettempdir()) / "tomeforge-jobs"
    base.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="tomeforge", version=__version__)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "version": __version__, "calibre": calibre_available()}

    @app.post("/convert")
    async def start_convert(
        file: UploadFile = File(...),
        ocr: str = Form("auto"),
        ollama_host: str | None = Form(None),
        model: str = Form("deepseek-ocr:3b"),
        dpi: int = Form(150),
        ocr_timeout: int = Form(600),
        num_ctx: int = Form(8192),
    ) -> dict:
        if not calibre_available():
            raise HTTPException(503, "Calibre (ebook-convert) is not installed in the sidecar")
        if ocr not in ("auto", "always", "never"):
            raise HTTPException(422, "ocr must be one of: auto, always, never")

        job_id = uuid.uuid4().hex
        job_dir = base / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        # Preserve the suffix — convert() dispatches on it (.pdf vs direct formats).
        suffix = Path(file.filename or "input").suffix.lower() or ".bin"
        src = job_dir / f"input{suffix}"
        with src.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)

        job = _Job(id=job_id, dir=job_dir, total_pages=_pdf_page_count(src))
        with _lock:
            _jobs[job_id] = job
        opts = dict(
            ocr=ocr, ollama_host=ollama_host or None, model=model,
            dpi=dpi, ocr_timeout=ocr_timeout, num_ctx=num_ctx,
        )
        threading.Thread(target=_run, args=(job, src, opts), daemon=True).start()
        return job.public()

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        with _lock:
            job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.public()

    @app.get("/jobs/{job_id}/result")
    def job_result(job_id: str):
        with _lock:
            job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        if job.status != DONE or not job.output or not job.output.exists():
            raise HTTPException(409, f"job not ready (status={job.status})")
        return FileResponse(
            job.output, media_type="application/epub+zip", filename="converted.epub"
        )

    @app.delete("/jobs/{job_id}")
    def job_delete(job_id: str) -> dict:
        with _lock:
            job = _jobs.pop(job_id, None)
        if job is None:
            raise HTTPException(404, "no such job")
        shutil.rmtree(job.dir, ignore_errors=True)
        return {"deleted": job_id}

    return app


def main(argv: list[str] | None = None) -> int:
    """`tomeforge serve` entry point."""
    import argparse

    p = argparse.ArgumentParser(
        prog="tomeforge serve",
        description="Run tomeforge as an HTTP conversion sidecar.",
    )
    p.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8400, help="bind port (default: 8400)")
    p.add_argument("--jobs-dir", default=None, help="scratch dir for job files (default: temp)")
    args = p.parse_args(argv)

    try:
        import uvicorn
    except ModuleNotFoundError:
        print("error: the HTTP service needs the 'service' extra: "
              "pip install 'tomeforge[service]'")
        return 2

    uvicorn.run(create_app(args.jobs_dir), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
