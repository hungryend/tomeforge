# syntax=docker/dockerfile:1.7
#
# tomeforge as an HTTP conversion sidecar (`tomeforge serve`).
# Bundles Calibre (the EPUB step) + PyMuPDF (the PDF→Markdown step). Scanned-PDF
# OCR is delegated to an external Ollama — point --ollama-host / the request's
# ollama_host at one (see docker-compose.yml for an optional local Ollama).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Calibre provides `ebook-convert` (Markdown/MOBI/AZW → EPUB). `upgrade` first so
# the slim base's lagging point-release CVEs (poppler/openssl/…) get patched.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        calibre \
        libxml2 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Drop the base image's stale bundled pip wheel and refresh pip (malicious-wheel
# / archive-traversal CVEs in older pip) before installing.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && find /usr/local/lib -path '*/ensurepip/_bundled/*' -name 'pip-*.whl' -delete

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package + the HTTP service extra (fastapi/uvicorn/python-multipart).
RUN pip install --no-cache-dir ".[service]"

EXPOSE 8400

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8400/healthz || exit 1

CMD ["tomeforge", "serve", "--host", "0.0.0.0", "--port", "8400"]
