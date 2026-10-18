FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install fitcore package first
COPY packages/fitcore /packages/fitcore
RUN pip install --no-cache-dir /packages/fitcore

# Install api service requirements
COPY services/api/requirements.txt .
RUN grep -v "\-e \.\./\.\./packages/fitcore" requirements.txt > reqs.txt && \
    pip install --no-cache-dir -r reqs.txt

# Copy service source code
COPY services/api/ /app/

# apilib/fit_mesh.py, garment_routes.py, and main.py import
# services.geometry.cloud_client unconditionally (a lightweight Firestore/
# Storage client abstraction with no torch/smplx/mediapipe dependency, unlike
# the rest of the geometry service). Ship just that module, under the same
# package path it has in the repo, so `services.geometry.cloud_client`
# resolves at /app the same way it does from the repo root in local dev. The
# heavier geometry modules (fitting/garment_heatmap/exporter/poses/worker)
# are deliberately NOT copied here: fit_mesh.py and main.py already guard
# those imports with try/except and return 503 if unavailable, since this
# image does not install torch/smplx/mediapipe.
COPY services/geometry/__init__.py /app/services/geometry/__init__.py
COPY services/geometry/cloud_client.py /app/services/geometry/cloud_client.py

EXPOSE 8000

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
