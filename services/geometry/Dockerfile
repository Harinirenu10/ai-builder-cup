FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8001

# Install system runtime dependencies for OpenCV and numerical libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pre-download MediaPipe pose landmarker heavy task model at build time
RUN mkdir -p /app/models && \
    curl -fsSL https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task \
    -o /app/models/pose_landmarker_heavy.task

# Install fitcore package
COPY packages/fitcore /packages/fitcore
RUN pip install --no-cache-dir /packages/fitcore

# Install geometry service requirements with CPU-only PyTorch
COPY services/geometry/requirements.txt .
RUN grep -v "\-e \.\./\.\./packages/fitcore" requirements.txt > reqs.txt && \
    pip install --no-cache-dir -r reqs.txt

# Copy service source code
COPY services/geometry/ /app/

EXPOSE 8001

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
