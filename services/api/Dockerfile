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

EXPOSE 8000

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
