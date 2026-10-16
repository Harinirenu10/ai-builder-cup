FROM mcr.microsoft.com/playwright/python:v1.49.0-jammy

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8002

# Install fitcore package
COPY packages/fitcore /packages/fitcore
RUN pip install --no-cache-dir /packages/fitcore

# Install reader service requirements
COPY services/reader/requirements.txt .
RUN grep -v "\-e \.\./\.\./packages/fitcore" requirements.txt > reqs.txt && \
    pip install --no-cache-dir -r reqs.txt

# Ensure Playwright browser binaries are installed
RUN playwright install chromium --with-deps

# Copy service source code
COPY services/reader/ /app/

EXPOSE 8002

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
