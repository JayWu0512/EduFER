FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    EDUFER_HOST=0.0.0.0 \
    EDUFER_PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system edufer && useradd --system --gid edufer --create-home edufer

COPY requirements.txt README.md LICENSE ./
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY config ./config
COPY data ./data
COPY scripts ./scripts
COPY src ./src

RUN python scripts/download_face_model.py
RUN chown -R edufer:edufer /app

USER edufer

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import json, urllib.request; data=json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/health')); assert data['status'] == 'ok'"

CMD ["uvicorn", "edufer.app:app", "--host", "0.0.0.0", "--port", "8000"]
