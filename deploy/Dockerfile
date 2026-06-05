FROM python:3.11-slim-bookworm

WORKDIR /app

# OpenCV + MediaPipe runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY src/ src/
COPY configs/ configs/
COPY outputs/best.pt outputs/best.pt

ENV PYTHONUNBUFFERED=1
ENV MODEL_CHECKPOINT=outputs/best.pt
ENV MODEL_CONFIG=configs/train/cpu_fast.yaml
ENV DEVICE=cpu
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
