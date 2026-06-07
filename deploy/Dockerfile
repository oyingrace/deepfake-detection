FROM python:3.11-slim-bookworm

WORKDIR /app

# OpenCV + MediaPipe runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY src/ src/
COPY configs/ configs/
COPY outputs/best.pt outputs/best.pt

# Inference-only bundle (matches deploy/sync_bundle.sh)
RUN rm -f src/data/stream_ff_dataset.py \
      src/data/build_metadata.py \
      src/data/extract_frames.py \
      src/data/extract_eye_sequences.py \
      src/train/train.py \
      src/train/adversarial.py && \
    rm -rf src/eval src/train src/viz

ENV PYTHONUNBUFFERED=1
ENV MODEL_CHECKPOINT=outputs/best.pt
ENV MODEL_CONFIG=configs/train/cpu_fast.yaml
ENV DEVICE=cpu
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
