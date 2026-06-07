FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MODEL_CHECKPOINT=outputs/best.pt \
    MODEL_CONFIG=configs/train/cpu_fast.yaml \
    DEVICE=cpu \
    PORT=7860

WORKDIR $HOME/app

COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user api/ api/
COPY --chown=user src/ src/
COPY --chown=user configs/ configs/
COPY --chown=user outputs/best.pt outputs/best.pt

RUN rm -f src/data/stream_ff_dataset.py \
      src/data/build_metadata.py \
      src/data/extract_frames.py \
      src/data/extract_eye_sequences.py \
      src/train/train.py \
      src/train/adversarial.py && \
    rm -rf src/eval src/train src/viz

EXPOSE 7860

CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
