# ==============================================================================
# CCTV Intelligence & Multi-Modal Surveillance Platform Dockerfile
# Multi-arch compatible (CPU-optimized, production-hardened)
# ==============================================================================

FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HOST=0.0.0.0

# Install critical OS libraries for OpenCV, video decoding, and network health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install lightweight PyTorch CPU wheels first for faster build times & lean image size
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create dedicated runtime user and prepare persistent directories
RUN useradd -m -u 1001 -s /bin/bash cctv_user && \
    mkdir -p /workspace/data/captures \
             /workspace/data/targets \
             /workspace/data/suspects \
             /workspace/data/tracks \
             /workspace/models && \
    chown -R cctv_user:cctv_user /workspace

# Copy codebase
COPY . .

# Set permissions
RUN chown -R cctv_user:cctv_user /workspace

USER cctv_user

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/watchlist/status || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
