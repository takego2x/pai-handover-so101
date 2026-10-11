# Reproducible CPU-only runtime for simulation, evaluation and CI.
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libegl1 libosmesa6 libglib2.0-0 libx11-6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# headless rendering without a GPU
ENV PYOPENGL_PLATFORM=osmesa \
    PYTHONUNBUFFERED=1
