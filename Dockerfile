# syntax=docker/dockerfile:1.7

FROM python:3.11-slim

# System deps: ffmpeg for pydub, libsndfile for soundfile, and basic build tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       libsndfile1 \
       git \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first to leverage Docker layer cache
COPY requirements.txt ./

# Install Python dependencies (CPU by default)
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY pyproject.toml ./
COPY src ./src
COPY examples ./examples
COPY README.md ./README.md

# Install package in editable mode so module is importable
RUN pip install --no-cache-dir -e .

# Recommended cache location for Hugging Face/Transformers (can be mounted as volume)
ENV TRANSFORMERS_CACHE=/models \
    HF_HOME=/models

# Default workdir for running commands with mounted host volume
WORKDIR /work

# Entrypoint runs the CLI; pass arguments after `docker run ... -- <args>` or directly
ENTRYPOINT ["python", "-m", "soundfree.cli"]

