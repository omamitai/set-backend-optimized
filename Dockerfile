FROM python:3.9-slim AS builder

WORKDIR /build

# Install essentials for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Create a virtual environment and install dependencies
# This helps minimize final image size
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Second stage: runtime image
FROM python:3.9-slim

WORKDIR /app

# Set environment variables for stability and performance
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8080 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=1 \
    PYTHONOPTIMIZE=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    TENSORRT_LIBRARY_PATH="" \
    XLA_FLAGS="--xla_gpu_cuda_data_dir=/"

# Install only runtime dependencies for OpenCV (minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code and models
# Keep models in a separate layer for better caching
COPY models/ ./models/
COPY utils/ ./utils/
COPY app.py ./

# Create a non-root user for better security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose the port
EXPOSE 8080

# Start with a direct uvicorn command for more reliability
CMD ["sh", "-c", "python app.py"]
