# Production Deployment Container for Azerbaijani Banknote Vision System
# AI Academy · Final Deep Learning Project · Track 3: Industry Product
# Provides containerized execution for Dashboard, Inference Bridge, and Pipelines.

FROM python:3.12-slim-bookworm

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install essential system dependencies for OpenCV and network tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and configurations
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY configs/data.yaml ./data/processed/data.yaml


# Expose port: 8000 (Mobile Vision Inference Bridge)
EXPOSE 8000

# Default command launches the low-latency Mobile Vision Bridge
CMD ["python", "scripts/mobile_bridge.py", "--host", "0.0.0.0", "--port", "8000"]
