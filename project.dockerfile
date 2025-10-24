# Dockerfile.headless
FROM python:3.10-slim

# Environment setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements-headless.txt /app/requirements-headless.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements-headless.txt

# Copy the rest of the application
COPY . /app

# Default command
CMD ["python", "face solution2.py"]
