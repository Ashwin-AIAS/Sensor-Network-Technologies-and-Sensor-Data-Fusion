# Dockerfile.gui
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-gui.txt /app/requirements-gui.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements-gui.txt

COPY . /app

CMD ["python", "face solution2.py"]
