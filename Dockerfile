# Build stage - use CUDA-enabled Python image for compilation
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS builder

WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y python3.10 python3.10-venv python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies to /root/.local (user site-packages)
RUN python -m pip install --upgrade pip && \
    python -m pip install --user --no-cache-dir -r requirements.txt


# Final stage - use CUDA-enabled runtime image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python and pip in final stage
RUN apt-get update && apt-get install -y python3.10 python3.10-venv python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Set PATH to include user site-packages and local bin
ENV PATH=/root/.local/bin:/usr/bin:$PATH \
    PYTHONPATH=/root/.local/lib/python3.10/site-packages:$PYTHONPATH

# Copy application code
COPY . .

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]