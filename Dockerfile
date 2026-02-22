# Build stage - use full Python image for compilation
FROM python:3.10 AS builder

WORKDIR /app

COPY requirements.txt .

# Install dependencies to /root/.local (user site-packages)
RUN pip install --user --no-cache-dir -r requirements.txt


# Final stage - use slim image for smaller size
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Set PATH to include user site-packages
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/root/.local/lib/python3.10/site-packages:$PYTHONPATH

# Copy application code
COPY . .

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]