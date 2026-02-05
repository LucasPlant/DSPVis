FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for the application
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application with Gunicorn
# CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--worker-class", "sync", "--timeout", "60", "app:server"]
