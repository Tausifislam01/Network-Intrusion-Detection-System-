FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models artifacts/screenshots artifacts/logs mlruns

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

# Default command - starts both services
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port 8000 & streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"]