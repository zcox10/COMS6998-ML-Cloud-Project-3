FROM python:3.11.2-slim

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY src/ ./src/

ENV PYTHONPATH=/app
