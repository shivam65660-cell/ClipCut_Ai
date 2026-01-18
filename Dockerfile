# --------- STAGE 1: Base ----------
FROM python:3.10-slim AS base

WORKDIR /app

# Install only required system tools
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# --------- STAGE 2: Dependencies ----------
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# --------- STAGE 3: Final Image ----------
COPY . .

# Create uploads folder
RUN mkdir -p /app/uploads

EXPOSE 5000

# Start Flask
CMD ["python", "app.py"]
