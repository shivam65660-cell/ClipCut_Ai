# Use Python base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Create uploads folder
RUN mkdir -p /app/uploads

# Expose Flask port
EXPOSE 5000

# Start Flask app
CMD ["python", "app.py"]
