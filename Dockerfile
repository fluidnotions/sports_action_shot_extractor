# Action Shot Extractor Docker Image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir streamlit

# Install the package in development mode
COPY . .
RUN pip install -e .

# Create directories for uploads and outputs
RUN mkdir -p /app/uploads /app/outputs /app/reference_frames

# Expose Streamlit port
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Create a startup script
RUN echo '#!/bin/bash\n\
echo "🚀 Action Shot Extractor is starting..."\n\
echo "📱 Web UI will be available at: http://localhost:8501"\n\
echo "🐳 If running in Docker, use: http://YOUR_IP:8501"\n\
echo ""\n\
echo "Starting Streamlit server..."\n\
streamlit run /app/streamlit_app.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Use the startup script
CMD ["/app/start.sh"]