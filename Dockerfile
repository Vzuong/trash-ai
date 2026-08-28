# ========================================================
# STAGE 1: Build Vue 3 Frontend Single Page Application
# ========================================================
FROM node:20-slim AS frontend-builder
WORKDIR /app/client

COPY client/package*.json ./
RUN npm install --include=dev

COPY client/ ./
RUN npm run build

# ========================================================
# STAGE 2: Python 3.11 + Node.js 20 Universal Runtime
# ========================================================
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (curl, libgl1 for OpenCV, Node.js 20)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    libgl1 \
    libglib2.0-0 \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU & Ultralytics dependencies (Lightweight & Fast for Cloud)
RUN pip install --no-cache-dir \
    torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    ultralytics==8.3.40 \
    flask==3.0.3 \
    flask-cors==4.0.1 \
    pillow \
    opencv-python-headless \
    numpy

# Copy Server Dependencies & Install Node packages
WORKDIR /app/server
COPY server/package*.json ./
RUN npm install --production

# Copy Application Source Code
WORKDIR /app
COPY best.pt /app/best.pt
COPY server /app/server
COPY --from=frontend-builder /app/client/dist /app/client/dist
COPY start.sh /app/start.sh

# Fix permissions
RUN chmod +x /app/start.sh

# Environment Variables
ENV NODE_ENV=production
ENV PORT=7860
ENV YOLO_SERVICE_URL=http://127.0.0.1:5001

# Expose default port (Hugging Face Spaces: 7860, Render: $PORT)
EXPOSE 7860

# Startup command
CMD ["/app/start.sh"]
