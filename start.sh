#!/bin/bash

echo "========================================================"
echo " 🌱 STARTING TRASH AI RECOGNITION SYSTEM IN DOCKER"
echo "========================================================"

export PORT=${PORT:-7860}
export YOLO_PORT=5001
export YOLO_SERVICE_URL="http://127.0.0.1:5001"

echo "[1/2] Starting YOLO Python AI Service on internal port 5001..."
python3 /app/server/yolo_service.py &
PYTHON_PID=$!

echo "Waiting for Python AI model to initialize..."
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:5001/health | grep -q "ready"; then
        echo "✅ Python AI Service is READY!"
        break
    fi
    sleep 1
done

echo "[2/2] Starting Node.js Backend & Web Server on port $PORT..."
cd /app/server && node server.js &
NODE_PID=$!

echo "🚀 Trash AI System is LIVE and serving traffic on port $PORT!"

# Monitor processes
wait -n $PYTHON_PID $NODE_PID
exit $?
