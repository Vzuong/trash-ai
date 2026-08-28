#!/bin/bash

echo "========================================================"
echo " 🌱 STARTING TRASH AI RECOGNITION SYSTEM IN DOCKER"
echo "========================================================"

export PORT=${PORT:-7860}
export YOLO_SERVICE_URL="http://127.0.0.1:5001"

echo "[1/2] Starting YOLO11s Python AI Service on port 5001..."
cd /app/server && python3 yolo_service.py &
PYTHON_PID=$!

echo "[2/2] Starting Node.js Backend & Web Server on port $PORT..."
cd /app/server && node server.js &
NODE_PID=$!

echo "🚀 System is LIVE and ready to serve at port $PORT!"

# Wait for both processes
wait -n $PYTHON_PID $NODE_PID

# Exit with status of process that exited first
exit $?
