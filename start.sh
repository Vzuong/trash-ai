#!/bin/bash

echo "========================================================"
echo " Starting Trash AI System (Docker Container)"
echo "========================================================"

export PORT=${PORT:-7860}
export YOLO_PORT=5001
export YOLO_SERVICE_URL="http://127.0.0.1:5001"

echo "Starting YOLO Python service on port 5001..."
python3 /app/server/yolo_service.py &
PYTHON_PID=$!

echo "Waiting for Python service to initialize..."
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:5001/health | grep -q "ready"; then
        echo "Python service is ready."
        break
    fi
    sleep 1
done

echo "Starting Node.js server on port $PORT..."
cd /app/server && node server.js &
NODE_PID=$!

echo "Trash AI service is online on port $PORT."

# Monitor processes
wait -n $PYTHON_PID $NODE_PID
exit $?
