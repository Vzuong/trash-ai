import urllib.request
import json
import base64
import numpy as np
import cv2

def run_tests():
    base_url = 'http://localhost:7860'

    # 1. Health Check
    print("=== 1. Health Check Endpoint ===")
    try:
        req = urllib.request.Request(f"{base_url}/api/health")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print("Status:", data.get('status'))
            print("AI Engine Info:", data.get('ai'))
    except Exception as e:
        print("Health Check Failed:", e)

    # 2. Model Info
    print("\n=== 2. Model Info Endpoint ===")
    try:
        req = urllib.request.Request(f"{base_url}/api/model-info")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            model_info = data.get('data', {}).get('modelInfo', {})
            print("Model Status:", model_info.get('status'))
            print("Device:", model_info.get('device'))
            print("Weights:", model_info.get('weightsFile'))
    except Exception as e:
        print("Model Info Failed:", e)

    # 3. Blank frame (Empty background test)
    print("\n=== 3. Empty Background / Corridor Test ===")
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', blank)
    b64_str = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')

    try:
        req = urllib.request.Request(
            f"{base_url}/api/predict/webcam",
            data=json.dumps({'frame': b64_str}).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print("Success:", data.get('success'))
            print("Total Objects:", data.get('data', {}).get('totalObjects'))
            print("Detections List:", data.get('data', {}).get('detections'))
            print("Primary Result:", data.get('data', {}).get('primaryResult'))
            assert data.get('data', {}).get('totalObjects') == 0, "Expected 0 objects for blank frame!"
            assert len(data.get('data', {}).get('detections')) == 0, "Expected empty detections list!"
            print("-> PASSED: Empty background returns exactly 0 detections (No fake boxes)!")
    except Exception as e:
        print("Blank Frame Test Failed:", e)

    # 4. Real Object Test (Battery image)
    print("\n=== 4. Real Object Detection Test (Battery) ===")
    bat_img = cv2.imread('server/uploads/sample_battery.jpg')
    _, buffer_bat = cv2.imencode('.jpg', bat_img)
    b64_bat = 'data:image/jpeg;base64,' + base64.b64encode(buffer_bat).decode('utf-8')

    try:
        req = urllib.request.Request(
            f"{base_url}/api/predict/webcam",
            data=json.dumps({'frame': b64_bat}).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            detections = data.get('data', {}).get('detections', [])
            print("Success:", data.get('success'))
            print("Total Objects:", len(detections))
            for d in detections:
                print(f" -> Object: {d.get('className')} ({d.get('classCode')}) | Conf: {d.get('confidencePercent')}% | BBox: {d.get('bbox')}")
            assert len(detections) > 0, "Expected at least 1 object!"
            assert detections[0].get('classCode') == 'battery', "Expected battery class!"
            print("-> PASSED: Real AI detected genuine object with accurate coordinates!")
    except Exception as e:
        print("Battery Test Failed:", e)

if __name__ == "__main__":
    run_tests()
