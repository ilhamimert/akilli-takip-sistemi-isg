from flask import Flask, render_template, Response, jsonify
import cv2
import math
from ultralytics import YOLO
import datetime
import os
import json
import time

app = Flask(__name__, template_folder='.')

model = YOLO("ppe.pt")
camera = None

def init_logs_dir():
    if not os.path.exists('../logs'):
        os.makedirs('../logs')

def log_violation(message):
    now_str = datetime.datetime.now().strftime('%H:%M:%S')
    log_msg = f"[{now_str}] ⚠️  KKD İHLALİ — {message} | Kamera: CAM-01"
    print(log_msg)
    
    init_logs_dir()
    with open('../logs/violations.txt', 'a', encoding='utf-8') as f:
        f.write(log_msg + '\n')
    
    alerts_file = '../logs/alerts.json'
    alerts = []
    if os.path.exists(alerts_file):
        try:
            with open(alerts_file, 'r', encoding='utf-8') as f:
                alerts = json.load(f)
        except:
            pass
    alerts.insert(0, {"time": now_str, "message": message, "camera": "CAM-01"})
    alerts = alerts[:10]
    with open(alerts_file, 'w', encoding='utf-8') as f:
        json.dump(alerts, f, ensure_ascii=False)

def update_stats(compliant, violations):
    init_logs_dir()
    stats_file = '../logs/stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({"compliant": compliant, "violations": violations}, f)

def get_stats():
    stats_file = '../logs/stats.json'
    if os.path.exists(stats_file):
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {"compliant": 0, "violations": 0}

def generate_frames():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    
    original_classes = model.names
    class_map = {
        'Hardhat': {'display': 'Baret Var', 'full': 'Baret Var'},
        'NO-Hardhat': {'display': 'KKD EKSIK - BARET', 'full': 'KKD İHLALİ - BARET'},
        'Safety Vest': {'display': 'Yelek Var', 'full': 'Yelek Var'},
        'NO-Safety Vest': {'display': 'KKD EKSIK - YELEK', 'full': 'KKD İHLALİ - YELEK'},
        'Person': {'display': 'Kisi', 'full': 'Kişi'},
        'Mask': {'display': 'Maske Var', 'full': 'Maske Var'},
        'NO-Mask': {'display': 'KKD EKSIK - MASKE', 'full': 'KKD İHLALİ - MASKE'},
        'Safety Cone': {'display': 'Guvenlik Konisi', 'full': 'Güvenlik Konisi'},
        'machinery': {'display': 'Makine', 'full': 'Makine'},
        'vehicle': {'display': 'Arac', 'full': 'Araç'}
    }

    last_log_times = {}
    frame_count = 0
    last_results = None
    compliant_count = 0
    violation_count = 0

    while True:
        success, img = camera.read()
        if not success:
            break
        
        # Only perform detection every 3 frames to reduce CPU load
        if frame_count % 3 == 0:
            results = model(img, stream=True, conf=0.5, imgsz=320, verbose=False)
            last_results = list(results) # Materialize generator for persistence
            compliant_count = 0
            violation_count = 0
            
            # Update counts based on new detection
            for r in last_results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    class_name_en = original_classes[cls]
                    if class_name_en in ['Hardhat', 'Safety Vest', 'Mask']:
                        compliant_count += 1
                    elif class_name_en in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
                        violation_count += 1
            
            update_stats(compliant_count, violation_count)

        frame_count += 1
        
        # Draw boxes (either from new or previous results)
        if last_results:
            for r in last_results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = math.ceil((box.conf[0]*100))/100
                    cls = int(box.cls[0])
                    class_name_en = original_classes[cls]
                    class_data = class_map.get(class_name_en, {'display': class_name_en, 'full': class_name_en})
                    class_name_display = class_data['display']
                    class_name_full = class_data['full']
                    
                    label = f'{class_name_display} {conf}'
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    
                    color = (255, 255, 255)
                    thickness = 2
                    
                    if class_name_en in ['Hardhat', 'Safety Vest', 'Mask']:
                        color = (0, 255, 0)
                    elif class_name_en in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
                        color = (0, 0, 255)
                        thickness = 5
                        
                        current_time = time.time()
                        if class_name_en not in last_log_times or current_time - last_log_times[class_name_en] > 3:
                            log_violation(class_name_full)
                            last_log_times[class_name_en] = current_time
                            
                    elif class_name_en == 'Person':
                        color = (255, 255, 255)
                    else:
                        color = (255, 255, 0)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                    cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
                    cv2.putText(img, label, (x1, y1-2), 0, 1, [0,0,0], thickness=2, lineType=cv2.LINE_AA)
        
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alerts')
def alerts():
    init_logs_dir()
    alerts_file = '../logs/alerts.json'
    items = []
    if os.path.exists(alerts_file):
        try:
            with open(alerts_file, 'r', encoding='utf-8') as f:
                items = json.load(f)
        except:
            pass
    stats = get_stats()
    return jsonify({"alerts": items, "stats": stats})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
