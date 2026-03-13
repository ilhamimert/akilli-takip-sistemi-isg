import argparse
import cv2
import math
from ultralytics import YOLO
import datetime
import os
import time

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='video source: 0 for webcam, RTSP URL, etc.')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size')
    return parser.parse_args()

def init_logs_dir():
    if not os.path.exists('logs'):
        os.makedirs('logs')

def log_violation(message):
    now_str = datetime.datetime.now().strftime('%H:%M:%S')
    log_msg = f"[{now_str}] ⚠️  KKD İHLALİ — {message} | Kamera: CAM-01"
    print(log_msg)
    with open('logs/violations.txt', 'a', encoding='utf-8') as f:
        f.write(log_msg + '\n')
    
    # Save alerts to a JSON file for the flask web app
    import json
    alerts_file = 'logs/alerts.json'
    alerts = []
    if os.path.exists(alerts_file):
        try:
            with open(alerts_file, 'r', encoding='utf-8') as f:
                alerts = json.load(f)
        except:
            pass
    alerts.insert(0, {"time": now_str, "message": message, "camera": "CAM-01"})
    alerts = alerts[:10]  # keep last 10
    with open(alerts_file, 'w', encoding='utf-8') as f:
        json.dump(alerts, f, ensure_ascii=False)

def main():
    opt = parse_opt()
    source = opt.source if not opt.source.isdigit() else int(opt.source)
    
    init_logs_dir()
    
    # Init counters globally for flask app access
    stats_file = 'logs/stats.json'
    def update_stats(compliant, violations):
        import json
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({"compliant": compliant, "violations": violations}, f)

    cap = cv2.VideoCapture(source)
    model = YOLO("models/best.pt")
    
    # English original: 'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle' 
    # (Assuming the model uses the same classes as YOLO_Video.py)
    # The user prompt: "Hardhat" -> "Baret Var", "NO-Hardhat" -> "KKD EKSİK - BARET", "Safety Vest" -> "Yelek Var", "NO-Safety Vest" -> "KKD EKSİK - YELEK", "Person" -> "Kişi"
    
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
    
    # We might need to map from actual model names
    original_classes = model.names
    
    last_log_times = {}

    while True:
        success, img = cap.read()
        if not success:
            break
            
        results = model(img, stream=True, imgsz=opt.imgsz, conf=opt.conf)
        
        compliant_count = 0
        violation_count = 0
        
        for r in results:
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
                
                color = (255, 255, 255) # Default White (Person)
                thickness = 2
                
                # Check for Compliant
                if class_name_en in ['Hardhat', 'Safety Vest', 'Mask']:
                    color = (0, 255, 0) # Green
                    compliant_count += 1
                # Check for Violation
                elif class_name_en in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
                    color = (0, 0, 255) # Red (BGR format)
                    thickness = 5 # Thick border
                    violation_count += 1
                    
                    # Log violation max once every 3 seconds per class
                    current_time = time.time()
                    if class_name_en not in last_log_times or current_time - last_log_times[class_name_en] > 3:
                        log_violation(class_name_full)
                        last_log_times[class_name_en] = current_time
                        
                elif class_name_en == 'Person':
                    color = (255, 255, 255) # White
                else:
                    color = (255, 255, 0) # Cyan for other objects
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
                cv2.putText(img, label, (x1, y1-2), 0, 1, [0,0,0], thickness=2, lineType=cv2.LINE_AA)

        update_stats(compliant_count, violation_count)
        cv2.imshow("PPE Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
