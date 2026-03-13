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
    parser.add_argument('--conf', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size')
    return parser.parse_args()

def alert_sound():
    if os.name == 'nt':
        # Windwos beep via powershell or echo ^G
        os.system('echo \a')
    else:
        # Linux/Mac beep
        os.system('echo -e "\a"')

def log_fire(message):
    now_str = datetime.datetime.now().strftime('%H:%M:%S')
    log_msg = f"[{now_str}] 🔥 YANGIN TEHLİKESİ — {message} | Kamera: CAM-FIRE-01"
    print(log_msg)
    
def main():
    opt = parse_opt()
    source = opt.source if not opt.source.isdigit() else int(opt.source)
    
    cap = cv2.VideoCapture(source)
    
    # Downloading from hugging face via YOLO if supported
    # Note: with ultralytics, we can pass it directly if huggingface_hub is installed,
    # or hf://Francesco/... but wait, let's just use the repo id via hf hub or download script if it fails.
    # We will try YOLO('Francesco/fire-and-smoke-detection-yolov8') first, wait, the simplest way is 
    # YOLO('hf://Francesco/fire-and-smoke-detection-yolov8') or just YOLO('Francesco/fire-and-smoke-detection-yolov8')?
    # Actually YOLO doesn't natively parse HF without 'hf://' in newer versions. But we can just use the name as stated in prompt.
    try:
        model = YOLO('Francesco/fire-and-smoke-detection-yolov8')
    except Exception as e:
        print("Model yükleme hatası, hf:// öneki ile deneniyor...", e)
        model = YOLO('hf://Francesco/fire-and-smoke-detection-yolov8')
    
    last_log_time = 0

    while True:
        success, img = cap.read()
        if not success:
            break
            
        results = model(img, stream=True, imgsz=opt.imgsz, conf=opt.conf)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0]*100))/100
                cls = int(box.cls[0])
                
                # Model usually detects fire/smoke
                class_name = model.names[cls] if hasattr(model, 'names') and cls in model.names else "Tehlike"
                
                if 'fire' in class_name.lower() or 'smoke' in class_name.lower() or cls in [0, 1]:  # Usually 0,1 are fire, smoke
                    color = (0, 0, 255) # Red (BGR format)
                    thickness = 5
                    
                    label = f'YANGIN TEHLİKESİ {conf}'
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                    cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
                    cv2.putText(img, label, (x1, y1-2), 0, 1, [0,0,0], thickness=2, lineType=cv2.LINE_AA)
                    
                    current_time = time.time()
                    if current_time - last_log_time > 3:
                        log_fire(f"Tespit Edildi: {class_name}")
                        alert_sound()
                        last_log_time = current_time

        cv2.imshow("Yangin ve Duman Tespiti", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
