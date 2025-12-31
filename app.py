import os
import cv2
import time
import glob
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
DEMO_VIDEO = 'static/demo/test_video.mp4'
REQUIRED_CLASSES = ['Helmet', 'Gloves', 'Vest']

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/demo', exist_ok=True)

# Load Model (YOLOv8 Medium)
model = YOLO("model/yolo8m.pt")

# --- Global State ---
camera_active = False
video_path = None
cap = None

def cleanup_uploads():
    """Removes all temporary files from the upload folder."""
    files = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
    for f in files:
        if ".gitkeep" not in f:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Cleanup error: {e}")

def cleanup_file(path):
    """Deletes a specific file if it exists."""
    if path and UPLOAD_FOLDER in path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            print(f"File deletion error: {e}")

def gen_frames(source):
    global cap, camera_active, video_path
    
    if cap is not None:
        cap.release()
    
    cap = cv2.VideoCapture(source)
    
    # PERFORMANCE: Scale down capture resolution to improve FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # PERFORMANCE: Reduce buffer size to prevent lag in live stream
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        # Stop signals
        if source == 0 and not camera_active: break
        if isinstance(source, str) and video_path is None: break

        success, frame = cap.read()
        if not success: break 
        
        # PERFORMANCE: Run inference at imgsz=320 for speed (standard is 640)
        # stream=True is more memory efficient
        results = model.predict(source=frame, conf=0.25, imgsz=320, stream=True, verbose=False)
        
        for result in results:
            detected = [result.names[int(cls)] for cls in result.boxes.cls.cpu().numpy()]
            annotated_frame = result.plot()
            
            # Real-time Missing PPE Detection
            missing = [item for item in REQUIRED_CLASSES if item not in detected]
            
            # Overlay Logic
            y0, dy = 30, 30
            cv2.putText(annotated_frame, "PPE AUDIT: ACTIVE", (10, y0), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            for i, item in enumerate(missing):
                cv2.putText(annotated_frame, f"MISSING: {item}", (10, y0 + (i+1)*dy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret: continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    if cap:
        cap.release()
        cap = None
    
    # Auto-delete video files after the stream ends
    if isinstance(source, str) and source != DEMO_VIDEO:
        cleanup_file(source)

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<source_type>')
def video_feed(source_type):
    global video_path, camera_active
    if source_type == 'webcam' and camera_active:
        return Response(gen_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif source_type == 'file' and video_path:
        return Response(gen_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    return ""

@app.route('/upload_image', methods=['POST'])
def upload_image():
    # Clean up previous results before new analysis
    cleanup_uploads()
    
    file = request.files.get('image')
    if file:
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        
        img = cv2.imread(path)
        # Process image
        results = model.predict(source=img, conf=0.25, imgsz=320)
        
        detected = [results[0].names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]
        missing = [item for item in REQUIRED_CLASSES if item not in detected]
        
        res_img = results[0].plot()
        
        # Add result overlays
        y0, dy = 40, 40
        cv2.putText(res_img, "IMAGE AUDIT RESULT", (20, y0), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        for i, item in enumerate(missing):
            cv2.putText(res_img, f"MISSING: {item}", (20, y0 + (i+1)*dy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        result_filename = "res_" + filename
        result_path = os.path.join(UPLOAD_FOLDER, result_filename)
        cv2.imwrite(result_path, res_img)
        
        # Delete original uploaded image, keep only the result
        cleanup_file(path)
        
        return jsonify({
            "success": True, 
            "result_url": f"/static/uploads/{result_filename}?t={int(time.time())}"
        })
    return jsonify({"success": False})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global video_path, camera_active
    # Clean up previous uploads to save space
    cleanup_uploads()
    
    file = request.files.get('video')
    if file:
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        video_path = path
        camera_active = False
        return jsonify({"success": True})
    return jsonify({"error": "No file"}), 400

@app.route('/start_demo', methods=['POST'])
def start_demo():
    global video_path, camera_active
    video_path = DEMO_VIDEO
    camera_active = False
    return jsonify({"success": True})

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_active, video_path, cap
    camera_active = not camera_active
    if not camera_active and cap:
        cap.release()
        cap = None
    video_path = None
    return jsonify({"status": camera_active})

@app.route('/stop_video', methods=['POST'])
def stop_video():
    global video_path, cap
    current_path = video_path
    video_path = None
    if cap:
        cap.release()
        cap = None
    if current_path != DEMO_VIDEO:
        cleanup_file(current_path)
    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)