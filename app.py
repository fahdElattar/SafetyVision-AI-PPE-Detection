import os
import cv2
from flask import Flask, render_template, Response, jsonify, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# Paths
UPLOAD_FOLDER = 'static/uploads'
DEMO_VIDEO = 'static/demo/test_video.mp4'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/demo', exist_ok=True)

# Load Model
model = YOLO("model/yolo8m.pt")

# Global State
camera_active = False
video_path = None
cap = None

def cleanup_file(path):
    """Deletes the file if it exists in the uploads folder to save space."""
    if path and UPLOAD_FOLDER in path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            print(f"Cleanup error: {e}")

def gen_frames(source):
    global cap, camera_active, video_path
    
    if cap is not None:
        cap.release()
    
    cap = cv2.VideoCapture(source)

    while True:
        # Stop signals
        if source == 0 and not camera_active: break
        if isinstance(source, str) and video_path is None: break

        success, frame = cap.read()
        if not success: break 
        
        results = model.predict(source=frame, conf=0.25, stream=True)
        for result in results:
            annotated_frame = result.plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret: continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    if cap:
        cap.release()
        cap = None
    
    # Auto-delete user uploads after use
    if isinstance(source, str) and source != DEMO_VIDEO:
        cleanup_file(source)

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

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global video_path, camera_active
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