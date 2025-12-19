import cv2
import time
import threading
from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import yt_dlp
from collections import defaultdict
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

try:
    model = YOLO('yolov8n.pt')
    print("Tải model YOLO thành công!")
except Exception as e:
    print(f"Lỗi khi tải model YOLO: {e}")
    model = None

stream_locks = defaultdict(threading.Lock)
stream_urls = {}
frame_cache = defaultdict(dict)
client_counts = defaultdict(int)

def get_youtube_stream_url(video_url):
    if video_url in stream_urls:
        return stream_urls[video_url]
    try:
        ydl_opts = {
            'format': 'best[ext=mp4][height<=720]/best[ext=mp4]/best',
            'quiet': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            url = info['url']
            stream_urls[video_url] = url
            return url
    except Exception as e:
        print(f"Không thể lấy stream URL: {e}")
        return None

def video_processing_thread(youtube_url):
    with stream_locks[youtube_url]:
        if frame_cache[youtube_url].get('processing'):
            return

        frame_cache[youtube_url]['processing'] = True
        
        if os.path.exists(youtube_url):
             stream_url = youtube_url
        else:
             stream_url = get_youtube_stream_url(youtube_url)

        if not stream_url:
            frame_cache[youtube_url]['processing'] = False
            return

        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print(f"Lỗi: Không thể mở stream video từ URL.")
            frame_cache[youtube_url]['processing'] = False
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1 / fps if fps > 0 else 0.04

        while client_counts[youtube_url] > 0:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results[0].plot()

            class_indices = results[0].boxes.cls.cpu().numpy()
            current_frame_detections = defaultdict(int)
            for class_index in class_indices:
                class_name = model.names[int(class_index)]
                current_frame_detections[class_name] += 1
            
            y_offset = 30
            for class_name, count in current_frame_detections.items():
                text = f"{class_name}: {count}"
                cv2.putText(annotated_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                y_offset += 30

            _, encoded_processed = cv2.imencode(".jpg", annotated_frame)
            frame_cache[youtube_url]['processed'] = encoded_processed.tobytes()

            elapsed_time = time.time() - start_time
            sleep_time = delay - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()
        if youtube_url in frame_cache:
            del frame_cache[youtube_url]
        if youtube_url in stream_urls:
            del stream_urls[youtube_url]
        print(f"Đã dừng xử lý cho stream: {youtube_url}")

def generate_frames(youtube_url):
    client_counts[youtube_url] += 1
    print(f"Client mới kết nối tới {youtube_url}. Tổng số client: {client_counts[youtube_url]}")
    try:
        if not frame_cache[youtube_url].get('processing'):
            thread = threading.Thread(target=video_processing_thread, args=(youtube_url,)) 
            thread.daemon = True
            thread.start()

        while True:
            if 'processed' in frame_cache[youtube_url]:
                frame_bytes = frame_cache[youtube_url]['processed']
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
    finally:
        client_counts[youtube_url] -= 1
        print(f"Client đã ngắt kết nối khỏi {youtube_url}. Số client còn lại: {client_counts[youtube_url]}")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Check extension
    ext = filename.rsplit('.', 1)[1].lower()
    
    if ext in ['jpg', 'jpeg', 'png', 'bmp', 'webp']:
        # Image processing
        try:
            results = model(filepath)
            annotated_frame = results[0].plot()
            result_filename = f"processed_{filename}"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            cv2.imwrite(result_path, annotated_frame)
            return render_template('index.html', image_result=result_path)
        except Exception as e:
            return f"Error processing image: {e}", 500

    elif ext in ['mp4', 'avi', 'mov', 'mkv']:
        # Video processing (stream)
        return render_template('index.html', video_url=filepath)
    
    return "Unsupported file type", 400
@app.route('/')
def index():
    video_url = request.args.get('video_url')
    return render_template('index.html', video_url=video_url)

@app.route('/video_feed')
def video_feed():
    video_url = request.args.get('url')
    if not video_url:
        return "Missing URL parameter", 400
    return Response(generate_frames(video_url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5001)