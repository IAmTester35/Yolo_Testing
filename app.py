import cv2
import time
import threading
from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import yt_dlp
from collections import defaultdict

app = Flask(__name__)

try:
    model = YOLO('yolo11n.pt')
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