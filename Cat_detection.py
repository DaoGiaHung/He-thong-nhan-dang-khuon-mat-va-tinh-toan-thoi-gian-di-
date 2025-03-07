from flask import Flask, render_template, Response, request
import cv2
from threading import Thread, Lock
import time
import datetime
import numpy as np

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# URL của luồng video và cấu hình các tham số
stream_url = 'http://192.168.165.126:8080/video'
cap = cv2.VideoCapture(stream_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Cấu hình kích thước bộ đệm để xử lý video mượt mà hơn

recording = False  # Cờ để theo dõi trạng thái ghi hình
out = None  # Biến để lưu trữ đối tượng VideoWriter khi ghi hình
last_detection_time = None  # Thời gian phát hiện mèo cuối cùng

fps = 25.0  # Số khung hình trên giây
width, height = 320, 240  # Kích thước khung hình

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec dùng để tạo tệp video
video_frame = None  # Biến để lưu trữ khung hình hiện tại
lock = Lock()  # Khóa để đảm bảo đồng bộ khi truy cập khung hình

# Tải mô hình MobileNet SSD đã được huấn luyện trước để phát hiện đối tượng
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Hàm để xử lý luồng video và phát hiện đối tượng
def camera_stream():
    global cap, recording, out, video_frame, last_detection_time
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)  # Chờ một chút trước khi thử lại nếu không đọc được khung hình
            continue

        # Thực hiện phát hiện đối tượng
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        cat_detected = False  # Cờ để theo dõi trạng thái phát hiện mèo
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Ngưỡng tin cậy
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] == "cat":  # Nếu phát hiện là mèo
                    cat_detected = True
                    last_detection_time = time.time()  # Cập nhật thời gian phát hiện mèo
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Bắt đầu ghi hình nếu phát hiện thấy mèo và đang không ghi hình
        if cat_detected and not recording:
            recording = True
            current_time = datetime.datetime.now()
            filename = current_time.strftime("%H-%M-%S_%d-%m-%Y") + ".mp4"
            print(f"Recording started: {filename}")
            out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        # Dừng ghi hình nếu không phát hiện thấy mèo trong 3 giây
        elif not cat_detected and recording:
            if last_detection_time and (time.time() - last_detection_time > 3):
                recording = False
                print("Recording stopped.")
                if out:
                    out.release()
                    out = None

        # Ghi khung hình vào tệp video nếu đang ghi hình
        if recording:
            out.write(frame)

        # Cập nhật khung hình hiện tại
        with lock:
            video_frame = frame.copy() if frame is not None else None

# Hàm để tạo khung hình gửi tới client
def gen_frames():
    global video_frame
    while True:
        with lock:
            if video_frame is None:
                time.sleep(0.1)
                continue
            ret, buffer = cv2.imencode('.jpg', video_frame)
            if not ret:
                continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Định tuyến Flask để truyền luồng video tới client
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Định tuyến Flask để phục vụ trang HTML chính
@app.route('/')
def index():
    return render_template('index_flask_server.html')

# Hàm để tắt máy chủ Flask
def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

# Định tuyến Flask để tắt máy chủ
@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

# Hàm chính để chạy ứng dụng Flask và xử lý luồng video
if __name__ == '__main__':
    camera_thread = Thread(target=camera_stream)
    camera_thread.start()
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        if cap.isOpened():
            cap.release()
        if out is not None:
            out.release()
