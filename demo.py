from flask import Flask, render_template, Response, jsonify
import cv2
from threading import Thread, Lock
import time
import face_recognition
import numpy as np
import datetime
import pygame
import requests  # Thêm thư viện requests để gửi HTTP request

app = Flask(__name__)

CAM_A_IP = 'http://192.168.1.101:4747/video'
CAM_B_IP = 'http://192.168.1.102:4747/video'
CAM_C_IP = 'http://192.168.1.110:4747/video'  # Thêm camera C

width, height = 160, 120  # Giảm kích thước khung hình để tăng tốc

lock = Lock()
frame_a, frame_b, frame_c = None, None, None

time_a_start = None
time_b_detected = None
time_c_detected = None
face_detected_a = False
cycle_completed = False

max_travel_time = 5
min_travel_time = 2
min_valid_time = 0.5

known_face_encoding = None
last_processed_time = datetime.datetime.now()
MOCKAPI_URL = "https://6788b3fc2c874e66b7d5f86e.mockapi.io/runner"

# Khởi tạo pygame mixer
pygame.mixer.init()
alert_sound = "static/sounds/alert.mp3"
ting_ting_sound = "static/sounds/ting.mp3"

def play_alert():
    pygame.mixer.music.load(alert_sound)
    pygame.mixer.music.play()

def play_ting_ting():
    pygame.mixer.music.load(ting_ting_sound)
    pygame.mixer.music.play()

# Global variable to track completion state and timestamp
completion_state = {
    "message": "",
    "timestamp": None
}

class CameraStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.lock = Lock()
        self.frame = None
        self.running = True
        self.motion_detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.previous_frame = None
        self.motion_detected = False
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = cv2.resize(frame, (width, height))
                    
    def detect_motion(self, frame):
        # Apply background subtraction
        fg_mask = self.motion_detector.apply(frame)
        
        # Threshold to get binary image
        _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Calculate the percentage of moving pixels
        motion_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.size
        motion_percentage = (motion_pixels / total_pixels) * 100
        
        # Return True if motion percentage is above threshold
        return motion_percentage > 1.5  # Adjust threshold as needed

    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                frame_copy = self.frame.copy()
                # Update motion detection
                self.motion_detected = self.detect_motion(frame_copy)
                return frame_copy
            return None

    def is_motion_detected(self):
        return self.motion_detected

    def stop(self):
        self.running = False
        self.cap.release()

camera_a = CameraStream(CAM_A_IP)
camera_b = CameraStream(CAM_B_IP)
camera_c = CameraStream(CAM_C_IP)  # Thêm camera C

def detect_and_recognize_faces(frame, interval=0.3):  # Giảm interval xuống 0.3 giây để tăng tần suất nhận diện
    global known_face_encoding, last_processed_time, face_detected_a, cycle_completed
    now = datetime.datetime.now()
    if (now - last_processed_time).total_seconds() < interval:
        # Trả về giá trị trước đó nếu chưa đủ interval
        return face_detected_a, []
    last_processed_time = now

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")  # Sử dụng model HOG cho tốc độ nhanh hơn
    if len(face_locations) > 0:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1)  # Giảm num_jitters để tăng tốc

        if known_face_encoding is None and len(face_encodings) > 0:
            known_face_encoding = face_encodings[0]

        if known_face_encoding is not None and len(face_encodings) > 0:
            # Giảm tolerance để tăng độ chính xác
            matches = face_recognition.compare_faces([known_face_encoding], face_encodings[0], tolerance=0.5)
            if True in matches:
                if not face_detected_a:  # Chỉ cập nhật khi trạng thái thay đổi từ False sang True
                    face_detected_a = True
                return True, face_locations

    if face_detected_a:  # Chỉ cập nhật khi trạng thái thay đổi từ True sang False
        face_detected_a = False
        global time_a_start
        time_a_start = None  # Reset timer when face is lost
        cycle_completed = False  # Reset cycle completion when face is lost
    return False, []

def save_to_mockapi(start_time, end_time, travel_time_a_b, travel_time_b_c, travel_time_c_a):
    """
    Gửi dữ liệu lên MockAPI thông qua HTTP POST request.
    """
    data = {
        "start_time": start_time,
        "end_time": end_time,
        "travel_time_a_b": travel_time_a_b,
        "travel_time_b_c": travel_time_b_c,
        "travel_time_c_a": travel_time_c_a
    }
    try:
        response = requests.post(MOCKAPI_URL, json=data)
        if response.status_code == 201:  # 201 là mã trạng thái thành công khi tạo mới
            print("Dữ liệu đã được lưu lên MockAPI thành công!")
        else:
            print(f"Lỗi khi lưu dữ liệu: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Lỗi kết nối đến MockAPI: {e}")

def process_frame(camera, camera_name):
    global frame_a, frame_b, frame_c, time_a_start, time_b_detected, time_c_detected, timer_enabled, waiting_for_start
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        current_time = time.time()
        if camera_name == "A":
            # Camera A uses face recognition
            recognized, face_locations = detect_and_recognize_faces(frame)
            if recognized:
                with lock:
                    if time_c_detected is not None:
                        # Tính toán thời gian di chuyển từ C về A
                        travel_time_c_a = current_time - time_c_detected
                        if travel_time_c_a >= min_valid_time:
                            # Tính toán các thời gian khác
                            travel_time_a_b = time_b_detected - time_a_start
                            travel_time_b_c = time_c_detected - time_b_detected
                            
                            # Lưu dữ liệu
                            save_to_mockapi(
                                datetime.datetime.fromtimestamp(time_a_start).isoformat(),
                                datetime.datetime.fromtimestamp(current_time).isoformat(),
                                travel_time_a_b,
                                travel_time_b_c,
                                travel_time_c_a
                            )
                            
                            # In thông tin
                            print(f"Thời gian di chuyển A → B: {travel_time_a_b:.2f}s")
                            print(f"Thời gian di chuyển B → C: {travel_time_b_c:.2f}s")
                            print(f"Thời gian di chuyển C → A: {travel_time_c_a:.2f}s")
                            
                            # Cập nhật trạng thái hoàn thành
                            completion_state["message"] = "🎉 Hoàn thành chu trình!"
                            completion_state["timestamp"] = current_time
                            
                            # Đánh dấu hoàn thành chu trình và reset các biến
                            cycle_completed = True
                            time_a_start = None
                            time_b_detected = None
                            time_c_detected = None
                    elif time_a_start is None:
                        time_a_start = current_time
                        print(f"Bắt đầu từ A: {datetime.datetime.fromtimestamp(time_a_start).isoformat()}")
                        print(f"Bắt đầu từ A: {datetime.datetime.fromtimestamp(time_a_start).isoformat()}")
                        
                # Vẽ bounding box cho khuôn mặt
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, "Face", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else:
            # Cameras B and C use motion detection
            motion_detected = camera.is_motion_detected()
            if motion_detected:
                if camera_name == "B":
                    with lock:
                        if time_a_start is not None and time_b_detected is None:
                            if current_time - time_a_start >= min_valid_time:
                                time_b_detected = current_time
                                travel_time = time_b_detected - time_a_start
                                print(f"Thời gian di chuyển A → B: {travel_time:.2f}s")
                                if travel_time <= min_travel_time:
                                    play_ting_ting()
                                elif travel_time > max_travel_time:
                                    play_alert()
                                # Vẽ thông báo chuyển động
                                cv2.putText(frame, "Motion Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                elif camera_name == "C":
                    with lock:
                        if time_b_detected is not None and time_c_detected is None:
                            if current_time - time_b_detected >= min_valid_time:
                                time_c_detected = current_time
                                travel_time = time_c_detected - time_b_detected
                                print(f"Thời gian di chuyển B → C: {travel_time:.2f}s")
                                if travel_time <= min_travel_time:
                                    play_ting_ting()
                                elif travel_time > max_travel_time:
                                    play_alert()
                                # Vẽ thông báo chuyển động
                                cv2.putText(frame, "Motion Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        with lock:
            if camera_name == "A":
                frame_a = frame.copy()
            elif camera_name == "B":
                frame_b = frame.copy()
            elif camera_name == "C":
                frame_c = frame.copy()

def gen_frames(camera_name):
    global frame_a, frame_b, frame_c
    
    while True:
        with lock:
            frame = frame_a if camera_name == "A" else frame_b if camera_name == "B" else frame_c
            if frame is None:
                time.sleep(0.1)
                continue

            # Chuyển đổi frame sang định dạng BGR để hiển thị
            display_frame = frame.copy()

            if camera_name == "A":
                # Camera A uses face detection
                recognized, face_locations = detect_and_recognize_faces(display_frame)
                if recognized:
                    for (top, right, bottom, left) in face_locations:
                        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(display_frame, "Face Detected", (left, top - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Display motion detection status for cameras B and C
                camera = camera_b if camera_name == "B" else camera_c
                if camera.is_motion_detected():
                    cv2.putText(display_frame, "Motion Detected", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Add detection mode label
            label = "Face Detection" if camera_name == "A" else "Motion Detection"
            cv2.putText(display_frame, label, (10, display_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Encode frame thành JPEG
            ret, buffer = cv2.imencode('.jpg', display_frame)
            if not ret:
                continue

        # Trả về dữ liệu hình ảnh dưới dạng stream
        yield (b'--frame\r\n' 
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/current_time')
def current_time():
    global time_a_start, face_detected_a, cycle_completed
    # Chỉ bắt đầu timer khi phát hiện khuôn mặt và chưa hoàn thành chu trình
    if face_detected_a and time_a_start is None and not cycle_completed:
        time_a_start = time.time()
    
    if time_a_start is not None:
        elapsed_time = time.time() - time_a_start
        return jsonify({"elapsed_time": elapsed_time})
    return jsonify({"elapsed_time": None})

@app.route('/face_status')
def face_status():
    global face_detected_a, time_a_start
    current_time = time.time()
    
    status = {
        "detected": face_detected_a,
        "message": "✅ Đã xác nhận - Đang đếm thời gian" if face_detected_a else "⏳ Chờ xác nhận",
        "time_started": time_a_start is not None,
        "elapsed": current_time - time_a_start if time_a_start is not None else None
    }
    return jsonify(status)

@app.route('/video_feed_a')
def video_feed_a():
    return Response(gen_frames("A"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_b')
def video_feed_b():
    return Response(gen_frames("B"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_c')
def video_feed_c():
    return Response(gen_frames("C"), mimetype='multipart/x-mixed-replace; boundary=frame')

def calculate_travel_time():
    global time_a_start, time_b_detected, time_c_detected
    if time_a_start and time_b_detected and time_c_detected:
        travel_time_a_b = time_b_detected - time_a_start
        travel_time_b_c = time_c_detected - time_b_detected
        travel_time_c_a = time.time() - time_c_detected
        return f"✅ A → B: {travel_time_a_b:.2f}s, B → C: {travel_time_b_c:.2f}s, C → A: {travel_time_c_a:.2f}s"
    return "⏳ Đang thu thập dữ liệu..."

@app.route('/completion_status')
def get_completion_status():
    global completion_state
    if completion_state["timestamp"] is not None:
        if time.time() - completion_state["timestamp"] > 5:
            completion_state = {"message": "", "timestamp": None}
    return jsonify(completion_state)

@app.route('/travel_time')
def travel_time():
    return jsonify({"message": calculate_travel_time()})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    thread_a = Thread(target=process_frame, args=(camera_a, "A"), daemon=True)
    thread_b = Thread(target=process_frame, args=(camera_b, "B"), daemon=True)
    thread_c = Thread(target=process_frame, args=(camera_c, "C"), daemon=True)
    thread_a.start()
    thread_b.start()
    thread_c.start()
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        camera_a.stop()
        camera_b.stop()
        camera_c.stop()