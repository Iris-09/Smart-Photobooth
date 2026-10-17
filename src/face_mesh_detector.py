import cv2
import mediapipe as mp
import math

# 1. Khởi tạo MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
# refine_landmarks=True giúp lấy chính xác hơn vùng mắt (478 điểm thay vì 468)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2, 
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def get_landmark_pixel(landmarks, index, img_w, img_h):
    """Chuyển đổi tọa độ chuẩn hóa (0.0 -> 1.0) sang tọa độ Pixel thực tế"""
    lm = landmarks.landmark[index]
    return int(lm.x * img_w), int(lm.y * img_h)

def calculate_face_metrics(landmarks, img_w, img_h):
    """
    Tính toán các thông số hình học của khuôn mặt để chuẩn bị ghép Filter AR:
    - Tâm đặt kính (giữa 2 chân mày - điểm 168)
    - Đỉnh trán đặt mũ (điểm 10)
    - Khoảng cách 2 đuôi mắt (điểm 33 và 263) -> Dùng để scale kích thước kính
    - Góc nghiêng của đầu (Roll angle) -> Dùng để xoay ảnh PNG
    """
    # Lấy tọa độ pixel các điểm quan trọng
    left_eye_outer = get_landmark_pixel(landmarks, 33, img_w, img_h)
    right_eye_outer = get_landmark_pixel(landmarks, 263, img_w, img_h)
    nose_bridge = get_landmark_pixel(landmarks, 168, img_w, img_h)
    forehead = get_landmark_pixel(landmarks, 10, img_w, img_h)
    chin = get_landmark_pixel(landmarks, 152, img_w, img_h)

    # 1. Tính khoảng cách Euclid giữa 2 đuôi mắt (độ rộng cơ sở của kính)
    dx = right_eye_outer[0] - left_eye_outer[0]
    dy = right_eye_outer[1] - left_eye_outer[1]
    eye_distance = math.hypot(dx, dy) # Tương đương sqrt(dx^2 + dy^2)

    # 2. Tính góc nghiêng của đầu (đơn vị: Độ)
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    return {
        "left_eye": left_eye_outer,
        "right_eye": right_eye_outer,
        "nose_bridge": nose_bridge,
        "forehead": forehead,
        "chin": chin,
        "eye_distance": eye_distance,
        "angle": angle_deg
    }

# --- CHẠY THỬ NGHIỆM TRÊN WEBCAM ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Chuyển sang RGB cho MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = face_mesh.process(image_rgb)
        image_rgb.flags.writeable = True

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                metrics = calculate_face_metrics(face_landmarks, w, h)

                # Vẽ các điểm neo chính để kiểm tra trực quan
                cv2.circle(frame, metrics["left_eye"], 5, (0, 255, 0), -1)     # Đuôi mắt trái (Xanh lá)
                cv2.circle(frame, metrics["right_eye"], 5, (0, 255, 0), -1)    # Đuôi mắt phải (Xanh lá)
                cv2.circle(frame, metrics["nose_bridge"], 6, (0, 0, 255), -1)  # Tâm kính (Đỏ)
                cv2.circle(frame, metrics["forehead"], 6, (255, 0, 0), -1)     # Đỉnh trán đặt mũ (Xanh dương)

                # Vẽ đường nối 2 mắt thể hiện trục nghiêng của kính
                cv2.line(frame, metrics["left_eye"], metrics["right_eye"], (0, 255, 255), 2)

                # Hiển thị thông số góc nghiêng và khoảng cách lên màn hình
                info_text = f"Scale: {int(metrics['eye_distance'])}px | Angle: {metrics['angle']:.1f} deg"
                cv2.putText(frame, info_text, (metrics["forehead"][0] - 100, metrics["forehead"][1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Smart Photobooth - Face Mesh Test (Week 4)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()