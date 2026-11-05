import cv2
import mediapipe as mp
import time
import os
import math

# Nhập module AR Overlay của bạn
from src.face_ar_overlay import FaceAROverlay

# 1. CẤU HÌNH THƯ MỤC VÀ MEDIAPIPE
SAVE_DIR = "captured_photos"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# 2. CÁC HÀM HỖ TRỢ
def is_v_sign(hand_landmarks):
    lm = hand_landmarks.landmark
    # Ngón trỏ và giữa mở, áp út và út đóng
    return (lm[8].y < lm[6].y) and (lm[12].y < lm[10].y) and \
           (lm[16].y > lm[14].y) and (lm[20].y > lm[18].y)

def get_landmark_pixel(landmarks, index, img_w, img_h):
    lm = landmarks.landmark[index]
    return int(lm.x * img_w), int(lm.y * img_h)

def calculate_face_metrics(landmarks, img_w, img_h):
    left_eye = get_landmark_pixel(landmarks, 33, img_w, img_h)
    right_eye = get_landmark_pixel(landmarks, 263, img_w, img_h)
    forehead = get_landmark_pixel(landmarks, 10, img_w, img_h)
    upper_lip = get_landmark_pixel(landmarks, 164, img_w, img_h)

    dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
    return {
        "left_eye": left_eye, "right_eye": right_eye, 
        "forehead": forehead, "upper_lip": upper_lip, 
        "eye_dist": math.hypot(dx, dy), 
        "angle": math.degrees(math.atan2(dy, dx))
    }

# 3. VÒNG LẶP CHÍNH
def main():
    ar_overlay = FaceAROverlay()
    cap = cv2.VideoCapture(0)
    
    # Biến trạng thái UI
    slot_eyes, slot_head, slot_mouth = True, False, False
    color_mode = 0
    
    # Biến đếm ngược
    is_counting_down = False
    countdown_start = 0
    last_saved_time = 0
    saved_filename = ""

    print("Hệ thống đã sẵn sàng! Chạy Smart Photobooth...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # --- BƯỚC 1: XỬ LÝ MÀU ---
        if color_mode == 1:
            b, g, r = cv2.split(frame)
            b = cv2.convertScaleAbs(b, alpha=0.9, beta=0)
            r = cv2.convertScaleAbs(r, alpha=1.1, beta=15)
            frame = cv2.merge((b, g, r))
        elif color_mode == 2:
            frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

        # --- BƯỚC 2: XỬ LÝ AI (FACE & HANDS) ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        face_results = face_mesh.process(rgb_frame)
        hand_results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # Áp dụng AR Filter lên khuôn mặt
        if face_results.multi_face_landmarks:
            for face_lms in face_results.multi_face_landmarks:
                metrics = calculate_face_metrics(face_lms, w, h)
                if slot_head: frame = ar_overlay.apply_hat(frame, metrics["forehead"], metrics["eye_dist"], metrics["angle"])
                if slot_eyes: frame = ar_overlay.apply_glasses(frame, metrics["left_eye"], metrics["right_eye"], metrics["eye_dist"], metrics["angle"])
                if slot_mouth: frame = ar_overlay.apply_mustache(frame, metrics["upper_lip"], metrics["eye_dist"], metrics["angle"])

        # CHỐT ẢNH LƯU: Copy frame tại thời điểm này (Đã có filter, chưa có UI đếm ngược)
        photo_frame = frame.copy()

        # Kiểm tra cử chỉ tay chữ V
        v_detected = False
        if hand_results.multi_hand_landmarks:
            for hand_lms in hand_results.multi_hand_landmarks:
                if is_v_sign(hand_lms): v_detected = True

        # --- BƯỚC 3: LOGIC ĐẾM NGƯỢC & LƯU ẢNH ---
        current_time = time.time()
        
        if v_detected and not is_counting_down and (current_time - last_saved_time > 2.0):
            is_counting_down = True
            countdown_start = current_time

        if is_counting_down:
            elapsed = current_time - countdown_start
            remaining = 3 - int(elapsed)

            if remaining > 0:
                cv2.putText(frame, str(remaining), (w//2 - 50, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 165, 255), 10)
            else:
                # Lưu ảnh sạch đã chép ở trên (photo_frame)
                filename = os.path.join(SAVE_DIR, f"photo_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(filename, photo_frame)
                saved_filename = filename
                is_counting_down = False
                last_saved_time = current_time

        # --- BƯỚC 4: HIỂN THỊ UI ---
        if current_time - last_saved_time < 2.0:
            cv2.putText(frame, "SAVED!", (w//2 - 80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.putText(frame, "Keys: [1]Glasses [2]Hat [3]Mustache [4]Color | [V-Sign]Capture | [Q]Quit", 
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Smart Photobooth - Final MVP', frame)
        
        # Bắt phím
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('1'): slot_eyes = not slot_eyes
        elif key == ord('2'): slot_head = not slot_head
        elif key == ord('3'): slot_mouth = not slot_mouth
        elif key == ord('4'): color_mode = (color_mode + 1) % 3

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()