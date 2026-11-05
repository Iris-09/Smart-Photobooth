import cv2
import mediapipe as mp
import math
from face_ar_overlay import FaceAROverlay

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def get_landmark_pixel(landmarks, index, img_w, img_h):
    lm = landmarks.landmark[index]
    return int(lm.x * img_w), int(lm.y * img_h)

def calculate_face_metrics(landmarks, img_w, img_h):
    left_eye = get_landmark_pixel(landmarks, 33, img_w, img_h)
    right_eye = get_landmark_pixel(landmarks, 263, img_w, img_h)
    forehead = get_landmark_pixel(landmarks, 10, img_w, img_h)
    upper_lip = get_landmark_pixel(landmarks, 164, img_w, img_h)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    eye_dist = math.hypot(dx, dy) 
    angle_deg = math.degrees(math.atan2(dy, dx)) # Tính lại góc nghiêng

    return {
        "left_eye": left_eye, "right_eye": right_eye, 
        "forehead": forehead, "upper_lip": upper_lip, 
        "eye_distance": eye_dist, "angle": angle_deg
    }

if __name__ == "__main__":
    ar_overlay = FaceAROverlay()
    cap = cv2.VideoCapture(0)

    slot_eyes = True
    slot_head = False
    slot_mouth = False
    color_mode = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # 1. FILTER MÀU INSTAGRAM THỰC THỤ TỪ KÊNH BGR
        if color_mode == 1:
            # Tách 3 kênh màu, đẩy tone ấm và tăng độ sáng nhẹ (Warm Peach Filter)
            b, g, r = cv2.split(frame)
            b = cv2.convertScaleAbs(b, alpha=0.9, beta=0)   # Giảm tone lạnh
            r = cv2.convertScaleAbs(r, alpha=1.1, beta=15)  # Tăng tone đỏ hồng & sáng da
            frame = cv2.merge((b, g, r))
        elif color_mode == 2:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 2. XỬ LÝ FACE MESH & AR
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = face_mesh.process(image_rgb)
        image_rgb.flags.writeable = True

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                metrics = calculate_face_metrics(face_landmarks, w, h)

                # Truyền thêm biến angle vào để phụ kiện xoay theo đầu
                if slot_head:
                    frame = ar_overlay.apply_hat(frame, metrics["forehead"], metrics["eye_distance"], metrics["angle"])
                if slot_eyes:
                    frame = ar_overlay.apply_glasses(frame, metrics["left_eye"], metrics["right_eye"], metrics["eye_distance"], metrics["angle"])
                if slot_mouth:
                    frame = ar_overlay.apply_mustache(frame, metrics["upper_lip"], metrics["eye_distance"], metrics["angle"])

        cv2.putText(frame, "Keys: [1]Glasses [2]Hat [3]Mustache | [4]IG Color Filter | [Q]Quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Smart Photobooth - Pro Filters', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('1'): slot_eyes = not slot_eyes
        elif key == ord('2'): slot_head = not slot_head
        elif key == ord('3'): slot_mouth = not slot_mouth
        elif key == ord('4'): color_mode = (color_mode + 1) % 3 

    cap.release()
    cv2.destroyAllWindows()