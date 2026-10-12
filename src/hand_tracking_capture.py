import cv2
import mediapipe as mp
import time
import math

# Khởi tạo MediaPipe (giữ nguyên như trước)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Hàm kiểm tra chữ V (đã viết ở bước trước)
def is_v_sign(hand_landmarks):
    lm = hand_landmarks.landmark
    return (lm[8].y < lm[6].y and lm[12].y < lm[10].y and 
            lm[16].y > lm[14].y and lm[20].y > lm[18].y)

# 1. KHỞI TẠO CÁC BIẾN TRẠNG THÁI CHO BỘ ĐẾM
is_counting_down = False
countdown_start_time = 0
countdown_duration = 3  # Đếm ngược 3 giây

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    frame = cv2.flip(frame, 1)
    
    # 1. TẠO MỘT BẢN SAO SẠCH ĐỂ DÀNH RIÊNG CHO VIỆC LƯU ẢNH
    clean_frame = frame.copy() 

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True

    v_sign_detected = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 2. CHỈ VẼ KHUNG XƯƠNG ĐỎ KHI CHƯA ĐẾM NGƯỢC
            if not is_counting_down:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Vẫn ngầm kiểm tra chữ V để lấy tín hiệu
            if is_v_sign(hand_landmarks):
                v_sign_detected = True

    # ==========================================
    # LOGIC ĐẾM NGƯỢC & CHỤP ẢNH
    # ==========================================
    if v_sign_detected and not is_counting_down:
        is_counting_down = True
        countdown_start_time = time.time()

    if is_counting_down:
        elapsed_time = time.time() - countdown_start_time
        remaining_time = math.ceil(countdown_duration - elapsed_time)

        if remaining_time > 0:
            height, width, _ = frame.shape
            cv2.putText(frame, str(remaining_time), (width//2 - 50, height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
        else:
            # 3. HẾT GIỜ -> LƯU BẢN SAO SẠCH (clean_frame) THAY VÌ frame HIỆN TẠI
            filename = f"photobooth_{int(time.time())}.jpg"
            cv2.imwrite(filename, clean_frame) 
            print(f"Đã lưu ảnh sạch: {filename}")
            
            cv2.putText(frame, "FLASH!", (width//2 - 150, height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 10)
            cv2.imshow('Smart Photobooth - Hand Tracking', frame)
            cv2.waitKey(500) 
            
            is_counting_down = False

    cv2.imshow('Smart Photobooth - Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()