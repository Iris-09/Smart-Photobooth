import cv2
import time

# Mở camera mặc định
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("LỖI: Không thể mở webcam.")
else:
    print("Webcam đã bật thành công!")
    print("=> Nhấn phím 'c' để CHỤP VÀ LƯU ẢNH.")
    print("=> Nhấn phím 'q' để THOÁT.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không nhận được tín hiệu hình ảnh.")
        break
        
    # Lật ảnh để tạo hiệu ứng soi gương chân thực
    frame = cv2.flip(frame, 1)

    # Hiển thị luồng video
    cv2.imshow('Smart Photobooth - Test Capture', frame)

    # Bắt sự kiện bàn phím
    key = cv2.waitKey(1) & 0xFF
    
    # Nếu nhấn phím 'c' (capture).\venv\Scripts\Activate.ps1python main.py
    if key == ord('c'):
        # Tạo tên file dựa trên thời gian thực để không bị ghi đè ảnh cũ
        file_name = f"captured_{int(time.time())}.jpg"
        
        # Lệnh lưu ảnh xuống ổ cứng
        cv2.imwrite(file_name, frame)
        print(f"📸 Tách! Đã lưu ảnh thành công: {file_name}")
        
        # Hiển thị chữ thông báo chớp nhoáng trên màn hình
        cv2.putText(frame, "DA CHUP!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Smart Photobooth - Test Capture', frame)
        cv2.waitKey(500) # Đóng băng khung hình nửa giây để tạo cảm giác vừa chụp xong
        
    # Nếu nhấn phím 'q' (quit)
    elif key == ord('q'):
        break

# Dọn dẹp bộ nhớ
cap.release()
cv2.destroyAllWindows()