import cv2

# 1. Đọc ảnh BẮT BUỘC có cv2.IMREAD_UNCHANGED
img_path = "src/sunglasses.png"
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

if img is None:
    print(f"Lỗi: Không tìm thấy ảnh tại '{img_path}'.")
else:
    # shape của ảnh sẽ trả về (Chiều cao, Chiều rộng, Số kênh màu)
    print(f"Kích thước ảnh hiện tại: {img.shape}")
    
    # Kiểm tra xem ảnh có đúng 4 kênh (BGRA) không
    if img.shape[2] == 4:
        print("Ảnh chuẩn 4 kênh (BGRA).")
        
        # 2. Tách ảnh thành 2 phần để quan sát
        # Lấy 3 kênh đầu (Blue, Green, Red)
        bgr_channels = img[:, :, :3]
        # Lấy riêng kênh thứ 4 (Alpha)
        alpha_channel = img[:, :, 3]
        
        # Hiển thị thử
        cv2.imshow("1. Hinh anh goc (BGR)", bgr_channels)
        cv2.imshow("2. Mat na trong suot (Alpha Mask)", alpha_channel)
        
        print("Bấm phím bất kỳ trên cửa sổ ảnh để thoát")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Ảnh này chỉ có 3 kênh màu.")