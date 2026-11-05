import cv2
import numpy as np
import os

class FaceAROverlay:
    def __init__(self):
        self.glasses = self._load_img("src/sunglasses.png")
        self.hat = self._load_img("src/hat.png")
        self.mustache = self._load_img("src/mustache.png")

    def _load_img(self, path):
        if not os.path.exists(path): return None
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[2] == 4: return img
        return None

    @staticmethod
    def rotate_image_bgra(image, angle):
        """Xoay ảnh 4 kênh mà không bị cắt mất góc"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
        
        cos, sin = np.abs(rot_mat[0, 0]), np.abs(rot_mat[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        rot_mat[0, 2] += (new_w / 2) - center[0]
        rot_mat[1, 2] += (new_h / 2) - center[1]
        
        return cv2.warpAffine(image, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    @staticmethod
    def overlay_transparent(background, overlay, center_x, center_y):
        bg_h, bg_w = background.shape[:2]
        ol_h, ol_w = overlay.shape[:2]

        x1, y1 = int(center_x - ol_w // 2), int(center_y - ol_h // 2)
        x2, y2 = x1 + ol_w, y1 + ol_h

        if x1 >= bg_w or y1 >= bg_h or x2 <= 0 or y2 <= 0: return background

        bg_x1, bg_y1 = max(0, x1), max(0, y1)
        bg_x2, bg_y2 = min(bg_w, x2), min(bg_h, y2)
        ol_x1, ol_y1 = bg_x1 - x1, bg_y1 - y1
        ol_x2, ol_y2 = ol_x1 + (bg_x2 - bg_x1), ol_y1 + (bg_y2 - bg_y1)

        roi_bg = background[bg_y1:bg_y2, bg_x1:bg_x2]
        roi_ol = overlay[ol_y1:ol_y2, ol_x1:ol_x2]

        ol_bgr = roi_ol[:, :, :3]
        alpha = roi_ol[:, :, 3:] / 255.0

        blended = (alpha * ol_bgr + (1.0 - alpha) * roi_bg).astype(np.uint8)
        background[bg_y1:bg_y2, bg_x1:bg_x2] = blended
        return background

    def apply_glasses(self, frame, left_eye_pt, right_eye_pt, eye_distance, angle):
        if self.glasses is None or eye_distance < 20: return frame
        target_w = int(eye_distance * 1.8)
        orig_h, orig_w = self.glasses.shape[:2]
        target_h = int(orig_h * (target_w / orig_w))
        resized = cv2.resize(self.glasses, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        rotated = self.rotate_image_bgra(resized, angle)
        cx, cy = (left_eye_pt[0] + right_eye_pt[0]) // 2, (left_eye_pt[1] + right_eye_pt[1]) // 2
        return self.overlay_transparent(frame, rotated, cx, cy)

    def apply_hat(self, frame, forehead_pt, eye_distance, angle):
        if self.hat is None or eye_distance < 20: return frame
        target_w = int(eye_distance * 2.5)
        orig_h, orig_w = self.hat.shape[:2]
        target_h = int(orig_h * (target_w / orig_w)) # Trả về tỷ lệ chuẩn
        
        resized = cv2.resize(self.hat, (target_w, target_h), interpolation=cv2.INTER_AREA)
        rotated = self.rotate_image_bgra(resized, angle)
        
        # Đẩy lên cao hơn trán một chút (20% chiều cao mũ) để lộ lông mày
        cx, cy = forehead_pt[0], forehead_pt[1] - int(target_h * 0.2)
        return self.overlay_transparent(frame, rotated, cx, cy)

    def apply_mustache(self, frame, upper_lip_pt, eye_distance, angle):
        if self.mustache is None or eye_distance < 20: return frame
        target_w = int(eye_distance * 1.1)
        orig_h, orig_w = self.mustache.shape[:2]
        target_h = int(orig_h * (target_w / orig_w))
        resized = cv2.resize(self.mustache, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        rotated = self.rotate_image_bgra(resized, angle)
        return self.overlay_transparent(frame, rotated, upper_lip_pt[0], upper_lip_pt[1])