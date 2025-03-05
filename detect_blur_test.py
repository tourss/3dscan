import cv2
import numpy as np
import os
import shutil

# 경로 설정
input_folder = r"C:\Users\MSI\Desktop\blur_test\images"
output_folder = os.path.join(input_folder, "no_blur")

# 폴더가 없으면 생성
os.makedirs(output_folder, exist_ok=True)

# 블러 감지 함수 (Laplacian 방식)
def is_not_blurry(image_path, threshold=60):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var > threshold  # 값이 크면 선명한 이미지

# 폴더 내 모든 PNG 파일 확인
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg")):
        file_path = os.path.join(input_folder, filename)

        # 블러 확인 후 복사
        if is_not_blurry(file_path):
            shutil.copy(file_path, os.path.join(output_folder, filename))
            print(f"✅ {filename} → no_blur 폴더로 복사됨")

print("✨ 블러가 없는 파일 복사가 완료되었습니다!")
