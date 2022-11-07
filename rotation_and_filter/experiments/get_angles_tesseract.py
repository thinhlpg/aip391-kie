import pytesseract as tess
import csv
import cv2
from glob import glob
from PIL import Image

# Script này đọc tất cả ảnh và trả về 1 file csv chứa góc xoay của chúng
input_dir = r'../../data/raw/train_images/*'
output_path = r'../../data/angle/angle.csv'
fields = ['img_id', 'angle']
rows = []
for img_path in glob(input_dir)[:20]:
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    (thresh, img_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow('', img_bw)
    # cv2.waitKey(0)

    # Dùng tess.image_to_osd để trích xuất góc của hình
    # ,config='--psm 0 -c min_characters_to_try=5', lang='vie'
    try:
        osd_result = tess.image_to_osd(img_bw, output_type=tess.Output.DICT,
                                       config='--psm 0 -c min_characters_to_try=10')
    except Exception:
        osd_result = dict()
        osd_result["rotate"] = -1
    print(img_path, osd_result["rotate"])

# with open(output_path, 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(fields)
#     writer.writerows(rows)
