import os

from time import time
from rembg import remove
from PIL import Image
from glob import glob


inp_img_dir = 'data/org_imgs_with_bg'
out_jpg_img_dir = 'data/org_imgs/'
os.makedirs(out_jpg_img_dir, exist_ok=True)

start_time_all = time()
for img_path in glob(inp_img_dir + '*'):
    start_time = time()
    img_name = os.path.basename(img_path).split('.')[0] + '.jpg'  # abc.??? -> abc -> abc.jpg
    out_img_path = os.path.join(out_jpg_img_dir, img_name)

    input = Image.open(img_path)
    output = remove(input)
    output = output.convert("RGB")
    output.save(out_img_path)
    print(img_name, time() - start_time)

print('Total process time:', time() - start_time_all)