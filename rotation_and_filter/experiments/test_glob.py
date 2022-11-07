from os import walk
from glob import glob

img_dir = r'../../data/raw/train_images/*'
for img_path in glob(img_dir)[:5]:
    print(img_path)
# for (dirpath, dirnames, filenames) in walk(img_dir):
#     print(filenames)