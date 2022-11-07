import os
import shutil
from glob import glob


if __name__ == '__main__':
    mixed_img_dir = r'../data/raw/warmup_images'
    target_img_dir = r'../data/raw/warmup_images_train'
    target_csv_path = r'../data/raw/warmup_train.csv'

    img_id_list = []
    with open(target_csv_path, 'r', encoding='utf-8') as csv_file:
        for row in csv_file.readlines()[1:]:
            img_id_list.append(row.split(',', 1)[0])
        print('Image count:', len(img_id_list))
        print(img_id_list)

    # todo: copy các file có trong list sang target_img_dir
    for img_id in img_id_list:
        img_src_path = os.path.join(mixed_img_dir, img_id)
        img_dst_path = os.path.join(target_img_dir, img_id)
        print(img_src_path)
        print(img_dst_path)
        shutil.copyfile(img_src_path, img_dst_path)
