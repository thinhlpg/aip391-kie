import glob
import shutil
import os
import csv
from myutils.common import get_file_list
from config import rot_out_reversed_hand_labelled_dir, rot_out_reversed_filtered_dir, \
    rot_out_csv_path, reversed_filtered_csv


def filter_reversed_img(src_dir, dst_dir):
    #  dán nhãn bằng tay: có dấu - là ảnh có vấn đề.
    bad_imgs_count = 0
    for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
        file_name = os.path.basename(jpgfile)
        if '-' not in file_name:
            shutil.copy(jpgfile, dst_dir)
        else:
            bad_imgs_count += 1
            print(file_name)
    print(bad_imgs_count)


def filter_revesed_csv(src_img_dir, src_csv_path, dst_csv_path):
    nice_img_list = get_file_list(src_img_dir)

    with open(src_csv_path, 'r', encoding='utf-8') as csv_file:
        has_header = csv.Sniffer().has_header(csv_file.readline())
        csv_file.seek(0)  # reset current position to 0
        if has_header:
            header = csv_file.readline().rstrip('\n')

        filtered_row_list = []
        for idx, row in enumerate(csv_file.readlines()):
            img_id = row.split(',', 1)[0]
            if img_id in nice_img_list:
                filtered_row_list.append(row.rstrip('\n'))
        print('Number of filtered rows:', len(filtered_row_list))

        filtered_csv_text = '\n'.join(filtered_row_list)
        if has_header:
            filtered_csv_text = header + '\n' + filtered_csv_text

        with open(dst_csv_path, 'w', encoding='utf-8') as out:
            out.write(filtered_csv_text)


if __name__ == '__main__':
    src_img_dir = rot_out_reversed_hand_labelled_dir
    dst_img_dir = rot_out_reversed_filtered_dir
    src_csv_path = rot_out_csv_path
    dst_csv_path = reversed_filtered_csv

    # filter_reversed_img(src_img_dir, dst_img_dir)

    # filter_revesed_csv(dst_img_dir, src_csv_path, dst_csv_path)
