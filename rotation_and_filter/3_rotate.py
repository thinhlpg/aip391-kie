import json
import cv2
import numpy as np
import pandas as pd

from scipy import ndimage
from config import (
    rot_out_img_dir,
    rot_out_txt_dir,
    raw_train_img_dir,
    raw_train_img_nobg_dir,
    filtered_csv,
    rot_out_csv_path,
    craft_rot_out_txt_dir,
    rot_angle_path,
    craft_boxes_json_path
)


def write_output(boxes, texts, labels, result_file_path):
    result = ''
    for box, text, label in zip(boxes, texts, labels):
        line = ','.join((map(str, box))) + ',' + text.strip() + ',' + label
        result += line + '\n'  # 100, 200, ..., 123\n
    result = result.rstrip('\n')
    with open(result_file_path, 'w', encoding='utf8') as res:
        res.write(result)


def write_output_craft(boxes, result_file_path):
    result = ''
    for box in boxes:
        line = ','.join((map(str, box)))
        result += line + '\n'  # 100, 200, ..., 123\n
    result = result.rstrip('\n')
    with open(result_file_path, 'w', encoding='utf8') as res:
        res.write(result)


if __name__ == '__main__':

    angle_input_path = rot_angle_path
    csv_input_path = filtered_csv
    csv_output_path = rot_out_csv_path
    img_inupt_dir = raw_train_img_nobg_dir
    img_output_dir = rot_out_img_dir
    txt_output_dir = rot_out_txt_dir
    craft_boxes_json_path = craft_boxes_json_path
    craft_txt_out_dir = craft_rot_out_txt_dir

    df = pd.read_csv(csv_input_path, index_col=0)
    with open(angle_input_path) as inp:
        img_angles_dict = json.load(inp)
    with open(craft_boxes_json_path) as inp:
        craft_boxes_dict = json.load(inp)

    for img_id, angle in img_angles_dict.items():
        # mỗi bbox và poly được đặt trong 1 dict, mỗi img_id có 1 list gồm nhiều dict (anno_polygons)
        try:
            annotations = df.loc[img_id]['anno_polygons']
            texts = df.loc[img_id]['anno_texts'].split('|||')
            labels = df.loc[img_id]['anno_labels'].split('|||')
            craft_boxes = craft_boxes_dict[img_id]['boxes']
        except KeyError:
            continue

        # đọc ảnh, tính các thông số để xoay ảnh
        img_input_path = img_inupt_dir + '/' + img_id
        img = cv2.imread(img_input_path)
        h, w = img.shape[:2]
        cx, cy = (int(w / 2), int(h / 2))

        annotations = eval(annotations)
        rotated_bboxes = []
        for anno in annotations:
            # xoay ảnh
            img_rotated = ndimage.rotate(img, angle)
            img_output_path = img_output_dir + '/' + img_id
            cv2.imwrite(img_output_path, img_rotated)

            # xoay bbox
            bbox_2 = anno['bbox']
            bbox = [
                bbox_2[0], bbox_2[1],
                bbox_2[0], bbox_2[1] + bbox_2[3],
                bbox_2[0] + bbox_2[2], bbox_2[1] + bbox_2[3],
                bbox_2[0] + bbox_2[2], bbox_2[1]
            ]
            bbox_edges = [(bbox[i], bbox[i+1]) for i in range(len(bbox) - 1)[::2]]
            rotated_bbox = []
            for coord in bbox_edges:
                M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                cos, sin = abs(M[0, 0]), abs(M[0, 1])
                newW = int((h * sin) + (w * cos))
                newH = int((h * cos) + (w * sin))
                M[0, 2] += (newW / 2) - cx
                M[1, 2] += (newH / 2) - cy
                v = [coord[0], coord[1], 1]
                adjusted_coord = np.dot(M, v)
                rotated_bbox.append((adjusted_coord[0], adjusted_coord[1]))
            rotated_bbox = [int(x) for t in rotated_bbox for x in t]
            # xuất file text riêng nếu cần
            rotated_bboxes.append(rotated_bbox)
            # gán ngược lại anno x , y (top left), width, height
            anno['bbox'] = [rotated_bbox[0],
                            rotated_bbox[1],
                            rotated_bbox[4] - rotated_bbox[0],
                            rotated_bbox[5] - rotated_bbox[1]]

            # xoay poly
            poly = anno['segmentation'][0]
            poly_edges = [(poly[i], poly[i+1]) for i in range(len(poly) - 1)[::2]]
            rotated_poly = []
            for coord in poly_edges:
                M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                cos, sin = abs(M[0, 0]), abs(M[0, 1])
                newW = int((h * sin) + (w * cos))
                newH = int((h * cos) + (w * sin))
                M[0, 2] += (newW / 2) - cx
                M[1, 2] += (newH / 2) - cy
                v = [coord[0], coord[1], 1]
                adjusted_coord = np.dot(M, v)
                rotated_poly.append((adjusted_coord[0], adjusted_coord[1]))
            rotated_poly = [int(x) for t in rotated_poly for x in t]

            # gán ngược lại
            anno['segmentation'] = [rotated_poly]

        # xoay craft boxes
        rotated_craft_boxes = []
        for box in craft_boxes:
            rotated_craft_box = []
            for coord in box:
                M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                cos, sin = abs(M[0, 0]), abs(M[0, 1])
                newW = int((h * sin) + (w * cos))
                newH = int((h * cos) + (w * sin))
                M[0, 2] += (newW / 2) - cx
                M[1, 2] += (newH / 2) - cy
                v = [coord[0], coord[1], 1]
                adjusted_coord = np.dot(M, v)
                rotated_craft_box.append((adjusted_coord[0], adjusted_coord[1]))
            rotated_craft_box = [int(x) for t in rotated_craft_box for x in t]
            rotated_craft_boxes.append(rotated_craft_box)

        # gán lại df của img đó
        df.at[img_id, 'anno_polygons'] = annotations

        # xuất txt
        write_output(rotated_bboxes, texts, labels, txt_output_dir + '/' + img_id.replace('jpg', 'txt'))
        write_output_craft(rotated_craft_boxes, craft_txt_out_dir + '/' + img_id.replace('jpg', 'txt'))
        print(len(rotated_craft_boxes))

    df.to_csv(csv_output_path)
