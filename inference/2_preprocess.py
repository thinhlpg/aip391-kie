import time
import os
import json
import math
import cv2
import numpy as np

from scipy import ndimage
from glob import glob
from config import inference_img_inp_dir, inference_preprocess_dir,\
    inference_extract_boxes_path, inference_angle_path,\
    inference_rotated_boxes_txt_dir, inference_rotated_img_dir
from craft_text_detector import (
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    empty_cuda_cache
)


def extract_boxes(input_imgs_dir, output_json_path):
    img_dir = input_imgs_dir
    output_boxes_json = output_json_path

    img_boxes = {}
    try:
        with open(output_boxes_json, 'r+') as out:
            saved_output = json.load(out)
    except IOError:
        saved_output = {}

    refine_net = load_refinenet_model(cuda=True)
    craft_net = load_craftnet_model(cuda=True)

    start_time_alpha = time.time()
    for img_path in glob(img_dir + '/*'):
        print(img_path)
        img_id = os.path.basename(img_path)
        if img_id in saved_output:
            img_boxes[img_id] = saved_output[img_id]
            continue

        # perform prediction
        start_time = time.time()
        prediction_result = get_prediction(
            image=img_path,
            craft_net=craft_net,
            refine_net=refine_net,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            cuda=True,
            long_size=1280
        )
        print(img_id, time.time() - start_time)

        boxes = prediction_result['boxes']
        boxes_as_ratios = prediction_result['boxes_as_ratios']
        boxes_dict = {
            'boxes': boxes if type(boxes) is list else boxes.tolist(),
            'boxes_as_ratios': boxes_as_ratios if type(boxes_as_ratios) is list else boxes_as_ratios.tolist(),
        }

        img_boxes[img_id] = boxes_dict
    print('Total time:', time.time() - start_time_alpha)

    with open(output_boxes_json, 'w') as out:
        json.dump(img_boxes, out)


def is_horizontal(boxes: list) -> bool:
    edge1_sum = 0
    edge2_sum = 0
    for box in boxes:
        # box chứa 4 list con (ví dụ [x1, y1]) là 4 đỉnh của box
        edge1_len = math.dist(box[0], box[1])
        edge2_len = math.dist(box[1], box[2])
        edge1_sum += edge1_len
        edge2_sum += edge2_len

    if edge1_sum > edge2_sum:
        return True
    return False


def get_img_angles(input_json_path, output_json_path):
    input_path = input_json_path
    output_path = output_json_path

    with open(input_path) as inp:
        img_boxes = json.load(inp)

    img_angles = {}
    for img_id, img in img_boxes.items():

        # xác định tọa độ các cạnh dài
        if is_horizontal(img['boxes']):
            print(img_id, 'is vertical.')
            long_edges = [(coord[0], coord[1]) for coord in img['boxes']]
        else:
            print(img_id, 'is horizontal.')
            long_edges = [(coord[1], coord[2]) for coord in img['boxes']]

        # tính trung bình góc xoay
        angle_sum = 0
        for (x1, y1), (x2, y2) in long_edges:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angle_sum += angle

        rotation_angle = angle_sum / len(long_edges) if len(long_edges) > 0 else 0
        print('Avg angle:', rotation_angle)
        img_angles[img_id] = rotation_angle

    # xuất ra json file
    with open(output_path, 'w') as out:
        json.dump(img_angles, out)


def write_output_craft(boxes, result_file_path):
    result = ''
    for box in boxes:
        line = ','.join((map(str, box)))
        result += line + '\n'  # 100, 200, ..., 123\n
    result = result.rstrip('\n')
    with open(result_file_path, 'w', encoding='utf8') as res:
        res.write(result)


def rotate_img(angle_input_path, img_inupt_dir, craft_boxes_json_path, img_output_dir, craft_txt_out_dir):

    with open(angle_input_path) as inp:
        img_angles_dict = json.load(inp)
    with open(craft_boxes_json_path) as inp:
        craft_boxes_dict = json.load(inp)

    for img_id, angle in img_angles_dict.items():
        try:
            craft_boxes = craft_boxes_dict[img_id]['boxes']
        except KeyError:
            continue

        # đọc ảnh, tính các thông số để xoay ảnh
        img_input_path = img_inupt_dir + '/' + img_id
        img = cv2.imread(img_input_path)
        h, w = img.shape[:2]
        cx, cy = (int(w / 2), int(h / 2))

        # xoay ảnh
        img_rotated = ndimage.rotate(img, angle)
        img_output_path = img_output_dir + '/' + img_id
        cv2.imwrite(img_output_path, img_rotated)

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

        # xuất txt
        write_output_craft(rotated_craft_boxes, craft_txt_out_dir + '/' + img_id[:-4] + '.txt')
        print(img_id, 'rotated', len(rotated_craft_boxes), 'boxes')


if __name__ == '__main__':
    img_dir = inference_img_inp_dir
    craft_boxes_json = inference_extract_boxes_path
    img_angle_json = inference_angle_path
    rotate_img_dir = inference_rotated_img_dir
    rotated_craft_boxes_dir = inference_rotated_boxes_txt_dir

    extract_boxes(img_dir, craft_boxes_json)
    get_img_angles(craft_boxes_json, img_angle_json)
    rotate_img(img_angle_json, img_dir, craft_boxes_json,
               rotate_img_dir, rotated_craft_boxes_dir)

