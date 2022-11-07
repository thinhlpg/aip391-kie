import json
import math

from config import (
    craft_boxes_json_path,
    rot_angle_path
)


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


if __name__ == "__main__":
    input_path = craft_boxes_json_path
    output_path = rot_angle_path

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
        img_angles[img_id] = rotation_angle

    # xuất ra json file
    with open(output_path, 'w') as out:
        json.dump(img_angles, out)

