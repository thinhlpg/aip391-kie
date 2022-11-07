import os
import json
import time

from glob import glob
from craft_text_detector import (
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    empty_cuda_cache
)
from config import (
    raw_train_img_nobg_dir,
    craft_boxes_json_path,
    raw_test_img_nobg_dir
)


if __name__ == "__main__":

    # set image path and export folder directory
    img_dir = raw_test_img_nobg_dir
    output_path = craft_boxes_json_path

    img_boxes = {}
    try:
        with open(output_path, 'r+') as out:
            saved_output = json.load(out)
    except FileNotFoundError:
        saved_output = {}

    refine_net = load_refinenet_model(cuda=True)
    craft_net = load_craftnet_model(cuda=True)

    start_time_alpha = time.time()
    for img_path in glob(img_dir + '/*'):
        img_id = os.path.basename(img_path)
        if img_id in saved_output:
            img_boxes[img_id] = saved_output[img_id]
            continue
        print(img_path)
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

    with open(output_path, 'w') as out:
        json.dump(img_boxes, out)
