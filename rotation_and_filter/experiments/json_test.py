import json

img_dict = {
    "img_1.jpg": {
      "boxes": [1, 2, 3, 4],
      "polys": [5, 4, 3, 2, 1]
    },
    "img_2.jpg": {
      "boxes": [1, 2, 3, 4],
      "polys": [5, 4, 3, 2, 1]
    },
    "img_3.jpg": {
      "boxes": [1, 2, 3, 4],
      "polys": [5, 4, 3, 2, 1]
    }
}
# poly_dict = {
# 'boxes': prediction_result['boxes'],

# }
# img_dict[img_id] = ''

print(img_dict['img_1.jpg']['boxes'])
