import os


CONFIG_ROOT = os.path.dirname(__file__)


def full_path(sub_path, file=False):
    path = os.path.join(CONFIG_ROOT, sub_path)
    if not file and not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            print('full_path. Error makedirs',path)
    return path


gpu = '0'  # None or 0,1,2...

# org_imgs data from organizer
raw_train_img_dir = full_path('data/raw/train_images')
raw_train_img_nobg_dir = full_path('data/raw/train_images_nobg')
raw_test_img_dir = full_path('data/raw/noempty_nohorizontal_warmup_images_train')
raw_test_img_nobg_dir = full_path('data/raw/test_images_nobg')
raw_test_csv = full_path('data/raw/noempty_nohorizontal_warmup_train.csv', file=True)
raw_csv = full_path('data/raw/mcocr_train_df.csv', file=True)

# EDA
json_data_path = full_path('EDA/final_data.json', file=True)
# filtered_train_img_dir=full_path('data/mc_ocr_train_filtered')
filtered_csv = full_path('data/rotation_and_filter/filtered_mcocr_train_df.csv', file=True)

# text detector
# det_model_dir = full_path('text_detector/PaddleOCR/inference/ch_ppocr_server_v2.0_det_infer')
# det_visualize = True
# det_db_thresh = 0.3
# det_db_box_thresh = 0.3
# det_out_viz_dir = full_path('text_detector/{}/viz_imgs'.format(dataset))
# det_out_txt_dir = full_path('text_detector/{}/txt'.format(dataset))

# rotation_and_filter corrector
craft_boxes_json_path = full_path('data/rotation_and_filter/img_craft_boxes.json', file=True)
rot_angle_path = full_path('data/rotation_and_filter/img_angles.json', file=True)
rot_out_img_dir = full_path('data/rotation_and_filter/imgs')
rot_out_txt_dir = full_path('data/rotation_and_filter/txt')
rot_out_csv_path = full_path('data/rotation_and_filter/rotated_filtered_mcocr_train_df.csv', file=True)
craft_rot_out_txt_dir = full_path('data/rotation_and_filter/txt_craft')
rot_out_reversed_hand_labelled_dir = full_path('data/rotation_and_filter/imgs_reversed_hand_labelled')
rot_out_reversed_filtered_dir = full_path('data/rotation_and_filter/imgs_reversed_filtered')
reversed_filtered_csv = full_path('data/rotation_and_filter/rotated_filtered_no_reversed.csv', file=True)

# text classifier (OCR)
cls_visualize = True
cls_ocr_thres = 0.65
cls_model_path = full_path('text_classifier/vietocr/vietocr/weights/vgg19_bn_seq2seq.pth', file=True)
cls_base_config_path = full_path('mc_ocr/text_classifier/vietocr/config/base.yml', file=True)
cls_config_path = full_path('mc_ocr/text_classifier/vietocr/config/vgg-seq2seq.yml', file=True)
cls_out_viz_dir = full_path('data/text_classifier/viz_imgs')
cls_out_txt_dir = full_path('data/text_classifier/txt')

# key information
kie_visualize = True
kie_model = full_path('PICK-pytorch-master/saved/models/PICK_Default/test_0121_212713/model_best.pth', file=True)
kie_boxes_transcripts = full_path('data/kie_results/boxes_and_transcripts')
kie_out_txt_dir = full_path('data/kie_results/txt')
combined_kie_out_txt_dir = full_path('data/kie_results/combined_txt')
combined_kie_train_images = full_path('data/kie_results/images')
kie_train_dir = full_path('data/kie_results')
kie_out_viz_dir = full_path('data/kie_results/viz_imgs')
new_kie_out_txt_dir = full_path('data/kie_results/new_txt')
new_kie_inp_img_dir = full_path('data/kie_results/new_images')
new_kie_out_viz_dir = full_path('data/kie_results/new_viz_imgs')

# inferences
inference_img_inp_dir = full_path(r'inference/data/org_imgs')
inference_extract_boxes_path = full_path('inference/data/preprocess/extract_boxes.json', file=True)
inference_angle_path = full_path('inference/data/preprocess/angle.json', file=True)
inference_rotated_boxes_txt_dir = full_path(r'inference/data/preprocess/rotated_boxes')
inference_rotated_img_dir = full_path(r'inference/data/preprocess/rotated_imgs')
inference_preprocess_dir = full_path(r'inference/data/preprocess')
inference_cls_out_txt_dir = full_path(r'inference/data/text_classifier/txt')
inference_cls_out_tsv_dir = full_path(r'inference/data/text_classifier/tsv')
inference_cls_out_viz_dir = full_path(r'inference/data/text_classifier/viz_imgs')
inference_pick_out_txt_dir = full_path(r'inference/data/output')
inference_pick_out_viz_dir = full_path(r'inference/data/output_viz')

if __name__ == '__main__':
    print(cls_out_txt_dir)
