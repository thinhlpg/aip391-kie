import os
import pandas as pd
import matplotlib.pyplot as plt
import imutils


from myutils.visualization import my_imshow


if __name__ == '__main__':
    org_img_input_dir = os.path.abspath(r'../data/raw/train_images') + '/'
    rotated_img_input_dir = os.path.abspath(r'../data/rotation_and_filter/imgs/') + '/'
    compare_output_dir = os.path.abspath('../data/rotation_and_filter/compare_results') + '/'
    csv_input_path = os.path.abspath(r'../data/rotation_and_filter/rotated_filtered_mcocr_train_df.csv')

    df = pd.read_csv(csv_input_path)

    for i in range(100, 200):
        img_id = df.iloc[i]['img_id']
        # org image
        org_img_path = org_img_input_dir + img_id
        org_img = plt.imread(org_img_path)

        # rotated img
        rotated_img_path = rotated_img_input_dir + img_id
        texts = df.iloc[i]['anno_texts'].split('|||')
        labels = df.iloc[i]['anno_labels'].split('|||')

        anno_polygons = eval(df.iloc[i]['anno_polygons'])
        bboxes = []
        for anno in anno_polygons:
            bboxes.append(anno['bbox'])

        rotated_img = my_imshow(img_path=rotated_img_path, bboxes=bboxes,
                                texts=texts, labels=labels, show=False)

        # side by side compare
        plt.subplot(1, 2, 1)
        plt.imshow(org_img)
        plt.title('Original')

        plt.subplot(1, 2, 2)
        plt.imshow(rotated_img)
        plt.title('Rotated')
        plt.savefig(compare_output_dir + img_id)