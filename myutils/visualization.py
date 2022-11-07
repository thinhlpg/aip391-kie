import cv2
import numpy as np
import imutils

from PIL import ImageFont, ImageDraw, Image


def my_imshow(img_path, bboxes, labels, texts, show=True):

    img = cv2.imread(img_path)
    cv2.imshow('', img)

    # Draw bboxes
    for bbox in bboxes:
        pt1 = (bbox[0], bbox[1])
        if len(bbox) == 4:
            pt2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
        else:
            pt2 = (bbox[4], bbox[5])

        color = (0, 255, 0)
        thickness = 2
        line_type = cv2.LINE_4
        img = cv2.rectangle(img, pt1, pt2, color, thickness, line_type)

    # Draw texts
    font_path = r"C:\Users\thinhtu1203\AppData\Local\Microsoft\Windows\Fonts\SVN-Arial Regular.ttf"
    font = ImageFont.truetype(font_path, 20)
    img_pil = Image.fromarray(img)

    for bbox, label, text in zip(bboxes, labels, texts):
        text = label + '\n' + text
        pt1 = (bbox[0], bbox[1] + 10)
        draw = ImageDraw.Draw(img_pil)
        draw.text(pt1,  text, font=font, fill=(0, 123, 255, 0))
        img_text = np.array(img_pil)

    if show is True:
        img_resized = imutils.resize(img_text, height=720)
        cv2.imshow('', img_resized)
        cv2.waitKey(0)

    return img_text