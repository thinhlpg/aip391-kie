import os
from config import cls_out_txt_dir, kie_out_txt_dir
from myutils.common import get_text_file_list


path = '/home/User/Documents'
print(os.path.dirname(path))
check_point_txt_list = get_text_file_list(cls_out_txt_dir)
print(cls_out_txt_dir)
print(check_point_txt_list)

kie_train_dir = os.path.dirname(kie_out_txt_dir)
print(os.path.join(kie_train_dir, 'images'))