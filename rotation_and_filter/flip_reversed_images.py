# todo: tìm cách để xoay 180 độ các ảnh bị ngược
#  pp 1: pytesseract osd -> nope vì đã thử từ trước, và nó flop
#  pp 2: dùng binary classification
#   - cần label hình. có thể in vertical hay horizontal ở bước tính góc, xong sửa các thằng vertical
#       (cột đầu là img_id, cột 2 là ngược hay ko)
#   - cần train model (kích thước ảnh lớn và không đồng đều) -> xử lý thế nào?