from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

ocr = PaddleOCR(use_angle_cls=True, lang='en')  # need to run only once to download and load model into memory
img_path = 'e.png'
# img_path = 'a.jpg'
result = ocr.ocr(img_path, cls=False)
# print(result)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txt = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txt, scores, font_path='Ubuntu.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
