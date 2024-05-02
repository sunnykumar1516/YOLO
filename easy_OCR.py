import easyocr
import cv2 as cv
import numpy as np

#get me text testing
def get_me_text():

    path = "./OCR_images/sample.jpg"
    reader = easyocr.Reader(['en']) # specify the language
    result = reader.readtext(path)
    image = cv.imread(path)
    print(image.shape)
    sample_img = np.zeros(image.shape, dtype=np.uint8)
    for (bbox, text, prob) in result:
        print(f'Text: {text}, Probability: {prob}')
        image = draw_box(image,bbox,text)

    display_image(image)


#draws box over image
def draw_box(image,bbbox,text):
    print("box",bbbox)
    start_p =(bbbox[0])
    end_p = (bbbox[2])
    print(f"start:{start_p[0]},end:{end_p}")

    color = (0, 255, 0)
    thick = 5

    image = cv.rectangle(image, (int(start_p[0]),int(start_p[1])), (int(end_p[0]),int(end_p[1])), color, thick)
    image = put_text(image,bbbox[0],text)
    return image

def put_text(image,bbox,text):
    font = cv.FONT_HERSHEY_SIMPLEX
    org = (int(bbox[0]), int(bbox[1]))
    fontScale = 2
    color = (10, 10, 150)
    thickness = 4
    image = cv.putText(image, text, org, font,
                        fontScale, color, thickness, cv.LINE_AA)
    return image
def display_image(image):
    cv.imshow("img", image)
    cv.waitKey(0)

get_me_text()
