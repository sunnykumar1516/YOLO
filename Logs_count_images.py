import cv2 as cv
from ultralytics import YOLO
import random


def count_logs():
    count = 0
    path = "./image/logs.jpeg"
    model = YOLO("Logs2.pt")

    results = model.predict(path, stream=False)
    img = cv.imread(path, cv.IMREAD_COLOR)
    cache_point=[]
    for result in results:

        for data in result.boxes.xywh.data.tolist():
            #if (count > 0):
                #image = draw_line(cache_point, data, image)
            cache_point = data
            count = count+1
            image = draw_circle(data,img,count)




    cv.imshow('image', image)

    # waitKey() waits for a key press to close the window and 0 specifies indefinite loop
    cv.waitKey(0)


def draw_circle(data,image,count):
    x = int(data[0])
    y = int(data[1])
    center_coordinates = (x, y)
    radius = int(data[2]/2)

    thickness = 10
    c1 = random.randint(0, 255)
    c2 = random.randint(0, 255)
    c3 = random.randint(0, 255)

    color = (c1, c2, c3)
    image = cv.circle(image, center_coordinates, radius, color, thickness)



    cv.putText(image, str(count), (x,y), cv.FONT_HERSHEY_SIMPLEX, 3, color, 4)
    return image

def draw_line(d1,d2,image):
    x1 = int(d1[0])
    y1 = int(d1[1])

    x2 = int(d2[0])
    y2 = int(d2[1])


    cv.line(image, (x1,y1),(x2,y2), (255, 0, 0), 5)
    return image



count_logs()