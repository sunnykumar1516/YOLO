import cv2 as cv
import numpy as np

def draw_line(path):
    #cv2.line(image, start_point, end_point, color, thickness)

    image = cv.imread(path)
    start_P = (0,0)
    end_p = (100,500)

    color = (0,255,0)
    thick = 10

    img = cv.line(image,start_P,end_p,color,thick)

    return img

def draw_Box(path):
    #cv.rectangle(image, start_point, end_point, color, thickness)
    img = cv.imread(path)

    start_p =(0,0)
    end_p =(500,500)
    color = (0,200,40)
    thick = -1

    img = cv.rectangle(img,start_p,end_p,color,thick)
    return img

def draw_circle(path):
    #cv2.circle(image, center_coordinates, radius, color, thickness)
    img = cv.imread(path)

    center = (500,500)
    radius = 200
    color = (0,0,200)
    thick = -1
    img = cv.circle(img, center, radius, color, thick)

    return img

def draw_polygon(path):
    img = cv.imread(path)

    pts = np.array([[250, 70], [250, 160],
                    [1100, 200], [200, 160],
                    [1000, 70], [110, 20]],
                   np.int32)

    pts = pts.reshape((-1, 1, 2))

    closed = True
    color = (0,2,200)
    thick = 10
    img = cv.polylines(img, [pts],
                          closed, color, thick)

    return img



def display_image():
    path = "../images/football.jpeg"
    #img = draw_line(path)
    #img = draw_Box(path)
    #img = draw_circle(path)
    img = draw_polygon(path)
    cv.imshow("img", img)
    cv.waitKey(0)

display_image()






