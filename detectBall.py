from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2 as cv
import imutils
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np


def track_and_count_object():
    count_lane_1 = 0
    count_lane_2 = 0
    count_lane_3 = 0
    cache_id=[]
    names = None

    writer = None
    (W, H) = (None, None)

    model = YOLO("../../yolov8l.pt")
    
    #model = YOLO("football_detect.pt")


    object_tracker = DeepSort(max_iou_distance=0.7,
                              max_age=30,
                              n_init=3,
                              nms_max_overlap=1.0,
                              max_cosine_distance=0.2,
                              nn_budget=None,
                              gating_only_position=False,
                              override_track_class=None,
                              embedder="mobilenet",
                              half=True,
                              bgr=True,
                              embedder_gpu=True,
                              embedder_model_name=None,
                              embedder_wts=None,
                              polygon=False,
                              today=None
                              )
    # Open the video file
    path = "../../../videos/football_playing.mp4"
    vs = cv.VideoCapture(path)
    prop = cv.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))
    (grabbed, frame) = vs.read()
    width, height, channel = frame.shape
    heatmap_image = np.zeros((width, height, 1), np.uint8)
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
    
        blank_image = np.zeros(frame.shape,np.uint8)
        
        #frame = draw_line(frame)
        results = model.predict(frame,stream=False,classes=[32])
        if(results[0]):
            
            for result in results:
                detections = []
                for data in result.boxes.data.tolist():
                    x1,y1,x2,y2, conf, id = data
                    h1 = int(abs(y1 - y2))
                    w1 = int(abs(x1 - x2))
                    #name = names[id]
                    #detections.append(data)
                    #print("data:-",data)
                    blank_image[int(y1):int(y1+h1), int(x1):int(x1+w1)] = frame[int(y1):int(y1+h1), int(x1):int(x1+w1)]
                    frame = draw_Box(data,blank_image,"ball")
                    heatmap_image = draw_circle(data,heatmap_image)
            
           # heatmap_image = cv.distanceTransform(heatmap_image, cv.DIST_L2, 5)
            heatmap_image = cv.blur(heatmap_image, (11, 11))
            heatmap_image = np.uint8(heatmap_image)
            dist_output = cv.applyColorMap(heatmap_image, cv.COLORMAP_HOT)
            if writer is None:
                # initialize our video writer
                fourcc = cv.VideoWriter_fourcc(*"MJPG")
                writer = cv.VideoWriter("op_heatmap2.mp4", fourcc, 24,
                                        (frame.shape[1], frame.shape[0]), True)
            print("writing")
            writer.write(dist_output)
        

            #cv.imshow('image', dist_output)


        # waitKey() waits for a key press to close the window and 0 specifies indefinite loop
            #cv.waitKey(24)






def get_details(result,image):

    classes = result.boxes.cls.numpy()
    conf = result.boxes.conf.numpy()
    xywh = result.boxes.xywh.numpy()

    detections = []
    for i,item in enumerate(xywh):
        sample = (item,conf[i] ,classes[i])
        detections.append(sample)

    return detections


def draw_Box(data,image,name):
    x1,y1,x2,y2,conf,id = data
    p1 = (int(x1),int(y1))
    p2 = (int(x2),int(y2))
    cv.rectangle(image,p1,p2,(0,0,255),1)
    cv.putText(image,name,p1, cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

    return image

def draw_box_onLane(p1,p2,image,lane):
    cv.rectangle(image, p1, p2, (100, 100, 255), -1)
    cv.putText(image, lane, p1, cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 10), 3)
    return image




def draw_line(image):

    p1 = (950,258)
    p2 = (950,454)

    p3 = (950, 530)
    p4 = (950, 694)

    p5 = (950, 730)
    p6 = (950, 940)

    image = cv.line(image, p1, p2, (0, 255, 0), thickness=10)
    image = cv.line(image, p3, p4, (0, 255, 0), thickness=10)
    image = cv.line(image, p5, p6, (0, 255, 0), thickness=10)

    #image = draw_box_onLane(p1,(1200,454),image,"lane 1")
    #image = draw_box_onLane(p3,(1200,694),image,"lane 2")
    #image = draw_box_onLane(p5, (700, 940), image,"lane 3")
    return image


def draw_polyLine(image,col):
    pts = np.array([[400, int(image.shape[0]/2+500)], [image.shape[1]-200, int(image.shape[0]/2 + 500)],
                     [image.shape[1]-100, int(image.shape[0]/2 + 300)],[600,int(image.shape[0]/2 + 300) ]],
                   np.int32)

    pts = pts.reshape((-1, 1, 2))
    isClosed = True

    # Blue color in BGR
    color = col

    # Line thickness of 2 px
    thickness = 12

    # Using cv2.polylines() method
    # Draw a Blue polygon with
    # thickness of 1 px
    image = cv.polylines(image, [pts],
                          isClosed, color, thickness)

    return image


def movement_heatmap(img,data):
    width, height, channel = img.shape
    
    draw_circle(data, heatmap_image)
    heatmap_image= cv.distanceTransform(heatmap_image, cv.DIST_L2, 5)


def draw_circle(data, image):
    x1, y1, x2, y2, conf, id = data
    
    center = (int(x1), int(y1))
    radius = 10
    color = (200, 200)
    thick = -1
    img = cv.circle(image, center, radius, color, thick)
    
    return img

track_and_count_object();







