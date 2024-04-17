import cv2 as cv
from ultralytics import YOLO
import random
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import imutils

def count_logs():
    count = 0
    path = "./video/logs.mp4"
    model = YOLO("Logs2.pt")


    writer = None
    (W, H) = (None, None)
    object_tracker = DeepSort(max_iou_distance=0.7,
                              max_age=5,
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

    vs = cv.VideoCapture(path)
    prop = cv.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))
    cache_count = []
    i = 0
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        if (len(cache_count)) > 10:
            cache_count.pop()

        cache_count.append(count)

        cv.rectangle(frame, (50, 60), (300, 10), (100, 10, 60), -1)
        cv.putText(frame, "Total Count:"+str(max(cache_count)) , (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        results = model.predict(frame, stream=False)

        cache_point=[]

        count = 0
        for result in results:

            for data in result.boxes.xywh.data.tolist():

                cache_point = data
                count = count+1
                frame = draw_circle(data,frame,count)



                '''  details = get_details(result, frame)
                tracks = object_tracker.update_tracks(details, frame=frame)

                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id

                    ltrb = track.to_ltrb()
                    bbox = ltrb
                    cv.putText(frame, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1])), cv.FONT_HERSHEY_SIMPLEX, 1,
                               (0, 0, 255), 3) '''



        if writer is None:
            # initialize our video writer
            fourcc = cv.VideoWriter_fourcc(*"MJPG")
            writer = cv.VideoWriter("./output/count.mp4", fourcc, 24,
                                    (frame.shape[1], frame.shape[0]), True)

        print("writing")
        writer.write(frame)

        cv.imshow('image', frame)
        cv.waitKey(24)


def draw_circle(data,image,count):
    x = int(data[0])
    y = int(data[1])
    center_coordinates = (x, y)
    radius = int(data[2]/2)

    thickness = 3
    c1 = random.randint(0, 255)
    c2 = random.randint(0, 255)
    c3 = random.randint(0, 255)

    color = (c1, c2, c3)
    image = cv.circle(image, center_coordinates, radius, color, thickness)



    #cv.putText(image, str(count), (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    return image

def draw_line(d1,d2,image):
    x1 = int(d1[0])
    y1 = int(d1[1])

    x2 = int(d2[0])
    y2 = int(d2[1])


    cv.line(image, (x1,y1),(x2,y2), (255, 0, 0), 5)
    return image

def get_details(result,image):

    classes = result.boxes.cls.numpy()
    conf = result.boxes.conf.numpy()
    xywh = result.boxes.xywh.numpy()

    detections = []
    for i,item in enumerate(xywh):
        sample = (item,conf[i] ,classes[i])
        detections.append(sample)

    return detections

count_logs()