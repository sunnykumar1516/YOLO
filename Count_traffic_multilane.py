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

    # Load the YOLOv8 model
    model = YOLO('best (1).pt')
    #model = YOLO("../yolov8l.pt")


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
    path = "../../videos/TopView/top_view_2.mp4"
    vs = cv.VideoCapture(path)
    prop = cv.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))


    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        frame = draw_line(frame)
        results = model.predict(frame,stream=False)
        if names == None:
            names = results[0].names

        for result in results:
            detections = []
            for data in result.boxes.data.tolist():
                _,_, _,_, conf, id = data
                name = names[id]
                detections.append(data)
                frame = draw_Box(data,frame,name)

            details = get_details(result, frame)
            tracks = object_tracker.update_tracks(details, frame=frame)

            cv.rectangle(frame, (100, 120), (1400, 30), (10, 10, 20), -1)

            cv.putText(frame, "Lane 1: " + str(count_lane_1), (100, 100), cv.FONT_HERSHEY_SIMPLEX, 2,
                       (0, 0, 200), 6)
            cv.putText(frame, "Lane 2: " + str(count_lane_2), (500, 100), cv.FONT_HERSHEY_SIMPLEX, 2,
                       (0, 0, 200), 6)
            cv.putText(frame, "Lane 3: " + str(count_lane_3), (1000, 100), cv.FONT_HERSHEY_SIMPLEX, 2,
                       (0, 0, 200), 6)





        for track in tracks:
            if not track.is_confirmed():
                break
            track_id = track.track_id
            bbox = track.to_ltrb()
            #if(track_id in cache_id):
                #cv.putText(frame, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1])), cv.FONT_HERSHEY_SIMPLEX, 1,
                          # (0, 255,0 ), 6)
            if(track_id not in cache_id):
                cv.putText(frame, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1])), cv.FONT_HERSHEY_SIMPLEX, 1,
                           (200,0, 0), 6)


            x = bbox[0]
            y = bbox[1]
            id = track_id

            if (x > 950) and (id not in cache_id) and (y > 258 and y < 454):

                cache_id.append(id)

                count_lane_1 = count_lane_1 + 1
                cv.putText(frame, "Lane 1: "+str(count_lane_1) , (100 ,100), cv.FONT_HERSHEY_SIMPLEX, 2,
                               (0, 0, 255), 6)
                p1 = (950, 258)
                p2 = (950, 454)
                frame = cv.line(frame, p1, p2, (255, 0, 0), thickness=10)


            elif (x > 950) and (id not in cache_id) and (y > 530 and y < 694):  # lane 2
                cache_id.append(id)
                count_lane_2 = count_lane_2 + 1
                cv.putText(frame, "Lane 2: "+str(count_lane_2), (500, 100), cv.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 255), 6)

                p3 = (950, 530)
                p4 = (950, 694)
                frame = cv.line(frame, p3, p4, (255, 0, 0), thickness=10)

            if (x < 950) and id not in cache_id  and  (y>730 and y<940):
                 cache_id.append(id)

                 count_lane_3 = count_lane_3 + 1
                 cv.putText(frame, "Lane 3: " + str(count_lane_3), (1000, 100), cv.FONT_HERSHEY_SIMPLEX, 2,
                                (0, 0, 255), 6)
                 p5 = (950, 730)
                 p6 = (950, 940)
                 frame = cv.line(frame, p5,p6, (255, 0, 0), thickness=10)


        if writer is None:
            # initialize our video writer
            fourcc = cv.VideoWriter_fourcc(*"MJPG")
            writer = cv.VideoWriter("traffic_top_indianRoad.mp4", fourcc, 24,
                                    (frame.shape[1], frame.shape[0]), True)
        print("writing")
        writer.write(frame)

        cv.imshow('image', frame)


        # waitKey() waits for a key press to close the window and 0 specifies indefinite loop
        cv.waitKey(24)






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





track_and_count_object();







