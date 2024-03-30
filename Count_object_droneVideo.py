from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2 as cv
import imutils
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

depth = 0;

def track_and_count_object():
    count = 0;
    counter_cache=[]
    names = None

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
    # Load the YOLOv8 model
    model = YOLO('yolov8l.pt')

    # Open the video file
    path = "../videos/cattle/cattle2.mp4"
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

        results = model.predict(frame,stream=False)
        print(results[0].names)
        if names == None:
            names = results[0].names
        print(names[0])
        frame = draw_line(frame)
        cv.putText(frame, "Cattles crossing the", (frame.shape[0]-100 , 475), cv.FONT_HERSHEY_SIMPLEX, 3,
                   (0, 200, 255),
                   9)
        cv.putText(frame, " green line will be counted", (frame.shape[0], 550),
                   cv.FONT_HERSHEY_SIMPLEX, 3,
                   (0, 200, 255),
                   9)

        #frame = draw_polyLine(frame, (0, 255, 0))
        for result in results:
            detections = []
            for data in result.boxes.data.tolist():
                _,_, _,_, conf, id = data
                name = names[id]
                detections.append(data)
                #print("data:-",data)
                frame = draw_Box(data,frame,name)


            details = get_details(result,frame )
            tracks = object_tracker.update_tracks(details, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id

                ltrb = track.to_ltrb()
                bbox = ltrb

                cv.putText(frame, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1])), cv.FONT_HERSHEY_SIMPLEX, 1,
                           (0, 0, 255), 6)

                cv.putText(frame, "Cattle Count: " + str(count),(frame.shape[0]-50, 400), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255),
                           9)
                # counting
                if bbox[1] > int(frame.shape[0] / 2+500) and track_id not in counter_cache:
                    counter_cache.append(track_id)
                    count = count + 1
                    cv.putText(frame, "Cattle Count: " + str(count), (frame.shape[0]-50, 400), cv.FONT_HERSHEY_SIMPLEX, 3,
                               (0, 0, 255), 9)
                    frame = draw_polyLine(frame, (255, 0, 0))

        if writer is None:
            # initialize our video writer
            fourcc = cv.VideoWriter_fourcc(*"MJPG")
            writer = cv.VideoWriter("cattle.mp4", fourcc, 24,
                                    (frame.shape[1], frame.shape[0]), True)
        print("writing")
        writer.write(frame)

        cv.imshow('image', frame)

        # waitKey() waits for a key press to close the window and 0 specifies indefinite loop
        cv.waitKey(24)






def get_details(result,image):
    lt = result.boxes.xywh
    lt = lt.numpy()
    detections = []
    for item in lt:
        sample = (item, 0.5, "")
        detections.append(sample)

    return detections


def draw_Box(data,image,name):
    x1,y1,x2,y2,conf,id = data
    p1 = (int(x1),int(y1))
    p2 = (int(x2),int(y2))
    cv.rectangle(image,p1,p2,(0,0,255),3)
    cv.putText(image," ",p1, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return image




def draw_line(image):
    depth = int(image.shape[0]/2)
    p1 = (400,int(image.shape[0]/2 + 500))
    p2 = (image.shape[1]-200,int(image.shape[0]/2+500))
    print(p1,p2)
    image = cv.line(image, p1, p2, (0, 255, 0), thickness=10)

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







