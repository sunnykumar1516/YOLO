from ultralytics import YOLO
import cv2 as cv

def process():
    writer = None
    (W, H) = (None, None)
    path = "../videos/gun.mp4"
    model = YOLO("gunModel.pt")
    vs = cv.VideoCapture(path)
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        results = model.predict(frame, stream=False)
        for result in results:
            for data in result.boxes.data.tolist():
                draw_Box(data,frame," ")
        
        if writer is None:
            # initialize our video writer
            fourcc = cv.VideoWriter_fourcc(*"MJPG")
            writer = cv.VideoWriter("op3.mp4", fourcc, 24,
                                    (frame.shape[1], frame.shape[0]), True)
        print("writing")
        writer.write(frame)
        cv.imshow('image', frame)
        cv.waitKey(24)
                

def draw_Box(data,image,name):
    x1,y1,x2,y2,_,_ = data
    p1 = (int(x1),int(y1))
    p2 = (int(x2),int(y2))
    cv.rectangle(image,p1,p2,(0,0,255),3)
    cv.putText(image, "gun", p1, cv.FONT_HERSHEY_SIMPLEX, 2, (0, 200, 255),
               9)
    
process()