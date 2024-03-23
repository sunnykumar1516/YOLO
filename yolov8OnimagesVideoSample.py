import cv2 as cv
from ultralytics import YOLO



# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model
path = '../images/football.jpeg'
# Run batched inference on a list of images
results = model.predict(source=path, classes=[32])#selecting only a particular class for detection
# Process results list
for result in results:

    #if result.boxes.cls[4] == 32:

        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        #result.show()  # display to screen
        print(result.verbose()[0])
        #print(boxes.xywh)

        image = result.plot(labels=True)
        loc = result.boxes.xywh[0]
        loc = loc.numpy()
        print(loc)
        print(loc[0])

        center_coordinates = (int(loc[0]), int(loc[1]))
        radius = 80
        color = (0, 255, 0)
        thickness = -1
        sample =  cv.circle(image, center_coordinates, radius, color, thickness)
        cv.imwrite("cvSample.png", sample)




