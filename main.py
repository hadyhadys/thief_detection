import cv2
import numpy as np
import time
from pygame import mixer


# Load YOLOv3 model and configuration files
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# initialize the first frame in the video stream
firstFrame = None

# capture video from a camera
cap = cv2.VideoCapture(0)

while True:
    # grab the current frame and initialize the occupied/unoccupied text
    ret, frame = cap.read()

    # if the frame could not be grabbed, then we have reached the end of the video
    if not ret:
        break

    # resize the frame and convert it to blob
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # loop over the detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":

                # get the bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(confidence)
                boxes.append([x, y, x + w, y + h])

    # perform NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # loop over the indices and draw the bounding boxes
    for i in indices:
        box = boxes[i]
        x, y = box[:2]

        # draw a bounding box around the human
        color = [int(c) for c in colors[class_id]]
        cv2.rectangle(frame, box[:2], box[2:], (95, 255, 89), 2)
        cv2.putText(frame, classes[class_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (95, 255, 89), 2)
        
    person_count = len(indices)
    text = "Number of detected persons: " + str(person_count)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (95, 255, 89), 2)
    # print("Number of detected persons: ", person_count)

    # play alarm sound
    if person_count >= 1 :
        mixer.init()
        sounda=mixer.Sound("alarm.wav")
        sounda.play()

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break