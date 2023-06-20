# import packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak predictions")
args = vars(ap.parse_args())

# List of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Assigning random colors to each of the classes
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# initialize object ID counter
object_id = 1

# initialize dictionary to store object IDs
object_ids = {}

# initialize selected object ID
selected_object_id = None

# loop over the frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    object_counts = {}

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            idx = int(detections[0, 0, i, 1])

            if idx > 0 and idx < len(CLASSES):
                label = CLASSES[idx]
                object_counts[label] = object_counts.get(label, 0) + 1

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Assign object ID if it is a new object
                if label not in object_ids:
                    object_ids[label] = object_id
                    object_id += 1

                object_id_text = "ID: {}".format(object_ids[label])
                cv2.putText(frame, object_id_text, (startX, startY - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                # Highlight selected object
                if object_ids[label] == selected_object_id:
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  COLORS[idx], 2)

                text = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    for label, count in object_counts.items():
        cv2.putText(frame, f"{label}: {count}", (10, 25 + (count * 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # Select object using number keys
    key = chr(key)
    if key.isdigit():
        key = int(key)
        for label, obj_id in object_ids.items():
            if obj_id == key:
                selected_object_id = obj_id
                print(f"Selected Object: {label}")
                break

    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
