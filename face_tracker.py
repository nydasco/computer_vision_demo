#!/usr/bin/env python3

# general
import numpy as np
import torch
from collections import defaultdict

# computer vision
import cv2
from ultralytics import YOLO


# configuration
model = YOLO("yolov8m-face.pt")

def init():
    """
    Initializes the global capture variable with a VideoCapture object.
    The VideoCapture object is set to capture video from the default camera (camera index 0).
    """
    global capture

    capture = cv2.VideoCapture(0)

# computer vision
def main():
    init()
    while True:
        success, frame = capture.read()
        if not success:
            break

        # Check if MPS is available (Macbook M1+)
        if torch.backends.mps.is_available():
            results = model.track(frame, persist = True, device="mps")
        else:
            results = model.track(frame, persist = True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("Face Tracking Demo", annotated_frame)

        # Exit if the user presses 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()