from ultralytics import YOLO
import cv2

# Load image from URL using requests
image_cv = cv2.imread("car.jpg")
model = YOLO("yolo11s-seg.pt")

results = model(image_cv)

results[0].show()
