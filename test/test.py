from ultralytics import YOLO
import numpy as np
import cv2
from show_result import plot_bboxes
from return_string import print_plate
from pre_process import pre_processing

detect_model = YOLO("D:/TIPA/License/Read_License_YOLOv8/models/Detect.pt")
ORC_model = YOLO("D:/TIPA/License/Read_License_YOLOv8/models/OCR-batch-1-4-23.pt")

image_name = "IPSS2.7_s244.6_2017-12-07.14-52-26.jpg"
#image = cv2.imread(f"D:/TIPA/License/Read_License_YOLOv8/test/test_img/{image_name}")
#image = cv2.imread(f"D:/TIPA/License/Read_License_YOLOv8/test/test_img/TEST BIEN SO/TEST BIEN SO/{image_name}")
image = cv2.imread(f"D:/TIPA/License/Read_License_YOLOv8/test/test_img/TEST BIEN SO/no_support/{image_name}")
image_arr = np.asarray(image)

license_results = detect_model.predict(image_arr)


boxes = license_results[0].boxes.xyxy.to('cpu').numpy().astype(int)
confidences = license_results[0].boxes.conf.to('cpu').numpy().astype(float)
labels = license_results[0].boxes.cls.to('cpu').numpy().astype(int) 

for box, conf, label in zip(boxes, confidences, labels):
    x_min, y_min, x_max, y_max = box
    image_crop = image[y_min:y_max, x_min:x_max]

license_img = pre_processing(image_crop)

results = ORC_model.predict(license_img)
plot_bboxes(license_img, results[0].boxes.boxes, score=False)
#print_plate(license_img, results[0].boxes.boxes)