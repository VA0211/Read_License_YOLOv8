from ultralytics import YOLO
from PIL import Image
import numpy as np
from show_result import plot_bboxes

model = YOLO("D:/TIPA/License/Read_License_YOLOv8/test/ORC.pt")
image = Image.open("D:/TIPA/License/Read_License_YOLOv8/test/test_img/2120.jpg")
image_arr = np.asarray(image)
results = model.predict(image_arr)
#print(results[0].boxes.boxes)
plot_bboxes(image_arr, results[0].boxes.boxes, score=False, conf=0.5)