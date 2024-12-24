import os
import cv2
from yolo5face.get_model import get_model
import numpy as np
from PIL import Image
        
def yolo(input_image):    

    model = get_model("yolov5n", device=-1, min_face=24)
    if isinstance(input_image, Image.Image):
        input_image = np.array(input_image)
    
    image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    boxes, _, scores = model(image, target_size=512)

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        confidence = scores[i]

        if confidence > 0.5:
            cropped_face = image[y_min:y_max, x_min:x_max]
            return cropped_face
    print("No face found.")
    exit(0)
    return None 

