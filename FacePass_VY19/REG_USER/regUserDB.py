import os
import torch
import torch.nn.functional as F
from PIL import Image
from VGG.vgg16 import face_model
from YOLO.yolo import yolo

class FaceDatabase:
    def __init__(self, threshold, folderPath):
        self.folder_path = folderPath
        self.threshold = threshold
        self.facialfeatures = []
        self.flag = 1
        self.last_folder_state = self.get_folder_state()
    
    def get_folder_state(self):
        return {f: os.path.getmtime(os.path.join(self.folder_path, f)) for f in os.listdir(self.folder_path)}
    
    def update_database(self):
        current_state = self.get_folder_state()
        if current_state != self.last_folder_state or self.flag == 1:
            self.facialfeatures = []  
            for file_name in os.listdir(self.folder_path):
                file_path = os.path.join(self.folder_path, file_name)
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = Image.open(file_path).convert('RGB')
                    yolo1=yolo(img)
                    transformed_image = face_model.transform(yolo1)
                    input_tensor = transformed_image.unsqueeze(0)
                    features = face_model.facial_features(input_tensor)
                    self.facialfeatures.append(features)
            self.last_folder_state = current_state
            self.flag = 0
            print("Database updated.")
    
    def compare_input(self, img):
        self.update_database()
        yolo1=yolo(img)
        transformed_image = face_model.transform(yolo1)
        input_tensor = transformed_image.unsqueeze(0)
        features = face_model.facial_features(input_tensor)
        best_similarity=0
        for stored_feature in self.facialfeatures:
            tensor1 = features.flatten()
            tensor2 = stored_feature.flatten()
            best_similarity = max((F.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0), dim=1)).item(), best_similarity)
        if best_similarity>=self.threshold:
            print(best_similarity)
            return True
        return False
    __call__= compare_input

    
            
        
    