import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from VGG.vgg16 import face_model
from YOLO.yolo import yolo
from torch.nn.functional import cosine_similarity
from REG_USER.regUserDB import FaceDatabase
from cam import *
import time

#input photo to yolo and channel the output to face_model and get the weights of the input face and compare it with reg_user
def fpmain(thres, folderpath):
    FaceDB = FaceDatabase(thres, folderpath)
    while(1):
        flag=int(input("Do you want to take photo from your camera directly (1) or\ndo you have a jpg/png/jpeg file?(0):    "))
        if flag:
            frame = camcapture()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            time.sleep(1)
            yoloimg = yolo(pil_image)
            lock = FaceDB(yoloimg)
        else:
            filename=input("filename? :")
            inp = Image.open(filename).convert('RGB')
            yoloimg = yolo(inp)
            lock = FaceDB(yoloimg)

        if lock:
            print("lock is Unlocked")
        else:
            print("User not registered")
        
        inn=int(input("Do you want to try again?: "))
        if not inn:
            break
        else:
            update=int(input("Do you want re-check by scanning all the database? if yes (1) or no (0): "))
            if update:
                FaceDB.flag=1

if __name__== "__main__":
        threshold=float(input("Input a threshold for the lock: "))
        path=input("Input the folderpath where the registered user photos are present: ")
        fpmain(threshold, path)
        
        
        
        
    
    
    
    


