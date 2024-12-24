# **Face Authentication System**

This project implements a face authentication system that uses a YOLO-based face detection model and VGG16-based feature extraction for facial recognition. The system compares the facial features of the input image with registered user images stored in a database to determine if the user is authorized.

---

## **Features**

1. **Face Detection**: Uses YOLOv5 to detect faces in the input image.
2. **Feature Extraction**: Employs a VGG16-based deep learning model for extracting facial features.
3. **Face Matching**: Compares the extracted features of the input face with registered faces using cosine similarity.
4. **Dynamic Database Updates**: Automatically updates the database when changes are detected in the registered user folder.

---

## **System Requirements**

- Python 3.7 or above
- A compatible GPU is recommended for faster model inference but is not required.
- Camera for live image capture (optional).

---

## **Key Components**
  YOLOv5: For face detection.
  VGG16: For facial feature extraction.
  Cosine Similarity: To compute the similarity between the input face and the database faces.
  
## **download the model file**
1. access this url "http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth"
2. Download this model, and place it in the VGG folder of the project 

## **Usage**
1. Install all the requirements
   > pip install -r requirements.txt
2. Add the registered users photos into ./REG_USER/registeredUser/ folder (it is advised to add many pictures) or you can have any folder of images, just be sure to give the path to the input of the program after running it. 
3. You can either use your camera to give input from your camera directly or you give path of the file where the image is and it will work
4. Note: Some times it might give Face not detected due to the YOLO model used might have some irregularities.
5. navigate to the project directory and run
   > python3 main.py
6. follow the instructions on screen to unlock the lock
7. Be sure to give the required paths in the inputs asked by the program

## **Future Enhancements**
1. Integrate with a web interface.
2. Add support for real-time video stream processing.
3. Improve feature extraction with newer architectures like ResNet or Transformer models.

## **Credits**
1. YOLOv5: YOLOv5 Face Model
2. VGG16: VGG16 Model
