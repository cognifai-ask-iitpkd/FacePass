import cv2
def camcapture():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not access the camera.")
        exit()
    ret, frame = camera.read()
    camera.release()
    if ret:
        return frame
    else:
        print("Error: Could not capture image.")
