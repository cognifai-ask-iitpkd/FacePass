import cv2 


# let the camera roll, press q when redy.

def take_photo():
    webcam = cv2.VideoCapture(0)

    while True:
        status, frame = webcam.read()
        
        if status:
            cv2.imshow("Capturing", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.imwrite(filename='saved_img.jpg', img=frame)
                break


    webcam.release()
    cv2.destroyAllWindows()




