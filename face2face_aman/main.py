from say_cheese import take_photo
from crop_face import detect_face
from faceRecognition import find_similarity, show_img


img1 = "original.jpg"
img2 = "face_1.jpg"



# let the camera roll, press q when redy.



if __name__ == "__main__":
    take_photo()

    img_path = "./saved_img.jpg"
    detect_face(img_path)

    similarity = find_similarity( img1, img2 )

