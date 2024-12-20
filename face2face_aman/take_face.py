from say_cheese import take_photo
from crop_face import detect_face


take_photo()

img_path = "./saved_img.jpg"
detect_face(img_path)