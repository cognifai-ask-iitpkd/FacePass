from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def detect_face(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Failed to read the image.")
        return

    # Perform inference using YOLO
    results = model.predict(source=image, conf=0.25, show=False)

    # Extract and save the face
    for i, result in enumerate(results[0].boxes):
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        face = image[y1:y2, x1:x2]

        # Save the cropped face
        face_path = f"face_{i+1}.jpg"
        cv2.imwrite(face_path, face)
        print(f"Face saved as {face_path}")

        # Optionally, display the cropped face
        cv2.imshow("Detected Face", face)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


