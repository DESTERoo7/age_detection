from keras._tf_keras.keras.models import load_model
import cv2
import numpy as np
from mtcnn import MTCNN

# Load the trained model
model_path = "D:/fold/Project/im/age_detection_model.h5"  # Update with your model path
model = load_model(model_path)

print(" Model Loaded Successfully!")

# Initialize MTCNN detector
detector = MTCNN()
IMG_SIZE = 200  # Update based on your model input size

def preprocess_frame(frame):
    """Detects faces, crops, resizes, and normalizes the face region."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    faces = detector.detect_faces(rgb_frame)

    processed_faces = []
    face_boxes = []

    for face in faces:
        x, y, w, h = face['box']
        face_img = rgb_frame[y:y+h, x:x+w]  # Crop face
        face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))  # Resize
        face_img = face_img.astype("float32") / 255.0  # Normalize
        face_img = np.expand_dims(face_img, axis=0)  # Reshape for model input

        processed_faces.append(face_img)
        face_boxes.append((x, y, w, h))

    return processed_faces, face_boxes

# Start webcam capture
cap = cv2.VideoCapture(0)  # 0 = Default Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Stop if no frame is captured

    # Preprocess frame
    faces, boxes = preprocess_frame(frame)

    # Predict age for each detected face
    for i, face in enumerate(faces):
        predicted_age = model.predict(face)[0][0]  # Model outputs a single age value
        x, y, w, h = boxes[i]

        # Draw bounding box and age on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Age: {int(predicted_age)}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the output
    cv2.imshow("Real-Time Age Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
