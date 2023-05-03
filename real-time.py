import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from sklearn.svm import SVC

# Load FaceNet model and MTCNN detector
model = load_model('/home/jawabreh/Desktop/HumaneX/face-recognition/facenet_keras.h5')
detector = MTCNN()

# Load face embeddings and labels
data = np.load('/home/jawabreh/Desktop/HumaneX/face-recognition/embeddings/unmasked-embeddings.npz')
face_embeddings, labels = data['arr_0'], data['arr_1']

# Train SVM classifier on face embeddings
classifier = SVC(kernel='linear', probability=True)
classifier.fit(face_embeddings, labels)

def get_embedding(model, face):
    # Preprocess the face image
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    face = np.expand_dims(face, axis=0)

    # Get the face embedding using the FaceNet model
    embedding = model.predict(face)

    return embedding[0]


# Define threshold for face recognition
threshold = 0.9

# Start video capture
cap = cv2.VideoCapture(0)

# Define default color
color = (0, 0, 255)  # Red

while True:
    # Read frame from video stream
    ret, frame = cap.read()

    # Detect faces in the frame
    results = detector.detect_faces(frame)

    # Iterate over detected faces
    for result in results:
        # Get face bounding box coordinates
        x1, y1, width, height = result['box']
        x2, y2 = x1 + width, y1 + height

        # Extract face embedding
        face = frame[y1:y2, x1:x2]
        face_embedding = get_embedding(model, face)

        # Predict label with SVM classifier
        confidence = classifier.predict_proba(face_embedding.reshape(1,-1))
        label = classifier.predict(face_embedding.reshape(1,-1))[0]

        # Label face with name or "unknown"
        if confidence[0][np.where(classifier.classes_ == label)[0][0]] > threshold:
            color = (0, 255, 0)  # green
        else:
            label = "unknown"

        # Draw bounding box and label on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show the frame with bounding box and label
    cv2.imshow("Object Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
