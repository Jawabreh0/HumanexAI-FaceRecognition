import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
import pandas as pd
import os 
# Load face detection model
detector = MTCNN()

# Load face recognition model
facenet_model = load_model('/home/jawabreh/Desktop/Face-Recognition/facenet_keras.h5')

# Load face embeddings
data = np.load('/home/jawabreh/Desktop/Face-Recognition/embeddings/Embeddings.npz')
trainX, trainy = data['arr_0'], data['arr_1']

# Normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)

# Label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)

# Define the classes
class_names = out_encoder.classes_
class_names = np.append(class_names, 'unknown')

# Train SVM classifier
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

# Define function to extract face embeddings
def extract_face_embeddings(image):
    # Detect faces in the image
    faces = detector.detect_faces(image)
    if not faces:
        return None
    # Extract the first face only
    x1, y1, width, height = faces[0]['box']
    x2, y2 = x1 + width, y1 + height
    face = image[y1:y2, x1:x2]
    # Resize face to the size required by facenet model
    face = cv2.resize(face, (160, 160))
    # Preprocess the face for facenet model
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    face = np.expand_dims(face, axis=0)
    # Generate embeddings using facenet model
    embeddings = facenet_model.predict(face)
    return embeddings[0]

# Define function to identify the identity of an input image
def identify_person(image):
    # Extract face embeddings from input image
    embeddings = extract_face_embeddings(image)
    if embeddings is None:
        return None, None
    # Normalize embeddings
    embeddings = in_encoder.transform([embeddings])
    # Predict the identity and confidence using SVM classifier
    prediction = model.predict(embeddings)
    confidence = model.predict_proba(embeddings)[0][prediction] * 100
    prediction = out_encoder.inverse_transform(prediction)
    return prediction[0].item(), confidence

# Define function to identify the identity and confidence of an input image
def identify_person_with_unknown(image, threshold=0.9):
    # Extract face embeddings from input image
    embeddings = extract_face_embeddings(image)
    if embeddings is None:
        return None, None
    # Normalize embeddings
    embeddings = in_encoder.transform([embeddings])
    # Predict the identity and confidence using SVM classifier
    predictions = model.predict_proba(embeddings)[0]
    max_idx = np.argmax(predictions)
    if predictions[max_idx] >= threshold:
        prediction = out_encoder.inverse_transform([max_idx])
        confidence = predictions[max_idx]
        return prediction[0].item(), confidence
    else:
        return None, None

# Define path to directory containing images
image_dir = '/home/jawabreh/Desktop/Face-Recognition/evaluation/accuracy/accuracy-test-data/Unknown/'

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['filename', 'predicted_identity', 'confidence'])

# Iterate through each image in the directory
for filename in os.listdir(image_dir):
    # Read the image
    image_path = os.path.join(image_dir, filename)
    image = cv2.imread(image_path)

    # Call the identify_person_with_unknown function to get the predicted identity and confidence
    predicted_identity, confidence = identify_person_with_unknown(image)

    # Add the results to the DataFrame
    results_df = results_df.append({'filename': filename, 'predicted_identity': predicted_identity, 'confidence': confidence}, ignore_index=True)

# Save the results to an Excel sheet
results_df.to_excel('/home/jawabreh/Desktop/Face-Recognition/evaluation/accuracy/Unknown-accuracy-results.xlsx', index=False)

