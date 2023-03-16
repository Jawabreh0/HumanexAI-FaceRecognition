# Face Feature Extracion using FaceNet (calculating the face embedding)
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

# load the face dataset
data = load('/home/jawabreh/Desktop/Face-Recognition/Datasets/Face_Detection_Datasets/Team_Unmasked_FaceDetection_Dataset.npz')
trainX, trainy  = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, )
# load the facenet model
model = load_model('/home/jawabreh/Desktop/Face-Recognition-Test/System_Code/facenet_keras.h5')
print('Loaded Model')
# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
	embedding = get_embedding(model, face_pixels)
	newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)
# save arrays to one file in compressed format
savez_compressed('/home/jawabreh/Desktop/Face-Recognition-Test/Datasets/Embedding_Datasets/Team_Unmasked_Embedding_Dataset.npz', newTrainX, trainy)