# HumanexAI Face Recognition

This project is a face recognition system powered by HumanexAI and coded by Ahmad Jawabreh, it uses MTCNN algorithm for face detection, FaceNet for feature extraction, and SVM for classification and identification. The system is able to detect and recognize faces in real-time video streams or still images, and can be customized to work with different datasets or classifiers, the algorithm giving accuracy of 99.63% on LFW dataset.

## Features
* MTCNN-based face detection: The system uses the MTCNN (Multi-Task Cascaded Convolutional Networks) algorithm for face detection, which is a popular and accurate method for detecting faces in images or videos.

* FaceNet-based feature extraction: The system uses FaceNet, a deep learning model for face recognition, to extract facial features from the detected faces. These features are then used for classification and identification.

* SVM-based classification: The system uses Support Vector Machines (SVM) for classification and identification of faces. SVM is a powerful and widely used algorithm for classification tasks, and is well-suited for face recognition applications.

* Real-time video stream processing: The system is able to process real-time video streams and detect/recognize faces in real-time.

* Customizable: The system can be customized to work with different datasets or classifiers, depending on the user's requirements.
## HumanexAI Face Recognition System Pipeline

![pipeline](pipeline.png)


## Use Case
* Survillance Systems 
* Biometric Passport
* Biometric Door Lock Systems
 


## Requirements
* Python 3.7.x
* Pillow
* matplotlib
* mtcnn
* tensorflow
* scikit-learn
* keras
* keras.models

## Installation
1. Clone this repository:
```bash
git clone https://github.com/username/repo-name.git
```

2. Install the required libraries::
```bash
pip install -r requirements.txt
```

3. Run the face recognition system:
```python
python identify.py
```

## Contributing
Contributions are welcome! If you find any issues or bugs, or if you have any suggestions for improving the system, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.
