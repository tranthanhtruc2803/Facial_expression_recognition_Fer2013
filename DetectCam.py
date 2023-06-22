import cv2
import numpy as np
from time import sleep
import tensorflow as tf
from keras.models import load_model

from efficientnet.tfkeras import EfficientNetB3
i = 0
# Model used
train_model = "ResNet"  # (Inception-v3, Inception-ResNet-v2): Inception,  (ResNet-50): ResNet

# Size of the images
img_width, img_height = 197, 197

emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

# Load trained model
model = tf.keras.models.load_model('ensembleEmo.h5')

# Create a face cascade
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

# Sets the video source fear, hp1, hp2, sd, anger, all
video_capture = cv2.VideoCapture('TestVid/all.mp4')

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

fps = video_capture.get(cv2.CAP_PROP_FPS)
fps = str(fps)

def preprocess_input(image):
    image = cv2.resize(image, (img_width, img_height))  # Resizing images for the trained model
    ret = np.empty((img_height, img_width, 3))
    ret[:, :, 0] = image
    ret[:, :, 1] = image
    ret[:, :, 2] = image
    x = np.expand_dims(ret, axis=0)  # (1, XXX, XXX, 3)

    x /= 127.5
    x -= 1.

    return x


def predict(emotion):
    # Generates output predictions for the input samples
    # x:    the input data, as a Numpy array (None, None, None, 3)
    prediction = model.predict(emotion)

    return prediction


while True:
    if not video_capture.isOpened():  # If the previous call to VideoCapture constructor or VideoCapture::open succeeded, the method returns true
        print('Unable to load camera.')
        sleep(5)  # Suspend the execution for 5 seconds
    else:
        sleep(0.5)
        ret, frame = video_capture.read()  # Grabs, decodes and returns the next video frame (Capture frame-by-frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Conversion of the image to the grayscale
        i = i + 1
        if i % 25 == 0:
            # Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles
            # image:		Matrix of the type CV_8U containing an image where objects are detected
            # scaleFactor:	Parameter specifying how much the image size is reduced at each image scale
            # minNeighbors:	Parameter specifying how many neighbors each candidate rectangle should have to retain it
            # minSize:		Minimum possible object size. Objects smaller than that are ignored
            faces = faceCascade.detectMultiScale(
                gray_frame,
                scaleFactor	= 1.2,
                minNeighbors = 6,
                minSize	= (100, 100))

            prediction = None
            x, y = None, None

            for (x, y, w, h) in faces:
                ROI_gray = gray_frame[y: y +h, x: x +w] # Extraction of the region of interest (face) from the frame

                cv2.rectangle(frame, (x, y), ( x +w, y+ h), (189,252,201), 2)

                emotion = preprocess_input(ROI_gray)
                prediction = predict(emotion)
                print(prediction[0][0])
                top_1_prediction = emotions[np.argmax(prediction)]
                print(top_1_prediction)
                # Draws a text string
                cv2.rectangle(frame, (x, y + h),
                              (x + w, y + h + 100), (189, 252, 201), -1)
                cv2.putText(frame, top_1_prediction, (x, y + (h + 70)), cv2.FONT_HERSHEY_SIMPLEX, 3, (25,25,112), 4,
                            cv2.LINE_AA)
                cv2.putText(frame, fps, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (25, 25, 112), 4,
                            cv2.LINE_AA)
                print(fps)
            # Display the resulting frame
            frame = cv2.resize(frame, (800, 500))
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()