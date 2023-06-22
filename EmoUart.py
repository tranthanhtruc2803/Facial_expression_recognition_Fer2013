##################### Realtime Detector with webcam ####################
import cv2
import numpy as np
import onnxruntime as rt
import tensorflow as tf
import serial
import time

from PIL import Image

HAAR_CASCADE_XML_FILE_FACE = "/home/tttruc/project/MyCode/haarcascade_frontalface_default.xml"
img_width = 197
img_height = 197



# Initialize UART
print("UART Initialize...")

serial_port = serial.Serial(
    port="/dev/ttyUSB0",
    baudrate=9600,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE
)

time.sleep(1)

if serial_port.inWaiting() > 0:
	serial_port.write("Jetson Nano Emotion Detection Result\r\n".encode())

# Setup onnx model
output_path = "/home/tttruc/project/MyCode/ensemble.onnx"
output_names = ['add']
providers = ['CPUExecutionProvider']

m = rt.InferenceSession('ensemble.onnx', providers=providers)

# Prediction dict
emotion_dict = {0: "A", 1: "D", 2: "F", 3: "H", 4: "N", 5: "S", 6: "W"}
emotion_dict1 = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Obtain face detection Haar cascade XML files from OpenCV
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_XML_FILE_FACE)

# Video Capturing class from OpenCV
video_capture = cv2.VideoCapture("/dev/video0")
t = 0

while True:
	return_key, image = video_capture.read()
	cv2.namedWindow("emotion detection", cv2.WINDOW_AUTOSIZE)
	grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	detected_faces = face_cascade.detectMultiScale(image, 1.2, 7)
	t += 1
	# Create rectangle around the face in the image canvas
	for (x, y, w, h) in detected_faces:
		# Extract Region of Interest
		ROI_gray = image[y:y+h, x:x+h]
		# Draw bounding box
		cv2.rectangle(image, (x, y), (x + w, y + h), (189, 252, 201), 2)
		# Resize
		preprocess_image = cv2.resize(ROI_gray, (197, 197))
		preprocess_image = tf.keras.preprocessing.image.img_to_array(preprocess_image)		
		# Convert grayscale image into 3 channels image
		#arr = np.empty((197, 197, 3))
		#arr[:, :, 0] = preprocess_image
		#arr[:, :, 1] = preprocess_image
		#arr[:, :, 2] = preprocess_image
		# Scale input pixels between -1 and 1
		#arr = arr / 255		
		preprocess_image = preprocess_image / 255
		preprocess_image = np.expand_dims(preprocess_image, axis=0)
		if t % 15 == 0:
			# Interfere onnx model
			onnx_pred = m.run(output_names, {"input": preprocess_image})
			res = int(np.argmax(onnx_pred))
			top_1_prediction = emotion_dict[res]
			print_prediction = emotion_dict1[res]
			#print(top_1_prediction)
			print(print_prediction)
			serial_port.write(top_1_prediction.encode())
			#serial_port.write("\r\n".encode())

			# Draws a text string
			cv2.rectangle(image, (x, y + h),(x + w, y + h + 100), (189, 252, 201), -1) #Filled; -1
			cv2.putText(image, print_prediction, (x, y + (h + 70)), cv2.FONT_HERSHEY_SIMPLEX, 1, (25,25,112), 4, cv2.LINE_AA)

	# Display resulting frame
	cv2.imshow("emotion detection", image)

	# Stop the program on the ESC key
	key = cv2.waitKey(1) & 0xff
	if key == 27:
		break

# Release the capture
serial_port.close()
video_capture.release()
cv2.destroyAllWindows()
