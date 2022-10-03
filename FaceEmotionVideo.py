
# Importthe libraires
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
import time

# Support to calculate the FPs
time_actualframe = 0
time_prevframe = 0

# Classes
classes = ['angry','disgust','fear','happy','neutral','sad','surprise']

# We Load the model
prototxtPath = r"/home/grover/my_env/scripts/Face_Emotion-master/face_detector/deploy.prototxt" 
weightsPath = r"/home/grover/my_env/scripts/Face_Emotion-master/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# CPick the model
emotionModel = load_model("/home/grover/my_env/scripts/Face_Emotion-master/modelFEC.h5") #/home/grover/my_env/scripts/Face_Emotion-master/modelFEC.h5

# CAPTURE OF THE VIDEO
#cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam=cv2.VideoCapture(2)
# Take picture, face detection models
# Return the locations of the faces and the predictions of emotions of each face
def predict_emotion(frame,faceNet,emotionModel):
	# Build a blob from the image
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))

	# Performs face detection from the image
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# Lists to save faces, locations and predictions
	faces = []
	locs = []
	preds = []
	
	# Loop through each detection
	for i in range(0, detections.shape[2]):
		
		# Set a threshold to determine that the detection is reliable
		# Taking the probability associated with the detection

		if detections[0, 0, i, 2] > 0.4:
			# Take the bounding box of the scaled detection
			# according to the dimensions of the image
			box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
			(Xi, Yi, Xf, Yf) = box.astype("int")

			# Validate the dimensions of the bounding box
			if Xi < 0: Xi = 0
			if Yi < 0: Yi = 0
			
			#Extract the face and convert BGR to GRAY
			# Finally scaled to 224x244
			face = frame[Yi:Yf, Xi:Xf]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
			face = cv2.resize(face, (48, 48))
			face2 = img_to_array(face)
			face2 = np.expand_dims(face2,axis=0)

			# Add faces and locations to lists
			faces.append(face2)
			locs.append((Xi, Yi, Xf, Yf))

			pred = emotionModel.predict(face2)
			preds.append(pred[0])

	return (locs,preds)

while True:
	# Se toma un frame de la cámara y se redimensiona
	ret, frame = cam.read()
	#frame = imutils.resize(frame, width=640)
	frame= imutils.resize(frame, width=640)
	(locs, preds) = predict_emotion(frame,faceNet,emotionModel)
	
	# For each finding, the bounding box and the class are drawn on the image.
	for (box, pred) in zip(locs, preds):
		
		(Xi, Yi, Xf, Yf) = box
		(angry,disgust,fear,happy,neutral,sad,surprise) = pred


		label = ''
		# The probability is added in the image label
		label = "{}: {:.0f}%".format(classes[np.argmax(pred)], max(angry,disgust,fear,happy,neutral,sad,surprise) * 100)

		cv2.rectangle(frame, (Xi, Yi-40), (Xf, Yi), (255,0,0), -1)
		cv2.putText(frame, label, (Xi+5, Yi-15),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
		cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (255,0,0), 3)


	time_actualframe = time.time()

	if time_actualframe>time_prevframe:
		fps = 1/(time_actualframe-time_prevframe)
	
	time_prevframe = time_actualframe

	cv2.putText(frame, str(int(fps))+" FPS", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

	cv2.imshow("Frame", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		 break

cv2.destroyAllWindows()
cam.release()
