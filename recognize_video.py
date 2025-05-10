# import libraries
import os
import cv2
import imutils
import time
import pickle
import numpy as np
import json
from imutils.video import FPS
from imutils.video import VideoStream
from datetime import datetime
from heapq import nlargest

# load serialized face detector
print("Loading Face Detector...")
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load serialized face embedding model
print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("output/recognizer", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

# load personal information
print("Loading Personal Information...")
with open("person_info.json", "r") as f:
	person_info = json.load(f)

# Print available classes for debugging
print("Available classes in label encoder:", le.classes_)
print("Available keys in person_info:", list(person_info.keys()))

# initialize the video stream, then allow the camera sensor to warm up
print("Starting Video Stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

# Set start time
start_time = time.time()
recognition_results = []
MIN_RESULTS = 10  # Minimum number of results we want
MAX_RESULTS = 12  # Maximum number of results we want
confidence_threshold = 50.0  # Initial confidence threshold

# loop over frames from the video file stream
while True:
	# Check if 10 seconds have passed
	if time.time() - start_time > 10:
		break

	# grab the frame from the threaded video stream
	frame = vs.read()

	# resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			# Only consider it a match if probability is high enough
			if proba > 0.5:
				# Try to find the person in the JSON data
				if name in person_info:
					info = person_info[name]
				else:
					name_lower = name.lower()
					matching_key = next((k for k in person_info.keys() if k.lower() == name_lower), None)
					if matching_key:
						info = person_info[matching_key]
					else:
						info = {"name": "Unknown", "id": "Unknown", "dob": "Unknown", "address": "Unknown"}
			else:
				info = {"name": "Unknown", "id": "Unknown", "dob": "Unknown", "address": "Unknown"}

			# Store recognition result
			result = {
				"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
				"name": info["name"],
				"confidence": float(proba * 100),
				"id": info["id"]
			}
			
			# Adjust confidence threshold based on number of results
			if len(recognition_results) < MIN_RESULTS:
				# If we have too few results, lower the threshold
				confidence_threshold = max(30.0, confidence_threshold - 1.0)
			elif len(recognition_results) > MAX_RESULTS:
				# If we have too many results, increase the threshold
				confidence_threshold = min(90.0, confidence_threshold + 1.0)
			
			# Only add if confidence is high enough and we haven't reached max results
			if result["confidence"] > confidence_threshold and len(recognition_results) < MAX_RESULTS:
				recognition_results.append(result)
				# Sort results by confidence and keep only top MAX_RESULTS
				recognition_results = sorted(recognition_results, key=lambda x: x["confidence"], reverse=True)[:MAX_RESULTS]

			# draw the bounding box of the face along with the associated probability
			text = f"{info['name']}: {proba * 100:.2f}%"
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
			
			# Display personal information
			info_text = [
				f"ID: {info['id']}",
				f"DOB: {info['dob']}",
				f"Address: {info['address']}"
			]
			
			for idx, line in enumerate(info_text):
				y_pos = endY + 20 + (idx * 20)
				cv2.putText(frame, line, (startX, y_pos),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	# update the FPS counter
	fps.update()

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))

# Ensure we have at least MIN_RESULTS
if len(recognition_results) < MIN_RESULTS:
	print(f"Warning: Only got {len(recognition_results)} results (minimum desired: {MIN_RESULTS})")
elif len(recognition_results) > MAX_RESULTS:
	print(f"Warning: Got {len(recognition_results)} results (maximum desired: {MAX_RESULTS})")

# Save recognition results to a file
output_file = "recognition_results.json"
with open(output_file, "w") as f:
	json.dump(recognition_results, f, indent=4)
print(f"Recognition results saved to {output_file}")
print(f"Total results saved: {len(recognition_results)}")
print(f"Final confidence threshold: {confidence_threshold:.2f}%")

# cleanup
cv2.destroyAllWindows()
vs.stop()