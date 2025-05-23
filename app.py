from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import os
import json
from datetime import datetime
import pickle
import imutils
from imutils.video import VideoStream
import time
import base64

app = Flask(__name__)

# Initialize global variables
vs = None
output_frame = None
lock = None
recognition_results = []
unknown_count = 0
known_count = 0
unknown_saved = False
output_message = None
is_recognition_running = False

# Load models and data
def load_models():
    global detector, embedder, recognizer, le, person_info
    
    # Load face detector
    protoPath = "face_detection_model/deploy.prototxt"
    modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    
    # Load face embedding model
    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
    
    # Load face recognition model
    recognizer = pickle.loads(open("output/recognizer", "rb").read())
    le = pickle.loads(open("output/le.pickle", "rb").read())
    
    # Load personal information
    with open("person_info.json", "r") as f:
        person_info = json.load(f)

# Create Unknown_images directory if it doesn't exist
unknown_dir = "Unknown_images"
if not os.path.exists(unknown_dir):
    os.makedirs(unknown_dir)

@app.route('/')
def index():
    return render_template('index.html')

def recognize_for_duration(duration=10):
    global vs, output_frame, lock, recognition_results, unknown_count, known_count, unknown_saved, output_message, is_recognition_running
    
    if is_recognition_running:
        return {
            'output_message': 'Recognition already in progress',
            'unknown_count': unknown_count,
            'known_count': known_count,
            'unknown_saved': unknown_saved,
            'results': recognition_results
        }
    
    is_recognition_running = True
    start_time = time.time()
    recognition_results = []
    MIN_RESULTS = 10
    MAX_RESULTS = 12
    confidence_threshold = 50.0
    unknown_count = 0
    known_count = 0
    unknown_saved = False
    output_message = None
    
    try:
        while True:
            if time.time() - start_time > duration:
                break
            if vs is None:
                raise Exception("Video stream not initialized")
                
            frame = vs.read()
            if frame is None:
                raise Exception("Failed to read frame from camera")
                
            frame = imutils.resize(frame, width=600)
            (h, w) = frame.shape[:2]
            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)
            detector.setInput(imageBlob)
            detections = detector.forward()
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    face = frame[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]
                    if fW < 20 or fH < 20:
                        continue
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                        (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()
                    preds = recognizer.predict_proba(vec)[0]
                    j = np.argmax(preds)
                    proba = preds[j]
                    name = le.classes_[j]
                    if proba > 0.5:
                        if name in person_info:
                            info = person_info[name]
                            known_count += 1
                        else:
                            name_lower = name.lower()
                            matching_key = next((k for k in person_info.keys() if k.lower() == name_lower), None)
                            if matching_key:
                                info = person_info[matching_key]
                                known_count += 1
                            else:
                                info = {"name": "Unknown", "id": "Unknown", "dob": "Unknown", "address": "Unknown"}
                                unknown_count += 1
                    else:
                        info = {"name": "Unknown", "id": "Unknown", "dob": "Unknown", "address": "Unknown"}
                        unknown_count += 1
                    unknown_filename = None
                    if info["name"] == "Unknown" and not unknown_saved and unknown_count > known_count:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        unknown_filename = f"unknown_{timestamp}.jpg"
                        unknown_path = os.path.join(unknown_dir, unknown_filename)
                        cv2.imwrite(unknown_path, face)
                        unknown_saved = True
                        print(f"Saved unknown face as {unknown_filename} (Unknown: {unknown_count}, Known: {known_count})")
                    if output_message is None:
                        if info["name"] == "Unknown":
                            if unknown_filename:
                                output_message = f"Output: Unknown (saved as {unknown_filename})"
                            else:
                                output_message = "Output: Unknown"
                        else:
                            output_message = f"Output: {info['name']} (ID: {info['id']}), Confidence: {proba * 100:.2f}%"
                    result = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "name": info["name"],
                        "confidence": float(proba * 100),
                        "id": info["id"]
                    }
                    if len(recognition_results) < MIN_RESULTS:
                        confidence_threshold = max(30.0, confidence_threshold - 1.0)
                    elif len(recognition_results) > MAX_RESULTS:
                        confidence_threshold = min(90.0, confidence_threshold + 1.0)
                    if result["confidence"] > confidence_threshold and len(recognition_results) < MAX_RESULTS:
                        recognition_results.append(result)
                        recognition_results = sorted(recognition_results, key=lambda x: x["confidence"], reverse=True)[:MAX_RESULTS]
    except Exception as e:
        print(f"Error during recognition: {str(e)}")
        raise e
    finally:
        is_recognition_running = False
        with open("recognition_results.json", "w") as f:
            json.dump(recognition_results, f, indent=4)
    
    return {
        'output_message': output_message,
        'unknown_count': unknown_count,
        'known_count': known_count,
        'unknown_saved': unknown_saved,
        'results': recognition_results
    }

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global vs
        try:
            if vs is None:
                vs = VideoStream(src=0).start()
                time.sleep(2.0)  # Allow camera to warm up
            
            while True:
                frame = vs.read()
                if frame is None:
                    break
                frame = imutils.resize(frame, width=600)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in video feed: {str(e)}")
            if vs is not None:
                vs.stop()
                vs = None
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    duration = request.json.get('duration', 10)
    result = recognize_for_duration(duration)
    return jsonify(result)

@app.route('/search_person', methods=['POST'])
def search_person():
    name = request.json.get('name', '').strip()
    if not name:
        return jsonify({'found': False, 'error': 'No name provided'})
    
    # Check if name exists in person_info
    if name in person_info:
        info = person_info[name]
        # Look for the person's image in the dataset
        dataset_path = os.path.join('dataset', name)
        if os.path.exists(dataset_path):
            # Get the first image from the person's folder
            image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                image_path = os.path.join(dataset_path, image_files[0])
                # Read and encode the image
                with open(image_path, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                return jsonify({
                    'found': True,
                    'name': name,
                    'id': info['id'],
                    'dob': info['dob'],
                    'address': info['address'],
                    'image': img_data
                })
    
    # If name not found or no image available
    return jsonify({
        'found': False,
        'message': 'Person not found in database'
    })

@app.route('/get_results')
def get_results():
    global output_message, unknown_count, known_count, unknown_saved, recognition_results
    return jsonify({
        'output_message': output_message,
        'unknown_count': unknown_count,
        'known_count': known_count,
        'unknown_saved': unknown_saved,
        'results': recognition_results
    })

if __name__ == '__main__':
    load_models()
    app.run(debug=True) 