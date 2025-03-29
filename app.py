import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf
from flask import Flask, Response, render_template, jsonify
import threading

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="Model/model_new.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.5)
offset = 0  # Reduced from 10 to 5 for a smaller box
imgSize = 300
labels = ["Hello", "Thank you", "Yes"]
prediction_history = []
history_size = 5

# Global settings
show_debug = False

def generate_frames():
    global prediction_history
    while True:
        success, img = cap.read()
        if not success:
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        label_text = "No Hand Detected"
        bbox = None

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            bbox = (x - offset, y - offset, x + w + offset, y + h + offset)

            imgHeight, imgWidth = img.shape[:2]
            y1 = max(0, y - offset)
            y2 = min(imgHeight, y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(imgWidth, x + w + offset)
            imgCrop = img[y1:y2, x1:x2]
            if imgCrop.size == 0:
                continue

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                if wCal > imgSize:
                    wCal = imgSize
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                if hCal > imgSize:
                    hCal = imgSize
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            input_shape = input_details[0]['shape']
            input_data = cv2.resize(imgWhite, (input_shape[1], input_shape[2]))
            input_data = np.expand_dims(input_data, axis=0).astype(np.float32) / 255.0
            interpreter.set_tensor(input_details[0]['index'], input_data)

            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            prediction_history.append(prediction[0])
            if len(prediction_history) > history_size:
                prediction_history.pop(0)
            avg_prediction = np.mean(prediction_history, axis=0)
            index = np.argmax(avg_prediction)
            confidence = avg_prediction[index]

            label_text = f"{labels[index]} ({confidence:.2f})"

        ret, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        with app.app_context():
            app.prediction_data = {'label': label_text, 'bbox': bbox}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    data = app.prediction_data if hasattr(app, 'prediction_data') else {'label': 'No Hand Detected', 'bbox': None}
    print(f"Sending prediction data: {data}")  # Debug log
    return jsonify(data)

if __name__ == "__main__":
    threading.Thread(target=generate_frames, daemon=True).start()
    app.run(debug=True, host='0.0.0.0', port=5000)