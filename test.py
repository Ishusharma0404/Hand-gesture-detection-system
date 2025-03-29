import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf

# Load the updated TFLite model
interpreter = tf.lite.Interpreter(model_path="Model/model_new.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input details:", input_details)
print("Output details:", output_details)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.5)
offset = 10  # Reduced offset for a smaller box
imgSize = 300
labels = ["Hello", "Thank you", "Yes"]
prediction_history = []
history_size = 5

# Colors for a professional look
TEXT_COLOR = (255, 255, 255)  # White text
BG_COLOR = (50, 50, 50)       # Dark gray background for text box
BOX_COLOR = (0, 0, 255)       # Red bounding box (changed from orange)
HEADER_COLOR = (30, 30, 30)   # Dark header background

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera")
        break
    imgOutput = img.copy()
    
    # Add a header bar
    cv2.rectangle(imgOutput, (0, 0), (img.shape[1], 50), HEADER_COLOR, -1)
    cv2.putText(imgOutput, "Sign Language Detector", (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)

    hands, img = detector.findHands(img)
    if hands:
        print("Hand detected:", hands[0]['bbox'])
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgHeight, imgWidth = img.shape[:2]
        y1 = max(0, y - offset)
        y2 = min(imgHeight, y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(imgWidth, x + w + offset)
        imgCrop = img[y1:y2, x1:x2]
        if imgCrop.size == 0:
            print("Empty crop, skipping")
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

        # Prepare input
        input_shape = input_details[0]['shape']
        input_data = cv2.resize(imgWhite, (input_shape[1], input_shape[2]))
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32) / 255.0
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        prediction_history.append(prediction[0])
        if len(prediction_history) > history_size:
            prediction_history.pop(0)
        avg_prediction = np.mean(prediction_history, axis=0)
        index = np.argmax(avg_prediction)
        confidence = avg_prediction[index]
        print("Prediction:", prediction[0], "Avg Prediction:", avg_prediction, "Index:", index, "Confidence:", confidence)

        # Professional text display with background
        label_text = f"{labels[index]} ({confidence:.2f})"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x, text_y = 10, 90  # Below header
        box_coords = ((text_x, text_y - 25), (text_x + text_size[0] + 10, text_y + 5))
        cv2.rectangle(imgOutput, box_coords[0], box_coords[1], BG_COLOR, -1)
        cv2.putText(imgOutput, label_text, (text_x + 5, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)

        # Small red bounding box around the detected hand
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), 
                      BOX_COLOR, 2)

        # Optional: Comment out to hide debug windows
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)
    else:
        print("No hand detected")
        cv2.putText(imgOutput, "No Hand Detected", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Sign Language Detector', imgOutput)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()