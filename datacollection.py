import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8)  # Higher confidence for better detection
offset = 20
imgSize = 300
counter = 0

folder = r"C:\Users\asus\Desktop\Sign-Language-detection - Copy (2)\Data\Yes"

# Distance range (in pixels, based on bounding box width)
MIN_DISTANCE = 100  # Too close if width < 100
MAX_DISTANCE = 250  # Too far if width > 250

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to capture image from webcam.")
        break

    imgOutput = img.copy()  # For displaying feedback
    hands, img = detector.findHands(img, draw=True)  # Draw hand landmarks for feedback
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Distance check using bounding box width
        if w < MIN_DISTANCE:
            cv2.putText(imgOutput, "Move farther from camera", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif w > MAX_DISTANCE:
            cv2.putText(imgOutput, "Move closer to camera", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Check for thumbs-up gesture using landmarks
            lmList = hand['lmList']  # List of 21 landmarks
            thumb_tip = lmList[4]   # Thumb tip (x, y)
            index_tip = lmList[8]   # Index finger tip (x, y)
            wrist = lmList[0]       # Wrist (x, y)

            # Thumbs-up: thumb tip higher than index tip and wrist
            if thumb_tip[1] < index_tip[1] and thumb_tip[1] < wrist[1]:
                cv2.putText(imgOutput, "Thumbs Up Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Crop and process image
                imgHeight, imgWidth = img.shape[:2]
                y1 = max(0, y - offset)
                y2 = min(imgHeight, y + h + offset)
                x1 = max(0, x - offset)
                x2 = min(imgWidth, x + w + offset)
                imgCrop = img[y1:y2, x1:x2]

                if imgCrop.size == 0:
                    print("Cropped image is empty. Skipping resize.")
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

                cv2.imshow('ImageCrop', imgCrop)
                cv2.imshow('ImageWhite', imgWhite)

                # Save image when 's' is pressed
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(f"Saved image {counter}")
            else:
                cv2.putText(imgOutput, "Show a Thumbs Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(imgOutput, "No Hand Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Image', imgOutput)
    key = cv2.waitKey(1)
    if key == ord("q"):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()