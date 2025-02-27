import os
import time
import cvzone
import cv2
from cvzone.PoseModule import PoseDetector

video = cv2.VideoCapture(0)

count = 0
button_press_time = 0
button_press_duration = 0.8  # seconds

detector = PoseDetector()
shirts_path = 'files/Shirts'
shirts = os.listdir(shirts_path)
scale_ratio = 280 / 190  # width of shirt / distance between lm11 to lm12 (calculated handly)

current_shirt = cv2.imread(os.path.join(shirts_path, shirts[0]), cv2.IMREAD_UNCHANGED)  # imread unchanged is necessary
left_button = cv2.imread('files/Resources/button.png', cv2.IMREAD_UNCHANGED)
left_button = cv2.resize(left_button, (90, 90))

while True:
    success, img = video.read()
    original_img = img.copy()  # Keep a clean copy of the original frame
    img = detector.findPose(img)

    lmList, bbox = detector.findPosition(img, bboxWithHands=False, draw=False)

    if lmList:
        lm11 = lmList[11][0:2]
        lm12 = lmList[12][0:2]
        lm21 = lmList[21][0:2]

        widthOfShirt = int((lm11[0] - lm12[0]) * scale_ratio)
        try:
            resized_shirt = cv2.resize(current_shirt, (widthOfShirt, int(1.32 * widthOfShirt)))
            original_img = cvzone.overlayPNG(original_img, resized_shirt, (lm12[0] - 45, lm12[1] - 40))
        except:
            pass

        original_img = cvzone.overlayPNG(original_img, left_button, (520, 180))

        if 420 < lm21[0] < 600 and 120 < lm21[1] < 240:
            if button_press_time == 0:
                button_press_time = time.time()
            elif time.time() - button_press_time >= button_press_duration:
                count = (count + 1) % len(shirts)
                current_shirt = cv2.imread(os.path.join(shirts_path, shirts[count]), cv2.IMREAD_UNCHANGED)
                button_press_time = 0  # Reset the timer
                print(f"Shirt changed to {shirts[count]}")
        else:
            button_press_time = 0  # Reset if the hand moves away

    cv2.imshow('webcam', original_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
