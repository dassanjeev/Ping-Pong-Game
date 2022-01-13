import cv2
import numpy as np
import mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, HandNo=2, detectionConfidence=0.5, trackingConfidence=0.5):
        self.static_image_mode = mode
        self.max_num_hands = HandNo
        self.model_complexity = 1
        self.min_detection_confidence = detectionConfidence
        self.min_tracking_confidence = trackingConfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode,self.max_num_hands,self.model_complexity,self.min_detection_confidence,self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def process(self, img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handlandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlandmarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def fingerdetector(self,img,handNo=0):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            handlandmarks = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(handlandmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
        return self.lmList

    def boundingbox(self,img,lmList,draw=True):
        maxX, maxY, minX, minY = (0, 0, 0, 0)
        lmList = np.array(lmList)
        maxX, maxY, minX, minY = max(lmList[:, 1]), max(lmList[:, 2]), min(lmList[:, 1]), min(lmList[:, 2])
        if draw:
            cv2.rectangle(img, (minX - 10, minY - 10), (maxX + 10, maxY + 10), (250, 0, 0), 2)
        return img, (maxX, maxY, minX, minY)

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.process(img,draw=True)
        lmList = detector.fingerdetector(img)
        print(lmList)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
