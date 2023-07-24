import cv2
import mediapipe as mp

class HandTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackingCon=0.5):
        # Mediapipe Variables
        self.mode = mode 
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
    
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(
                static_image_mode=self.mode, 
                max_num_hands=self.maxHands, 
                min_detection_confidence=self.detectionCon,
                min_tracking_confidence=self.trackingCon
        )

    # Use Mediapipe to detect if any hands are in image and update
    # results accordingly.
    def detect_hands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

    # If any hands are in the results, draw each hand
    def draw_all_hands(self, img):
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)
        return img
    
    # Given a specific hand, return a list of each landmark's position.
    def find_hand_landmarks(self, img, handNo=0):
        hands = self.results.multi_hand_landmarks 
        if hands and handNo >= 0 and len(hands) > handNo:
            landmarks = []
            for lm in hands[handNo].landmark:
                height, width, _ = img.shape
                centerX, centerY = int(lm.x * width), int(lm.y * height)
                landmarks.append([centerX, centerY])
            return landmarks
