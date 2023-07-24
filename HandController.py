import cv2
import time
import math
import pyautogui
import mediapipe as mp
import HandTracking
from enum import IntEnum
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

pyautogui.FAILSAFE = False

class ControllerState(IntEnum):
    # Mapping of Controller States to Integers
    IDLE = 0
    MOVE_CURSOR = 1
    LEFT_CLICK = 2
    RIGHT_CLICK = 3
    DRAG = 4
    DOUBLE_CLICK = 5
    SCROLLING = 6

class HandController():
    # Camera Variables
    CAM_WIDTH, CAM_HEIGHT = 0, 0
    capture = None
    # State variables
    currentState = ControllerState.IDLE
    stateToSwitchTo = currentState
    # HandTracker object to detect hands
    handTracker = HandTracking.HandTracker(maxHands=1, detectionCon=0.7)
    # Variable that keeps track of the previous position
    prevHandPosition = None
    # Variable that keeps track of where the hand started at beginning of state
    startHandPosition = None
    # The number of frames that pass while holding a new gesture
    frameCount = 0
    # Whether or not the mouse can be clicked
    canClick = False
    # Whether the mouse is currently being dragged
    mouseDrag = False
    # Whether shift is being held down
    holdShift = False

    def __init__(self):
        # Set capture to default web camera
        HandController.capture = cv2.VideoCapture(0)
        HandController.CAM_WIDTH = HandController.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        HandController.CAM_HEIGHT = HandController.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    def getDistanceBetweenPoints(self, p1, p2):
        dist = (self.landmarks[p1][0] - self.landmarks[p2][0])**2
        dist += (self.landmarks[p1][1] - self.landmarks[p2][1])**2
        dist = math.sqrt(dist)
        return dist
    
    # Checks if a specific finger(not thumb) is open. 8 = Index, 12 = Middle, etc. 
    def isFingerOpen(self, landmarkNo):
        distFromKnuckleToWrist = self.getDistanceBetweenPoints(landmarkNo - 3, 0)
        distFromTipToWrist = self.getDistanceBetweenPoints(landmarkNo, 0)
        return (distFromTipToWrist * 0.6) > distFromKnuckleToWrist
    
    def isThumbOpen(self):
        return self.getDistanceBetweenPoints(4, 9) > 75

    def getHandPosition(self):
        knuckleLandmark = self.handTracker.results.multi_hand_landmarks[0].landmark[9]
        wt, ht = pyautogui.size()
        return (int(knuckleLandmark.x * wt), int(knuckleLandmark.y * ht))

    def getChangeInHandPositions(self, prev):
        # Get current hand position
        x, y = self.getHandPosition()
        # Find the "distance" between the previous and new hand position
        if prev:
            if self.prevHandPosition is None: self.prevHandPosition = (x, y)
            deltaX, deltaY= x - self.prevHandPosition[0], y - self.prevHandPosition[1]
            self.prevHandPosition = (x, y)
        else:
            if self.startHandPosition is None: self.startHandPosition = (x, y)
            deltaX, deltaY= x - self.startHandPosition[0], y - self.startHandPosition[1]
        deltaDist = (deltaX**2 + deltaY**2)**(1/2)
        return (deltaX, deltaY, deltaDist)

    def moveCursor(self):
        deltaX, deltaY, deltaDist = self.getChangeInHandPositions(True)
        # Check delta to see how far the mouse should be moved
        if deltaDist <= 10:
            ratio = 0
        elif deltaDist <= 35:
            ratio = 0.05 * deltaDist
        else:
            ratio = 1.5
        # Use the old mouse position to figure out the new position and move the mouse accordingly
        oldX, oldY = pyautogui.position()
        pyautogui.moveTo(oldX + deltaX * ratio, oldY + deltaY * ratio, 0.1)
    
    def scroll(self):
        deltaX, deltaY, _ = self.getChangeInHandPositions(False)
        if abs(deltaY) >= abs(deltaX):
            if abs(deltaY) >= 5:
                if self.holdShift:
                    self.holdShift = False
                    pyautogui.keyUp(key="shift")
                scrollAmount = -deltaY
                if abs(deltaY) > 125: scrollAmount = 125 if scrollAmount > 0 else -125
                pyautogui.scroll(scrollAmount)
        else:
            if abs(deltaX) >= 5:
                if not self.holdShift: 
                    self.holdShift = True
                    pyautogui.keyDown(key="shift")
                scrollAmount = deltaX
                if abs(deltaX) > 125: scrollAmount = 125 if scrollAmount > 0 else -125
                pyautogui.scroll(scrollAmount)

    # Check which gesture is currently being held. If a new gesture is held for 5 frames,
    # the controller will switch the corresponding state.
    def changeState(self):
        thumbOpen = self.isThumbOpen()
        indexOpen = self.isFingerOpen(8)
        middleOpen = self.isFingerOpen(12)
        ringOpen = self.isFingerOpen(16)
        pinkyOpen = self.isFingerOpen(20)
        candidateState = ControllerState.IDLE

        if (not thumbOpen and not pinkyOpen):
            if not ringOpen:
                if indexOpen and middleOpen: 
                    distBetweenIndexAndMiddle = self.getDistanceBetweenPoints(8, 12)
                    if distBetweenIndexAndMiddle >= 25: candidateState = ControllerState.MOVE_CURSOR
                    else: candidateState = ControllerState.DOUBLE_CLICK
                elif middleOpen: candidateState = ControllerState.LEFT_CLICK
                elif indexOpen: candidateState = ControllerState.RIGHT_CLICK
                else: 
                    candidateState = ControllerState.DRAG
            else:
                candidateState = ControllerState.SCROLLING

        if candidateState != self.currentState: 
            if candidateState != self.stateToSwitchTo:
                self.frameCount = 0
                self.stateToSwitchTo = candidateState
            else:
                self.frameCount += 1
                if (self.frameCount == 1): 
                    self.currentState = candidateState
                    print(self.currentState)

    def start(self):
        while HandController.capture.isOpened():
            # Get the current frame being displayed by the camera
            success, img = HandController.capture.read()
            # If the frame is empty, continue looping
            if not success: 
                print("Camera Frame Is Empty")
                continue
            # Using the HandTracker, detect any hand in frame. If a hand is detected
            # store and draw the landmarks.
            self.handTracker.detect_hands(img)
            img = self.handTracker.draw_all_hands(img)
            self.landmarks = self.handTracker.find_hand_landmarks(img)
            if self.landmarks: 
                self.changeState()
                # Reset the mouseDrag if the current state is not the DRAG state.
                if self.mouseDrag and self.currentState != ControllerState.DRAG:
                    self.mouseDrag = False
                    pyautogui.mouseUp(button="left")
                # Reset the previous hand position if the current state does not require it
                if (
                    self.currentState != ControllerState.MOVE_CURSOR and 
                    self.currentState != ControllerState.DRAG and
                    self.currentState != ControllerState.SCROLLING
                ): 
                        self.prevHandPosition = None
                # Reset holdShift if current state is not SCROLLING
                if self.holdShift and self.currentState != ControllerState.SCROLLING:
                    self.holdShift = False
                    pyautogui.keyUp(key="shift")
                # Switch case to check the current controller state. 
                match self.currentState:
                    case ControllerState.MOVE_CURSOR:
                        # The cursor is positioned at the middle knuckle
                        if not self.canClick: self.canClick = True
                        self.moveCursor()
                    case ControllerState.LEFT_CLICK:
                        if self.canClick:
                            pyautogui.click()
                            self.canClick = False
                    case ControllerState.RIGHT_CLICK:
                        if self.canClick:
                            pyautogui.click(button="right")
                            self.canClick = False
                    case ControllerState.DRAG:
                        if not self.mouseDrag:
                            self.mouseDrag = True
                            pyautogui.mouseDown(button="left")
                        self.moveCursor()
                    case ControllerState.DOUBLE_CLICK:
                        if self.canClick:
                            pyautogui.doubleClick()
                            self.canClick = False
                    case ControllerState.SCROLLING:
                        self.scroll()
            else:
                self.currentState = ControllerState.IDLE
                self.frameCount = 0

            cv2.imshow("Controller", img)
            if cv2.waitKey(5) == ord('q'):
                HandController.capture.release()
                cv2.destroyAllWindows()

def main():
    controller = HandController()
    controller.start()

if __name__ == "__main__":
    main()
