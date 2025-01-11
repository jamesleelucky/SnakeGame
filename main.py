import math
import random

import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# camera number 2 -> id == 1
cap = cv2.VideoCapture(1)

# size of the webcam
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)

class SnakeGameClass:
    def __init__(self, food_path):
        self.points = []
        self.lengths = []    # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150
        self.previousHead = 0, 0

        self.imgFood = cv2.imread(food_path, cv2.IMREAD_UNCHANGED)

        if self.imgFood is None:
            raise FileNotFoundError(f"Food image not found at path: {food_path}")

            # Add an alpha channel if it doesn't exist
        if self.imgFood.shape[2] == 3:
            self.imgFood = cv2.cvtColor(self.imgFood, cv2.COLOR_BGR2BGRA)

        self.hFood, self.wFood, _ = self.imgFood.shape # dimension of the food
        self.foodPoint = 0, 0
        self.score = 0
        self.gameOver = False

        self.random_food_location()

    def random_food_location(self):
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

    def update(self, image, current_head):
        if self.gameOver:
            cvzone.putTextRect(image, "Game Over!", [300,400], scale=7, thickness=5, offset=20)
            cvzone.putTextRect(image, f'Your Score: {self.score}', [300, 550], scale=7, thickness=5, offset=20)
        else:
            px, py = self.previousHead
            cx, cy = current_head

            self.points.append([cx,cy])
            distance = math.hypot(cx-px, cy-py)

            self.lengths.append(distance)
            self.currentLength += distance
            self.previousHead = cx, cy

            # length reduction
            if self.currentLength > self.allowedLength:
                for i, length in enumerate(self.lengths):
                    self.currentLength -= length
                    self.lengths.pop(i)
                    self.points.pop(i)

                    if self.currentLength < self.currentLength:
                        break

            # check if the snake ate the food
            rx, ry = self.foodPoint
            if rx - self.wFood//2 < cx < rx + self.wFood//2 and ry - self.hFood//2 < cy < ry + self.hFood//2:
                self.random_food_location()
                self.allowedLength += 10
                self.score += 1
                print(self.score)

            # Draw Snake
            if self.points:
                for i,point in enumerate(self.points):
                    if i != 0:
                        cv2.line(image, self.points[i-1], self.points[i], (0,255,255), 20)
                cv2.circle(image, self.points[-1], 20, (0, 0, 200), cv2.FILLED)

            # Draw Food
            image = cvzone.overlayPNG(image, self.imgFood, (rx-self.wFood//2, ry-self.hFood//2))

            # show score
            cvzone.putTextRect(image, f'Score: {self.score}', [50, 80], scale=3, thickness=3, offset=10)

            # check for collision
            pts = np.array(self.points[:-2], np.int32)  # ignore the closest two points from the head
            pts = pts.reshape((-2, 1, 2))
            cv2.polylines(image, [pts], False, (0, 200, 0), 3)
            min_distance = cv2.pointPolygonTest(pts, (cx, cy), True)
            print(min_distance)

            if -1 <= min_distance <= 1:
                print("Hit!")
                self.gameOver = True
                self.points = []
                self.lengths = []  # distance between each point
                self.currentLength = 0  # total length of the snake
                self.allowedLength = 150
                self.previousHead = 0, 0
                self.random_food_location()

        return image

game = SnakeGameClass("donut2.png")

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2] # excluding z axis (not 3D)
        img = game.update(img, pointIndex)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        game.gameOver = False
