import math
import random
import time
import cv2
import numpy as np
import cvzone
from cvzone.HandTrackingModule import HandDetector

# Camera setup
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

# Hand detector
detector = HandDetector(detectionCon=0.5, maxHands=1)

class SnakeGameClass:
    def __init__(self, food_path):
        self.points = []
        self.lengths = []
        self.currentLength = 0
        self.allowedLength = 150
        self.previousHead = 0, 0

        self.imgFood = cv2.imread(food_path, cv2.IMREAD_UNCHANGED)
        if self.imgFood is None:
            raise FileNotFoundError(f"Food image not found at path: {food_path}")
        if self.imgFood.shape[2] == 3:
            self.imgFood = cv2.cvtColor(self.imgFood, cv2.COLOR_BGR2BGRA)

        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 0, 0
        self.score = 0
        self.gameOver = False
        self.startTime = time.time()
        self.timeLimit = 120  # 2 minutes
        self.random_food_location()

    def random_food_location(self):
        self.foodPoint = random.randint(100, 500), random.randint(100, 400)

    def update(self, image, current_head):
        elapsedTime = time.time() - self.startTime
        if self.gameOver or elapsedTime > self.timeLimit:
            cvzone.putTextRect(image, "Game Over!", [100, 200], scale=3, thickness=3, offset=10)
            cvzone.putTextRect(image, f'Score: {self.score}', [100, 300], scale=3, thickness=3, offset=10)
            return image

        px, py = self.previousHead
        cx, cy = current_head
        self.points.append([cx, cy])
        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        self.currentLength += distance
        self.previousHead = cx, cy

        if self.currentLength > self.allowedLength:
            while self.currentLength > self.allowedLength and len(self.lengths) > 0:
                self.currentLength -= self.lengths.pop(0)
                self.points.pop(0)

        rx, ry = self.foodPoint
        if rx - self.wFood//2 < cx < rx + self.wFood//2 and ry - self.hFood//2 < cy < ry + self.hFood//2:
            self.random_food_location()
            self.allowedLength += 10
            self.score += 1

        for i in range(1, len(self.points)):
            cv2.line(image, tuple(self.points[i-1]), tuple(self.points[i]), (0, 255, 255), 10)
        if self.points:
            cv2.circle(image, self.points[-1], 15, (0, 0, 200), cv2.FILLED)

        image = cvzone.overlayPNG(image, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))
        cv2.putText(image, f"Score: {self.score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Self collision detection
        if len(self.points) > 4:
            pts = np.array(self.points[:-30], np.int32).reshape((-1, 1, 2))
            min_distance = cv2.pointPolygonTest(pts, (cx, cy), True)
            if min_distance >= -30:
                self.gameOver = True

        # Boundary collision detection
        h, w, _ = image.shape
        if cx <= 0 or cx >= w or cy <= 0 or cy >= h:
            self.gameOver = True

        return image

# Initialize game
game = SnakeGameClass("Donut.png")

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img, flipType=False)
    if hands:
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]
        img = game.update(img, pointIndex)

    elapsedTime = int(time.time() - game.startTime)
    remainingTime = max(0, game.timeLimit - elapsedTime)
    cv2.putText(img, f'Time Left: {remainingTime}s', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Snake Game", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        game = SnakeGameClass("Donut.png")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
