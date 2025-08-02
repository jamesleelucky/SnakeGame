import math
import random
import time
import cv2
import numpy as np
import cvzone
from cvzone.HandTrackingModule import HandDetector

# Camera setup (lower resolution for speed)
cap = cv2.VideoCapture(1)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Hand detector with lower detection confidence for better accuracy
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
        self.random_food_location()

    def random_food_location(self):
        self.foodPoint = random.randint(100, 500), random.randint(100, 400)

    def update(self, image, current_head):
        if self.gameOver:
            cvzone.putTextRect(image, "Game Over!", [100,200], scale=3, thickness=3, offset=10)
            cvzone.putTextRect(image, f'Score: {self.score}', [100, 300], scale=3, thickness=3, offset=10)
        else:
            px, py = self.previousHead
            cx, cy = current_head
            self.points.append([cx, cy])
            distance = math.hypot(cx - px, cy - py)
            self.lengths.append(distance)
            self.currentLength += distance
            self.previousHead = cx, cy

            # Maintain snake length
            if self.currentLength > self.allowedLength:
                while self.currentLength > self.allowedLength and len(self.lengths) > 0:
                    self.currentLength -= self.lengths.pop(0)
                    self.points.pop(0)

            # Check if snake eats food
            rx, ry = self.foodPoint
            if rx - self.wFood//2 < cx < rx + self.wFood//2 and ry - self.hFood//2 < cy < ry + self.hFood//2:
                self.random_food_location()
                self.allowedLength += 10
                self.score += 1

            # Draw snake
            for i in range(1, len(self.points)):
                cv2.line(image, tuple(self.points[i-1]), tuple(self.points[i]), (0, 255, 255), 10)
            if self.points:
                cv2.circle(image, self.points[-1], 15, (0, 0, 200), cv2.FILLED)

            # Draw food
            image = cvzone.overlayPNG(image, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))

            # Display score
            cv2.putText(image, f"Score: {self.score}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Collision detection
            # Improved collision detection threshold
            if len(self.points) > 4:
                pts = np.array(self.points[:-50], np.int32).reshape((-1, 1, 2))
                min_distance = cv2.pointPolygonTest(pts, (cx, cy), True)
                if min_distance >= -40:  
                    self.gameOver = True
                    self.points, self.lengths = [], []
                    self.currentLength, self.allowedLength = 0, 150
                    self.previousHead = 0, 0
                    self.random_food_location()

        return image

# Initialize game
game = SnakeGameClass("Donut.png")

# Metrics tracking
prev_time = time.time()
fps_list, latency_list = [], []
total_frames, detected_frames = 0, 0
previous_position, movement_detected_time = None, None

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    total_frames += 1

    # FPS Calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    fps_list.append(fps)
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hand Detection
    hands, img = detector.findHands(img, flipType=False)
    if hands:
        detected_frames += 1
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]

        # Detect movement for latency
        if previous_position:
            movement = abs(pointIndex[0] - previous_position[0]) + abs(pointIndex[1] - previous_position[1])
            if movement > 30 and movement_detected_time is None:
                movement_detected_time = time.time()

        img = game.update(img, pointIndex)

        # Record latency
        if movement_detected_time:
            latency = (time.time() - movement_detected_time) * 1000
            latency_list.append(latency)
            movement_detected_time = None

        previous_position = pointIndex

    cv2.imshow("Snake Game", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        game.gameOver = False
    elif key == ord('q'):
        break

# Results
avg_fps = sum(fps_list) / len(fps_list)
detection_rate = (detected_frames / total_frames) * 100
avg_latency = sum(latency_list) / len(latency_list) if latency_list else 0

print(f"--- Game Metrics ---")
print(f"Average FPS: {avg_fps:.2f}")
print(f"Detection Accuracy: {detection_rate:.2f}%")
print(f"Algorithmic Latency: {avg_latency:.2f} ms")
