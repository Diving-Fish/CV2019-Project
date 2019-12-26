import cv2 as cv
import edge_detector


def swap(t):
    return t[1], t[0]


img = cv.imread("cap.jpg")
points = edge_detector.detect_bottle_cap(img)
for point in points:
    cv.rectangle(img, swap(point[0]), swap(point[1]), (0, 255, 0), 2)
cv.imshow("winname", img)
cv.waitKey(0)
