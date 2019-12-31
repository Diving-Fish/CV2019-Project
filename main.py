from model import *
from util import ImageUtils
import os
import cv2 as cv


def main():
    imgs = os.listdir("testdata")
    classes = ["back", "front", "side"]
    model = load_model()
    for imgname in imgs:
        img = cv.imread(os.path.join("testdata", imgname))
        img = ImageUtils.standard_shape(img)
        points = ImageUtils.detect_bottle_cap(img)
        caps = ImageUtils.crop_arr(img, points, save=False)
        predicts = detect_arr_list(caps, model)
        print(predicts)
        print(points)
        print(len(caps), len(points))
        for i in range(len(caps)):
            point = points[i]
            cv.rectangle(img, ImageUtils.swap_xy(point[0]), ImageUtils.swap_xy(point[1]), (0, 255, 0), 2)
            cv.putText(img,
                       classes[predicts[i]["predict"]],
                       ImageUtils.swap_xy(point[0]),
                       cv.FONT_HERSHEY_SIMPLEX,
                       1.0,
                       (255, 0, 0),
                       2)
        cv.imwrite("{}\\{}".format("testres", imgname), img)

# img = cv.imread("1.jpg")
# img = cv.resize(img, (800, 600))
# points = edge_detector.detect_bottle_cap(img)
# for point in points:
#     cv.rectangle(img, swap(point[0]), swap(point[1]), (0, 255, 0), 2)
# cv.imshow("winname", img)
# cv.waitKey(0)


if __name__ == '__main__':
    main()

