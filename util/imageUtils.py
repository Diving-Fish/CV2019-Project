from typing import List

import cv2 as cv
import os
import logging
import uuid
import numpy as np


def cvt_groups(group):
    """
    cvt_groups(group) -> points

    :param group: point groups.
    :return points: 2-vertex array of the rectangle.
    """
    minx = group[0][0]
    miny = group[0][1]
    maxx = minx
    maxy = miny
    for point in group:
        if point[0] < minx:
            minx = point[0]
        if point[1] < miny:
            miny = point[1]
        if point[0] > maxx:
            maxx = point[0]
        if point[1] > maxy:
            maxy = point[1]
    return [(minx, miny), (maxx, maxy)]


def set_equal(equal_labels, a, b):
    if a == b:
        return
    for label in equal_labels:
        if a in label or b in label:
            if a not in label:
                label.append(a)
            if b not in label:
                label.append(b)
            return
    equal_labels.append([a, b])


class ImageUtils:
    @staticmethod
    def swap_xy(point):
        return point[1], point[0]

    @staticmethod
    def detect_bottle_cap(img, min_size=400, max_spacing=8):
        """
        detect_bottle_cap(img) -> points

        :param img: 8-bit input image.
        :param min_size: the min_size of a rectangle. the default is 400 pixels. (20 x 20)
        :param max_spacing: the max_spacing of two edges detected.
                         Two edges which spacing < max_spacing will be recognized as one bottle cap.
        :return points: 2-point array of the rectangle containing the bottle cap.
        """
        image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(image, 15, 150)
        equal_labels = []
        labels = np.zeros(edges.shape, int)
        label = 1
        _label = 1
        # 2-Pass
        # First:
        for x in range(edges.shape[0]):
            for y in range(edges.shape[1]):
                if edges[x][y] == 255:
                    flag = False
                    for _x in range(x - max_spacing, x + max_spacing + 1, 1):
                        for _y in range(y - max_spacing, y + max_spacing + 1, 1):
                            if _x < 0 or _y < 0 or _x >= edges.shape[0] or _y >= edges.shape[1]:
                                continue
                            cur_label = labels[_x][_y]
                            if cur_label != 0:
                                labels[x][y] = cur_label
                                if cur_label != label:
                                    for __x in range(x - max_spacing, x + max_spacing + 1, 1):
                                        for __y in range(y - max_spacing, y + max_spacing + 1, 1):
                                            if not (__x < 0 or __y < 0 or __x >= edges.shape[0] or __y >= edges.shape[
                                                1]) \
                                                    and labels[__x][__y] != 0:
                                                set_equal(equal_labels, labels[__x][__y], cur_label)
                                    label = cur_label
                                flag = True
                                break
                    if not flag:
                        labels[x][y] = _label
                        label = _label
                        _label += 1
        # Second:
        d = {}
        for x in range(edges.shape[0]):
            for y in range(edges.shape[1]):
                if labels[x][y] != 0:
                    val = 0
                    flag = False
                    for label in equal_labels:
                        if labels[x][y] in label:
                            flag = True
                            minv = label[0]
                            for v in label:
                                if v < minv:
                                    minv = v
                            labels[x][y] = minv
                            val = minv
                    if not flag:
                        val = labels[x][y]
                    try:
                        d[val].append((x, y))
                    except KeyError:
                        d[val] = [(x, y)]
        points = []
        for key in d:
            point = cvt_groups(d[key])
            flag = True
            for p in points:  # detect containing
                if p[0][0] <= point[0][0] and p[0][1] <= point[0][1] \
                        and p[1][0] >= point[1][0] and p[1][1] >= point[1][1]:
                    flag = False
                    break
            if flag and (point[1][0] - point[0][0]) * (point[1][1] - point[0][1]) >= min_size:
                points.append(point)
        return points

    @staticmethod
    def crop_file(srcfile: str, points, save=True, savepath="") -> List[np.ndarray]:
        if not os.path.exists(srcfile):
            logging.error("File %s not exists" % srcfile)
            exit(1)
        img = cv.imread(srcfile)
        return ImageUtils.crop_arr(img, points, save, savepath)

    @staticmethod
    def crop_arr(img: np.ndarray, points, save=True, savepath="") -> List[np.ndarray]:
        res: List = []
        for point in points:
            if point[1][0] - point[0][0] < 50 \
                    or point[1][1] - point[0][1] < 50:  # threshold
                continue
            if save:
                cv.imwrite("{}\\{}.png".format(savepath, str(uuid.uuid1())),
                           img[point[0][0]:point[1][0], point[0][1]:point[1][1]], (cv.IMWRITE_PNG_COMPRESSION, 1))
            res.append(img[point[0][0]:point[1][0], point[0][1]:point[1][1]])
        return res

    @staticmethod
    def standard_shape(img: np.ndarray) -> np.ndarray:
        shape = img.shape
        height = shape[0]
        width = shape[1]

        if height > width:
            #  resize (width, height)
            return cv.resize(img, (600, 800), interpolation=cv.INTER_CUBIC)
        else:
            #  resize (width, height)
            return cv.resize(img, (800, 600), interpolation=cv.INTER_CUBIC)


def main():
    path = "D:\\courses\\computer-vision\\project\\object-detection\\resource\\origin-images"
    files = os.listdir(path)

    for file in files:
        img = ImageUtils.standard_shape(cv.imread("".join([path, "\\", file])))
        points = ImageUtils.detect_bottle_cap(img)
        print(points)
        ImageUtils.crop_arr(img, points,
                            savepath="D:\\courses\\computer-vision\\project\\object-detection\\resource\\classes")


if __name__ == '__main__':
    main()
