import cv2 as cv
import os
import logging
import uuid


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


def search(point, img, groups):
    """
        search(point, img, group) -> none

        :param point: the start point (x, y) for bfs searching.
        :param img: the source image after edge detection.
        :param groups: point group searched.::
        """
    for group in groups:
        if point in group:
            return
    if img[point[0]][point[1]] != 255:
        return
    queue = [point]
    searched = []
    while queue:
        p = queue.pop(0)
        for x in range(p[0] - 5, p[0] + 6, 1):
            for y in range(p[1] - 5, p[1] + 6, 1):
                if x < 0 or y < 0 or x >= img.shape[0] or y >= img.shape[1]:
                    continue
                if img[x][y] == 255 and (x, y) not in searched and (x, y) not in queue:
                    queue.append((x, y))
        if p not in searched:
            searched.append(p)
    groups.append(searched)


class ImageUtils:
    @staticmethod
    def detect_bottle_cap(img):
        """
        detect_bottle_cap(img) -> points

        :param img: 8-bit input image.
        :return points: 2-point array of the rectangle containing the bottle cap.
        """
        image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(image, 50, 150)
        groups = []
        for x in range(edges.shape[0]):
            for y in range(edges.shape[1]):
                if edges[x][y] == 255:
                    search((x, y), edges, groups)
        points = []
        for group in groups:
            point = cvt_groups(group)
            flag = True
            for p in points:  # detect containing
                if p[0][0] <= point[0][0] and p[0][1] <= point[0][1] \
                        and p[1][0] >= point[1][0] and p[1][1] >= point[1][1]:
                    flag = False
                    break
            if flag:
                points.append(point)
        return points

    @staticmethod
    def crop_caps(srcfile: str, savepath: str):
        if not os.path.exists(srcfile):
            logging.error("File %s not exists" % srcfile)
            exit(1)
        img = cv.imread(srcfile)
        points = ImageUtils.detect_bottle_cap(img)
        for point in points:
            cv.imwrite("%s\\%s.jpg" % (savepath, str(uuid.uuid1())),
                       img[point[0][0]:point[1][0], point[0][1]:point[1][1]])


def main():
    ImageUtils.crop_caps("cap.jpg",
                         "D:\\courses\\computer-vision\\project\\object-detection\\resource\\classes")


if __name__ == '__main__':
    main()
