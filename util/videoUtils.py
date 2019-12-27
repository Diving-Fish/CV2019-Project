import cv2 as cv
import os
import logging

logging.getLogger().setLevel(logging.INFO)


class VideoUtils:
    @staticmethod
    def capture_frames(step: int, srcfile: str, savepath: str) -> int:
        if not os.path.exists(srcfile):
            logging.error("File %s does not exists. Please check the filename" % srcfile)
            exit(1)

        cap = cv.VideoCapture(srcfile)
        if not cap.isOpened():
            logging.error("Open video failed")
            return -1

        ok = True
        cnt = 0  # read frame count
        i = 0  # step count
        videoname = srcfile[srcfile.rfind("\\"):]
        while ok:
            ok, frame = cap.read()
            if not ok:
                break
            if i < step:
                i = i + 1
                continue
            else:
                i = 0
                cnt = cnt + 1
            logging.info("Saving image %s\\%s%d.jpg" % (savepath, videoname, cnt))
            cv.imwrite("%s\\%s%d.jpg" % (savepath, videoname, cnt), frame)


def main():
    path = "D:\\courses\\computer-vision\\project\\object-detection\\resource\\small-videos"
    files = os.listdir(path)
    for file in files:
        VideoUtils.capture_frames(10, "".join([path, "\\", file]),
                                  "D:\\courses\\computer-vision\\project\\object-detection\\resource\\origin-images")


if __name__ == '__main__':
    main()
