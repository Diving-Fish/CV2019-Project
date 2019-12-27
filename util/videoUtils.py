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
        i = 0   # step count
        while ok:
            if i < step:
                i = i + 1
                continue
            else:
                ok, frame = cap.read()
                i = 0
                cnt = cnt + 1
            if not ok:
                break
            else:
                logging.info("Saving image %s\\%d.jpg" % (savepath, cnt))
                cv.imwrite("%s\\%d.jpg" % (savepath, cnt), frame)


def main():
    VideoUtils.capture_frames(30, "D:\\courses\\computer-vision\\project\\small-vedios\\VID_20191227_133259.mp4",
                              "D:\\courses\\computer-vision\\project\\object-detection\\resource")


if __name__ == '__main__':
    main()
