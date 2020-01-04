from model import load_model, detect_arr_list
from util import ImageUtils
import os
import cv2 as cv
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter import filedialog
import numpy as np

img_in_display = None
img_out_display = None
path = None


def choose_file():
    global img_in_display, path
    path = filedialog.askopenfilename(title='选择文件')
    img_in = Image.open(path).convert('RGB').resize((400, 300))
    tk_img_in = ImageTk.PhotoImage(img_in)
    img_in_display.config(image=tk_img_in)
    img_in_display.image=tk_img_in


def run():
    global img_out_display, path
    classes = ["back", "front", "side"]
    model = load_model()
    print(path)
    img = cv.imread(path)
    img = ImageUtils.standard_shape(img)
    print(1)
    points = ImageUtils.detect_bottle_cap(img)
    print(2)
    caps = ImageUtils.crop_arr(img, points, save=False)
    print(3)
    predicts = detect_arr_list(caps, model)
    print(predicts)
    print(points)
    print(len(caps), len(points))
    for i in range(len(caps)):
        point = points[i]
        cv.rectangle(img, ImageUtils.swap_xy(point[0]), ImageUtils.swap_xy(point[1]), (0, 255, 0), 2)
        p1 = ImageUtils.swap_xy(point[0])
        p2 = ImageUtils.swap_xy(point[1])
        cv.putText(img,
                    classes[predicts[i]["predict"]],
                   (int((p1[0] + p2[0]) / 4), int((p1[1] + p2[1]) / 2)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 0, 0),
                    2)
    img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB)).resize((400, 300))
    tk_img_out = ImageTk.PhotoImage(img)
    img_out_display.config(image=tk_img_out)
    img_out_display.image=tk_img_out


def main():
    global img_in_display, img_out_display

    root = Tk()
    root.title("超级瓶盖分类器")
    root.geometry('800x500')
    root.resizable(0, 0)
    
    Button(root, text ="Choose file", command = choose_file).place(x=10, y=10)
    Button(root, text="Run", command = run).place(x=100, y=10)

    img_in_display = Label(root, width=400, height=300)
    img_in_display.place(x=10, y=100)

    img_out_display = Label(root, width=400, height=300)
    img_out_display.place(x=400, y=100)

    root.mainloop()


if __name__ == '__main__':
    main()

