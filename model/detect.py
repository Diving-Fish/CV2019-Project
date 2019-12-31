from typing import List, Dict

import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from torch.autograd import Variable as V
import os
import logging
import torch as t
import torch.nn as nn
import torchvision.transforms as T


def detect_one_arr(img: np.ndarray, model: nn.Module) -> int:
    trans = T.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.fromarray(img)
    inputs = trans(img).unsqueeze(0)
    model.eval()
    inputs = V(inputs.cuda())
    predict = model(inputs)
    probability = t.nn.functional.softmax(predict, dim=1)
    return t.max(probability, 1)[1].cpu().numpy()[0]


def detect_arr_list(arrlist: List[np.ndarray], model: nn.Module) -> List[Dict]:
    return [{"predict": detect_one_arr(arrlist[i], model), "name": i} for i in range(len(arrlist))]


def detect_one(srcfile: str, model: nn.Module) -> int:
    trans = T.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if not os.path.exists(srcfile):
        logging.error("File {} not exists".format(srcfile))
        exit(1)
    img = Image.open(srcfile)
    inputs = trans(img).unsqueeze(0)
    model.eval()
    inputs = V(inputs.cuda())
    predict = model(inputs)
    probability = t.nn.functional.softmax(predict, dim=1)
    return t.max(probability, 1)[1].cpu().numpy()[0]


def detect_all(folder: str, model: nn.Module) -> List[Dict]:
    if not os.path.exists(folder):
        logging.error("Folder {} not exists".format(folder))
        exit(1)
    files = os.listdir(folder)
    return [{"predict": detect_one(os.path.join(folder, file), model), "name": file} for file in files]


def load_model(path=".\\model\\model.pth"):
    model = t.load(path)
    return model


def main():
    folder = "C:\\Users\\ALIENWARE\\Desktop\\test"
    model = t.load("resnet152-bottlecap_v0.91.pth")
    for entry in detect_all(folder, model):
        if entry["predict"] != 1:
            print(entry)


if __name__ == '__main__':
    main()
