import numpy as np
import torch as t
import os
import platform
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import logging

logging.getLogger().setLevel(logging.ERROR)


class BottleCap(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        #  Acquire all images path
        # - root/
        #   - front/
        #       - file1.jpg
        #       ...
        #   - back/
        #   - side/
        #
        folders = os.listdir(root)
        imgs = []
        for folder in folders:
            files = os.listdir(os.path.join(root, folder))
            if test:
                imgs.extend([os.path.join(root, folder, file) for file in files])
            elif train:
                imgs.extend([os.path.join(root, folder, file) for file in files][:int(0.7*len(files))])
            else:
                imgs.extend([os.path.join(root, folder, file) for file in files][int(0.7*len(files)):])
        self.imgs = imgs

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[.229, .224, .225])
            if test or not train:
                self.transforms = T.Compose([
                    T.Resize(224, interpolation=Image.BICUBIC),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(256, interpolation=Image.BICUBIC),
                    T.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        img_path: str = self.imgs[index]
        label: int
        path_sep = "\\" if "Windows" in platform.system() else "/"
        classes = img_path.split(path_sep)[-2]
        if "front" in classes:
            label = 1
        elif "back" in classes:
            label = 2
        elif "side" in classes:
            label = 0
        else:
            label = -1
            logging.error("Unknown classes %s" % classes)
        pil_img = Image.open(img_path)
        arr = np.asarray(pil_img)
        dt = t.from_numpy(arr)
        return dt, label

    def __len__(self):
        return len(self.imgs)


def main():
    train_dataset = BottleCap("..\\data\\bottlecap")
    img, label = train_dataset[300]
    print(img.size(), img.float().mean(), label)


if __name__ == '__main__':
    main()
