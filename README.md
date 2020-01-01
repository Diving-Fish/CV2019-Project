# Project For Computer Vision
-----------------------------------------------------
Team Member:
* Fangyue Dai<br/>
* Qingyuan Liu<br/>
* Lingfei Zhou<br/>
* Shenglong Zhao<br/>

## Project
### Target:
* To detect bottle cap from an input image.
* Label the bottle cap on the input image.
* To detect the state of bottle cap. [front(正面) / back(背面) / side(侧面)]
* To detect the center of bottle cap.
* To provide user interface.

### Extra objectives
* To detect bottle cap in complicated background.
* To detect bottle cap in different photo conditions
* To detect different kinds of bottole cap.
* To detect bottle caps with different colors.

## Implementation

### Canny + ResNet152
Use canny edge detection algorithm to detect bottle caps first. Then use resnet152 model to classify the bottle caps.

1. Canny edge detection algorithm

Use Canny edge detection to detect big edges and then use BFS  searching algorithm to determine the bottle caps. 
However, its detection rate is limited by the complication of the image. If the background is too complicated or bottle 
caps are overlapped, it may not work correctly.

2. Resnet152

Build model with 1000+ images of bottle caps. The max loss of train is 96.55%, and the max loss of evaluation is 98.77%.
We test the model with ten brand new images, each one containing ten bottle caps and the result is fairly well.

### File Structure

```
├── README.md                   
├── api                         // api for ui 
├── model                       // model for classify bottle cap
│   ├── detect.py               // api for classify bottle cap
│   ├── main.py                 // files related to model training
│   └── ...
├── testdata
├── testres                     // folder for saving images of detect outputs
├── ui                          // ui components package
├── util
|   ├── imageUtils              // utils of image, standard_shape, crop_arr
|   └── videoUtils              // utils of video, capture frames...
├── demo.py                     // demo for main.py 
└── main.py                     // main program 
```

## How To Use

For UI developing, you only need to 
1. read the image into np.ndarray
2. call `ImageUtils.standard_shape(img)`, which reshape the image into (800, 600) or (600, 800).<br/>
    It is only a wrapper for `reshape` function
3. call `ImageUtils.detect_bottle_cap(img)`, which return all the points of bottle caps.<br/>
    We have already images in the `testdata`, it will output correct results for each image.<br/>
    The points format is showed as follows:<br/>
    [[(x1, y1), (x2, y2)], [(), ()] ...]<br/>
    Each item is list containing two points of a rectangle.
4. call `ImageUtils.crop_arr(img, points, save=False)`. The function crop the image and return the 
    cropped piece as a list of np.ndarray. You could treat the list as a list of imgs.
5. Use our model to detect each cropped piece in the list returned by `step 4`. <br/>
    You could call `model = load_model`, after that you could call `detect_arr_list(caps, model)`,
    which would return predicts for the list. <br/>
    The predicts format is showed as follows: <br/>
    [{'predict': 1, 'name': 0}, {'predict': 0, 'name': 1}, ...] <br/>
    `predict` is the predict results for the img, `1` is `front`, `0` is `back`, `2` is `side`. You could
    ignore the `name` field which is the name of each img(index of list item if the argument is np.ndarray).

6. After all the steps above, you've got a points list and a predicts list. You could draw rectangles for each
    points and label them with corresponding names. You'd better compute the center point of each bottle cap 
    which should be very easy.
 
 
## CHANGELOG

 01-01-2020

1. Add all backend apis
2. Publish v0.01
    * detect bottle caps in one image
    * get points of bottle caps
    * crop the bottle caps
    * classify the bottle caps


