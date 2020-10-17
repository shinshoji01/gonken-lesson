# OpenCV

## Step1. What is OpenCV

**OpenCV** (*Open Source Computer Vision Library*) is a [library of programming functions](https://en.wikipedia.org/wiki/Library_(computing)) mainly aimed at real-time [computer vision](https://en.wikipedia.org/wiki/Computer_vision). 

Originally developed by [Intel](https://en.wikipedia.org/wiki/Intel_Corporation), it was later supported by [Willow Garage](https://en.wikipedia.org/wiki/Willow_Garage) then Itseez (which was later acquired by Intel). 

### Applications

OpenCV's application areas include:

- 2D and 3D feature toolkits

- [Egomotion](https://en.wikipedia.org/wiki/Egomotion) estimation

- [Facial recognition system](https://en.wikipedia.org/wiki/Facial_recognition_system)

- [Gesture recognition](https://en.wikipedia.org/wiki/Gesture_recognition)

- [Human–computer interaction](https://en.wikipedia.org/wiki/Human–computer_interaction) (HCI)

- [Mobile robotics](https://en.wikipedia.org/wiki/Mobile_robotics)

- Motion understanding

- Object identification

- [Segmentation](https://en.wikipedia.org/wiki/Segmentation_(image_processing)) and recognition

- [Stereopsis](https://en.wikipedia.org/wiki/Stereopsis) stereo vision: depth perception from 2 cameras

- [Structure from motion](https://en.wikipedia.org/wiki/Structure_from_motion) (SFM)

- [Motion tracking](https://en.wikipedia.org/wiki/Video_tracking)

- [Augmented reality](https://en.wikipedia.org/wiki/Augmented_reality)

  To support some of the above areas, OpenCV includes a statistical [machine learning](https://en.wikipedia.org/wiki/Machine_learning) library that contains:

- [Boosting](https://en.wikipedia.org/wiki/Boosting_(meta-algorithm))

- [Decision tree learning](https://en.wikipedia.org/wiki/Decision_tree_learning)

- [Gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) trees

- [Expectation-maximization algorithm](https://en.wikipedia.org/wiki/Expectation-maximization_algorithm)

- [k-nearest neighbor algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm)

- [Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

- [Artificial neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network)

- [Random forest](https://en.wikipedia.org/wiki/Random_forest)

- [Support vector machine](https://en.wikipedia.org/wiki/Support_vector_machine) (SVM)

- [Deep neural networks](https://en.wikipedia.org/wiki/Deep_neural_network) (DNN)

### Programming language

**OpenCV is written in [C++](https://en.wikipedia.org/wiki/C%2B%2B) and its primary interface is in C++.**

All of the new developments and algorithms appear in the C++ interface.

**There are bindings in [Python](https://en.wikipedia.org/wiki/Python_(programming_language)), [Java](https://en.wikipedia.org/wiki/Java_(programming_language)) and [MATLAB](https://en.wikipedia.org/wiki/MATLAB)/[OCTAVE](https://en.wikipedia.org/wiki/GNU_Octave).** 

The API for these interfaces can be found in the online documentation.

### OS support

**OpenCV runs on the following desktop operating systems: [Windows](https://en.wikipedia.org/wiki/Microsoft_Windows), [Linux](https://en.wikipedia.org/wiki/Linux), [macOS](https://en.wikipedia.org/wiki/MacOS), [FreeBSD](https://en.wikipedia.org/wiki/FreeBSD), [NetBSD](https://en.wikipedia.org/wiki/NetBSD), [OpenBSD](https://en.wikipedia.org/wiki/OpenBSD).** 

OpenCV runs on the following mobile operating systems: [Android](https://en.wikipedia.org/wiki/Android_(operating_system)), [iOS](https://en.wikipedia.org/wiki/IOS), [Maemo](https://en.wikipedia.org/wiki/Maemo),[[16\]](https://en.wikipedia.org/wiki/OpenCV#cite_note-Maemo_Port-16) [BlackBerry 10](https://en.wikipedia.org/wiki/BlackBerry_10).[[17\]](https://en.wikipedia.org/wiki/OpenCV#cite_note-17) 

The user can get official releases from [SourceForge](https://en.wikipedia.org/wiki/SourceForge) or take the latest sources from [GitHub](https://en.wikipedia.org/wiki/GitHub). 

OpenCV uses [CMake](https://en.wikipedia.org/wiki/CMake).

## Step2. Installation

[installation documentation](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)

[installation documentation from PyPI](https://pypi.org/project/opencv-python/)

> #### What is PyPI
>
> [Python Package Index](https://pypi.org/)
>
> プログラミング言語Pythonの、サードパーティーソフトウェアリポジトリのこと。
>
> 多くのPythonパッケージはPyPI上にある。ちなみにPythonのパッケージ管理システムがpip。
>
> The Python Package Index (abbreviated as PyPI) is a third-party software rare for the export language Python. Think of it as a software template for software here. Download and its dependencies from PyPI and install the software. The first spring is the Python Cheese Shop.
>

#### Packages for standard desktop environments (Windows, macOS, almost any GNU/Linux distribution)

- Option 1 - Main modules package: `pip install opencv-python`
- Option 2 - Full package (contains both main modules and contrib/extra modules): `pip install opencv-contrib-python` (check contrib/extra modules listing from [OpenCV documentation](https://docs.opencv.org/master/))

## Step3. Let's use OpenCV

### Let's make 2 Python files

```bash
$ mkdir ./Week_03_OpenCV # "mkdir" is a makig directory command 
$ cd Week_03_OpenCV # "cd" is a changing directory command
$ touch download_cat.py # "touch" is a making file command
$ touch opencv_demo.py
$ atom download_cat.py pillow_demo.py # If you use atom editor
```

```python
# ---download_cat.py--- #
# for downloading cat picture

import os
import pprint
import time
import urllib.error
import urllib.request

def download_file(url, dst_path):
    try:
        with urllib.request.urlopen(url) as web_file:
            data = web_file.read()
            with open(dst_path, mode='wb') as local_file:
                local_file.write(data)
    except urllib.error.URLError as e:
        print(e)
        
url = 'https://1.bp.blogspot.com/-jlHonWZdPp0/Xq5vQuVPQrI/AAAAAAABYtI/S0mjN1WK-wEJBBSS2M6xTEhEmVjM5mUwwCEwYBhgL/s1600/shigoto_zaitaku_cat_man.png'
dst_path = 'cat.png'
download_file(url, dst_path)
```

```python
# ---opencv_demo.py--- #
# for displaying cat picture

import cv2
import numpy as np

img = cv2.imread('cat.png')
cv2.imshow("color",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Step4. Let's play OpenCV freely

```python
# Let's add this code to opencv_demo.py

img = cv2.imread('cat.png')
# 1
print(type(im))
# 2 row*col*color
print(im.shape)
# 3
print(im.dtype)
# 4 ndarray's color is BGR, not RGB
# [row, col, (B, G, R)]
img[:, :, (0, 1)] = 0
cv2.imwrite('red_cat.png', img)
# 5 gray scale
im_gray = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)
print(im_gray.shape)
cv2.imwrite('gray_cat.png', im_gray)
```

## Step5. What is the difference between Pillow and OpenCV

- Pillow : Image processing library
  - When you want to perform basic processing such as resizing and trimming
- NumPy ( + Pillow or OpenCV) : Scientific calculation library
  - When you want to perform processing such as arithmetic operations for each pixel value
    - Read the image as a NumPy array
    - Image files cannot be read by NumPy alone, so use it with Pillow, OpenCV, etc.
- OpenCV : Computer vision library
  - When you want to perform computer vision processing such as face recognition in addition to the processing of ndarray

## Reference

- [OpenCV -wikipedia-](https://en.wikipedia.org/wiki/OpenCV)
- [OpenCV -official documentation-](https://opencv.org/)
- [Python, OpenCVで画像ファイルの読み込み、保存（imread, imwrite）](https://note.nkmk.me/python-opencv-imread-imwrite/)
- [Pythonで画像処理: Pillow, NumPy, OpenCVの違いと使い分け](https://note.nkmk.me/python-image-processing-pillow-numpy-opencv/)