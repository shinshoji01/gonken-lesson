import warnings
warnings.filterwarnings("ignore")
from PIL import Image 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import cv2


def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min+1e-8)
    return result


def image_from_output(output):
    image_list = []
    output = output.detach().to("cpu").numpy()
    for i in range(output.shape[0]):
        a = output[i]
        a = np.tile(np.transpose(a, axes=(1,2,0)), (1,1,int(3/a.shape[0])))
        a = min_max(a)*2**8 
        a[a>255] = 255
        a = np.uint8(a)
        a = Image.fromarray(a)
        image_list.append(a)
    return image_list


### https://qiita.com/derodero24/items/f22c22b22451609908ee ###
def pil2cv(image):
    ''' PIL -> OpenCV '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image
def cv2pil(image):
    ''' OpenCV -> PIL '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def move_image(img, transform, mi=2):
    image = image_from_output(img)[0]
    a = pil2cv(image)
    mxy = [[-mi, 0], [mi, 0], [0, -mi], [0, mi]]
    new_list = img
    for i in range(len(mxy)):
        M = np.float32([[1,0,mxy[i][0]],[0,1,mxy[i][1]]])
        b = cv2.warpAffine(a, M, (32, 32))
        new_image = cv2pil(b)
        new_image = transform(new_image)[0:1].view(1, 1, 32, 32)
        new_list = torch.cat([new_list, new_image], dim=0)
    return new_list


def cuda_to_numpy(x):
    return x.detach().to("cpu").numpy()


def weights_init(m):
    # 重みの初期化
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('linear') != -1:        # 全結合層の場合
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('batchnorm') != -1:     # バッチノーマライゼーションの場合
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)