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


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
    
    
def generate_edge_filter(vertical=True, device="cpu", nch=3):
    edge_filter = nn.Conv2d(nch, nch, 3, 1, 1)
    parameters = edge_filter.state_dict()
    bias = torch.zeros(parameters["bias"].shape)
    parameters["bias"] = bias
    weight = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    if not vertical:
        weight = weight.T
    weight = np.tile(np.reshape(weight, (1,1,3,3)), (nch,nch,1,1))
    parameters["weight"] = torch.tensor(weight)
    edge_filter.load_state_dict(parameters)
    return edge_filter.to(device)


def generate_fc_edge_filter(vertical=True, nch=3, size=32):
    weight = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    if not vertical:
        weight = weight.T
    fc_vertical_filter = nn.Linear(size**2*nch, size**2*nch)
    parameters = fc_vertical_filter.state_dict()
    bias = torch.zeros(parameters["bias"].shape)
    parameters["bias"] = bias
    for k in range(nch):
        for i in range(size):
            for j in range(size):
                weights = np.zeros((size+2, size+2))
                weights[i:i+nch, j:j+nch] = weight
                weights = np.tile(np.resize(weights[1:-1,1:-1], (1, size, size)), (nch,1,1))
                parameters["weight"][k*size**2+i*size+j] = torch.tensor(weights).view(-1)
    fc_vertical_filter.load_state_dict(parameters)
    return fc_vertical_filter


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