import numpy as np
from PIL import Image

def min_max(x, mean0=False):
    min = x.min()
    max = x.max()
    result = (x-min)/(max-min+1e-8)
    if mean0 :
        result = result*2 - 1
    return result

def image_from_numpy(output):
    image_list = []
    for i in range(output.shape[0]):
        a = output[i]
        a = np.tile(np.transpose(a, axes=(1,2,0)), (1,1,int(3/a.shape[0])))
        a = min_max(a)*2**8 
        a[a>255] = 255
        a = np.uint8(a)
        a = Image.fromarray(a)
        image_list.append(a)
    return image_list