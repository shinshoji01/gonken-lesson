import numpy as np
from PIL import Image
import glob
import torch
import torchvision.transforms as transforms

colors =  [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#8c564b',  # chestnut brown
    '#bcbd22',  # curry yellow-green
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#17becf']

def cuda2numpy(x):
    return x.detach().to("cpu").numpy()

def min_max(x, mean0=False):
    min = x.min()
    max = x.max()
    result = (x-min)/(max-min+1e-8)
    if mean0 :
        result = result*2 - 1
    return result

def image_from_output(output):
    """
    convert torch.Tensor into PIL image

    ------------
    Parameters
    ------------

    output : torch.Tensor, shape=(sample_num, channel, length, width)
        either cuda or cpu tensor
        
    ------------
    Returns
    ------------

    image_list : list
        list includes PIL images

    ------------

    """
    if len(output.shape)==3:
        output = output.unsqueeze(0)
        
    image_list = []
    output = cuda2numpy(output)
    for i in range(output.shape[0]):
        a = output[i]
        a = np.tile(np.transpose(a, axes=(1,2,0)), (1,1,int(3/a.shape[0])))
        a = min_max(a)*2**8 
        a[a>255] = 255
        a = np.uint8(a)
        a = Image.fromarray(a)
        image_list.append(a)
    return image_list

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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('linear') != -1:        
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('batchnorm') != -1:     
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    return

class MinMax(object):
    def __init__(self, mean0=True):
        self.mean0 = mean0
        pass

    def __call__(self, img):
        return min_max(img, self.mean0)

    def __repr__(self):
        return self.__class__.__name__
    
def do_test_VAE(net, testloader, device="cuda", mode="train", losses_mean=True):
    
    if mode=="train":
        net.train()
    elif mode=="eval":
        net.eval()
    else:
        return None
    
    labels = np.array([])
    losses = []
    with torch.no_grad():
        for itr, data in enumerate(testloader):
            images = data[0].to(device)
            label = data[1].to(device)
            output, z, loss = net(images, True)
            labels = np.append(labels, label.to("cpu").detach().numpy())
            losses.append(loss.to("cpu").detach().numpy())
            if itr==0:
                latents = z.to("cpu").detach().numpy()
                inputs = images.to("cpu").detach().numpy()
                outputs = output.to("cpu").detach().numpy()
            else:
                latents = np.concatenate([latents, z.to("cpu").detach().numpy()], axis=0)
                inputs = np.concatenate([inputs, images.to("cpu").detach().numpy()], axis=0)
                outputs = np.concatenate([outputs, output.to("cpu").detach().numpy()], axis=0)
                
    if losses_mean:
        losses = np.mean(losses)
    return labels, inputs, outputs, losses, latents