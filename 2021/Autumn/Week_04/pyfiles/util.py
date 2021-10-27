import numpy as np
from PIL import Image
import glob
import torch
import torchvision.transforms as transforms

transform_original = transforms.Compose([
    transforms.ToTensor(),
])

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
    
class Dataset_Fashion_MNIST(torch.utils.data.Dataset):
    def __init__(self, root, classes, mode="train", transform=None, balance=[0.7,0.15,0.15], each_data_num=10000000):
        
        self.transform = transform
        self.images = []
        self.labels = []

        images = {} 
        labels = {}
        
        for cl in classes:
            
            # get data which is affiliated with a selected class
            path_list = glob.glob(root + f"{cl}/*") 
            path_list.sort()
            path_list = path_list[:each_data_num]
            
            # define the amount of training, validation, and test dataset
            train_num = int(balance[0]*len(path_list))
            val_num = int(balance[1]*len(path_list))
            test_num = int(balance[2]*len(path_list))
            
            # get data which is affiliated with a selected mode
            if mode=="train":
                path_list = path_list[:train_num]
            elif mode=="val":
                path_list = path_list[train_num:train_num+val_num]
            elif mode=="test":
                path_list = path_list[-test_num:]
                
            images[str(cl)] = path_list
            labels[str(cl)] = [cl]*len(path_list)
            
        # combine them together
        for label in classes:
            for image, label in zip(images[str(label)], labels[str(label)]):
                self.images.append(image)
                self.labels.append(label)

    def __getitem__(self, index):
        
        image = self.images[index]
        label = self.labels[index]
        
        with open(image, 'rb') as f:
            image = Image.open(f)
            image = image.convert("L")
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
    
    def __len__(self):
        return len(self.images)
        
############ https://www.kaggle.com/grfiv4/plot-a-confusion-matrix #############
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()