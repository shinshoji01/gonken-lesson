import torch
import torchvision.transforms as transforms
import glob
from PIL import Image

from util import MinMax

transform = {}
transform["train"] = transforms.Compose([
    transforms.Resize(16),
    transforms.ToTensor(),
    MinMax(mean0=True),
])
    
transform["test"] = transforms.Compose([
    transforms.Resize(16),
    transforms.ToTensor(),
    MinMax(mean0=True),
])

class Dataset_MNIST(torch.utils.data.Dataset):
    def __init__(self, root, classes, mode="train", transform=None, balance=[0.7,0.15,0.15], each_datanum=100):
        
        self.transform = transform
        self.images = []
        self.labels = []

        images = {} 
        labels = {}
        
        for cl in classes:
            
            # get data which is affiliated with a selected class
            path_list = glob.glob(root + f"{cl}/*") 
            path_list.sort()
            path_list = path_list[:each_datanum]
            
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