import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import ToTensor
import os
from PIL import Image

#Data already sorted 80% train and 20% val

class MyDataset(Dataset):
    def __init__(self, data_dir, is_train=True, transform=None):
        super().__init__()
        self.is_train = is_train
        self.transform = transform  
        self.data = self.load_data(data_dir) 

    
    def load_data(self, data_dir):
        data = [] #empty list (will contain tuples)
        if self.is_train:
            data_subdir = 'train'
        else:
            data_subdir = 'val'
        data_dir = os.path.join(data_dir, data_subdir)
        
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if not os.path.isdir(label_dir):
                continue
            label_id = label #if it is train or val
            for filename in os.listdir(label_dir):
                data.append((os.path.join(label_dir, filename), label))
        return data
            
    def __getitem__(self, idx):
        ###################################
        # get the sample with index idx, it is self.data[idx] if all data is in memory, 
        # otherwise acquire the data associated with the reference self.data[idx]
        image_path, label = self.data[idx]
        image = Image.open(image_path)
        ###################################
        if self.transform:
            image = self.transform(image)                  # modify raw data if necessary
        return image, label

    def __len__(self):
        return len(self.data)


def demo1(dl):
    ''' Using iterators and next with an iterable such as a Dataset or a DataLoader, 
        a StopIteration exception may be thrown if no further data is available.
    ''' 
    it = iter(dl)
    sample = next(it)
    print(sample)
    sample = next(it)
    print(sample)

    
def demo2(dl):
    ''' Using a while loop with an iterable such as a Dataset or a DataLoader
    ''' 
    for sample in dl:
        print(sample)


if __name__ == '__main__':
    # Test your dataset first in a pure Python/Numpy environment, you do not need to know
    # much about Torch for it
    dir = "C:\\Users\\taube\\OneDrive\\Desktop\\Uni - LaptopAsus\\WS23-24\\ML_Projekt\\Projekt_ML\\traffic_light_data"
    train_ds = MyDataset(data_dir=dir,is_train=True)
    demo1(train_ds)
    demo2(train_ds)
    
