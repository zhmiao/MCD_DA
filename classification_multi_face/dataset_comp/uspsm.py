import gzip
import os.path

from urllib.parse import urljoin
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import datasets

# Within package imports
from .data_loader import register_dataset_obj, register_data_params
from . import util
from .data_loader import DatasetParams

@register_data_params('uspsm')
class USPSMParams(DatasetParams):
    
    num_channels = 3
    image_size   = 16
    mean         = 0.5
    std          = 0.5
    num_cls      = 10

@register_dataset_obj('uspsm')
class USPSM(data.Dataset):

    """USPSM
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
download=False):

        self.root = root
        self.train = train
        if self.train:
            self.txt = os.path.join(root, 'uspsm_train.txt')
        else:
            self.txt = os.path.join(root, 'uspsm_test.txt')

        print('Use %s'%self.txt)

        # import pdb
        # pdb.set_trace()

        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(self.txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label
        
        
    
