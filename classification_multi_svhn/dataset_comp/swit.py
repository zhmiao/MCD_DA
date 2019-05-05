import os.path
from PIL import Image
from torch.utils.data import Dataset

# Within package imports
from .data_loader import register_dataset_obj, register_data_params
from .data_loader import DatasetParams

@register_data_params('swit')
class SWITParams(DatasetParams):
    
    num_channels = 3
    image_size   = 99
    mean         = 0.5
    std          = 0.5
    num_cls      = 10

@register_dataset_obj('swit')
class SWIT_dataset(Dataset):

    """MultiPIE datasets
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
download=False):

        self.root = root
        self.train = train
        self.dataroot = os.path.join(self.root, 'numbers')

        # if self.train:
        #     self.txt = os.path.join(root, '%s_train.txt'%self.src)
        # else:
        #     self.txt = os.path.join(root, '%s_test.txt'%self.src)

        self.txt = os.path.join(self.root, 'swit_all.txt')

        self.img_path = []
        self.labels = []
        self.transform = transform

        with open(self.txt) as f:
            for line in f:
                self.img_path.append(os.path.join(self.dataroot, line.split()[0]))
                self.labels.append(int(line.split()[1]))


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        target = self.labels[index]
        
        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)

        return image, target
