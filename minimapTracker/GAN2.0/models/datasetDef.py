import os, os.path
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.io import read_image

class mmPortraits(Dataset):
    def __init__(self, img_dir, transform=None, train=True, realValue = 1.0):
        self.img_dir = img_dir
        if train:
            self.img_dir += "train/"
        else:
            self.img_dir += "test/"
            
        self.transform = transform
        self.img_paths = os.listdir(self.img_dir)
        self.num_imgs = len(os.listdir(self.img_dir))
        self.label = realValue

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(self.img_dir + img_path).float()
        if self.transform:
            image = self.transform(image)
        sample = {"image": image, "label": self.label}
        return sample
