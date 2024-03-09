from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import random
import torch
from PIL import Image
import numpy as np

class SiameseDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        should_get_same_class = random.randint(0, 1)  # Ensure roughly equal chance of same class or not
        if should_get_same_class:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0]).convert("RGB")
        img1 = Image.open(img1_tuple[0]).convert("RGB")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img0_tuple[1] != img1_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

# Use the ImageFolder to load data
# dataset = ImageFolder(root='./dataset')
# siamese_dataset = SiameseDataset(imageFolderDataset=dataset,
#                                  transform=transforms.Compose([transforms.Resize((512, 512)),
#                                                                transforms.ToTensor()]))

# print('The dataset is ' + str(type(siamese_dataset)))
