from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np

class Horse2ZebraDataset(Dataset):
    def __init__(self, rootZebra, rootHorse, transform=None):
        self.rootZebra = rootZebra
        self.rootHorse = rootHorse
        self.transform = transform

        self.zebraImages = os.listdir(rootZebra)
        self.horseImages = os.listdir(rootHorse)

        self.lenghtDataset = max(len(self.zebraImages), len(self.horseImages))

        self.zebraLen = len(self.zebraImages)
        self.horseLen = len(self.horseImages)


    def __getitem__(self, index):
        zebraImg = self.zebraImages[index % self.zebraLen]
        horseImg = self.horseImages[index % self.horseLen]

        zebraPath = os.path.join(self.rootZebra, zebraImg)
        horsePath = os.path.join(self.rootHorse, horseImg)



        zebraImg = np.array(Image.open(zebraPath).convert("RGB"))
        horseImg = np.array(Image.open(horsePath).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image = zebraImg, image0 = horseImg)

            zebraImg = augmentations["image"]       #image e image0 sono i nomi dati nel config file
            horseImg = augmentations["image0"]

        return zebraImg, horseImg

    def __len__(self):
        return self.lenghtDataset


