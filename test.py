import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torchsummary import summary
from torch.optim import SGD, Adam
import numpy as np, cv2
import matplotlib.pyplot as plt
from glob import glob
from imgaug import augmenters as iaa

tfm = iaa.Sequential(iaa.Resize(28))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class X0(Dataset):
    def __init__(self, folder):
        self.files = glob(folder)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        f = self.files[idx]
        im = tfm.augment_image(cv2.imread(f)[:,:,0])
        im = im[None]
        cl = f.split('/')[-1].split('@')[0]=='x'
        return torch.tensor(1-im/255).to(device).float(), torch.tensor([cl]).float().to(device)

data = X0("all/*")
R, C = 7, 7
fig, ax = plt.subplots(R, C, figsize=(5,5))
for label_class, plot_row in enumerate(ax):
    for plot_cell in plot_row:
        plot_cell.grid(False);plot_cell.axis('off')
        idx = np.random.choice(1000)
        im, label = data[idx]
        print()
        plot_cell.imshow(im[0].cpu(), cmap='gray')
plt.tight_layout()
plt.show()