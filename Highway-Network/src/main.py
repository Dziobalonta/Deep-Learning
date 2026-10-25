import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# Fixed seed - same numbers, every time
SEED = 42
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "./data"

# For now only ToTensor (pixels -> floats in [0, 1])
transform = T.ToTensor()

train_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)

print(f"device: {DEVICE}")
print(train_set)
print(test_set)
print("classes:", train_set.classes)

# One example
image, label = train_set[0]
print(f"one image: shape={tuple(image.shape)}, \
      dtype={image.dtype}, "
      f"min={image.min():.3f}, \
      max={image.max():.3f}, \
      label={label} ({train_set.classes[label]})")

# Class balance, decides whether plain accuracy is a sensible metric
counts = torch.bincount(torch.tensor(train_set.targets), minlength=10)
print("train images per class:", counts.tolist())

# One mini-batch, to check the DataLoader works
loader = DataLoader(train_set, batch_size=64, shuffle=True)
images, labels = next(iter(loader))
print(f"one batch: images {tuple(images.shape)}, labels {tuple(labels.shape)}")