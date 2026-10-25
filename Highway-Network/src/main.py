# The CIFAR-10 dataset
# The CIFAR-10 dataset consists of 60 000 32x32 colour images in 10 classes, with 6000 images per class. There are 50 000 training images and 10 000 test images

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# Fixed seed - same numbers, every time
SEED = 42
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "./data"


def get_dataloaders(data_dir, val_size=5000, batch_size=128, seed=SEED):
	'''Prepping data'''
	
	# Loading raw data once without transforms, only to read labels and pixels
	raw_train  = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)

	# STRATIFIED & SEEDED SPLIT
	# Seeded - the same random pick every time
	# Stratified - every class receives an equal share

	indices = list(range(len(raw_train.targets)))
	train_idx, val_idx = train_test_split(
		indices,
		test_size=val_size,			# how many go to validation
		stratify=raw_train.targets,	# stratified: keep class proportions
		random_state=seed,			# seeded: same split every run
	)

	# Rescaling: colors from [0, 255] to [0, 1] for better training,
	# only to compute mean and std for each channel (RGB)
	pixels = torch.tensor(raw_train.data[train_idx], dtype=torch.float32) / 255.0
	mean = pixels.mean(dim=(0, 1, 2)).tolist()
	std = pixels.std(dim=(0, 1, 2)).tolist()

	train_transform = T.Compose([
		T.RandomCrop(32, padding=4),	# Adding pixel shifts to aviod overfitting
		T.RandomHorizontalFlip(),		# Adding horizontal flips to avoid overfitting
		T.ToTensor(), 					# Rescaling: colors from [0, 255] to [0, 1]
		T.Normalize(mean, std), 		# Normalization for Gradient Descent - values around −2 to +2
	])

	val_transform = T.Compose([ # Validation and Test must be in the same format - the model learned from
		T.ToTensor(),
		T.Normalize(mean, std),
	])

	train_view  = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=train_transform)
	val_view = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=val_transform)

	test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=val_transform)


	train_set = Subset(train_view, train_idx)
	val_set = Subset(val_view, val_idx)
	
	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
	val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)


	return train_loader, val_loader, test_loader

class HighwayNetwork(nn.Module):
	"""Highway convolutional layer:
	y = H(x)*T(x) + x*(1 - T(x))
    H = conv + ReLU (transform)
	T = conv + sigmoid (gate)
    """
	def __init__(self, in_channels, out_channels, kernel_size=3, gate_bias=-2.0):
		super().__init__()

		# A convolution's window can't hang over the edge, so the image shrinks by kernel_size − 1 pixels.
		# Adding zeros on each side, which exactly replaces the lost pixels
		padding = kernel_size // 2

		self.transform = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)   # H
		self.gate = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)        # T


		# Gate bias starts negative -> layer mostly carries x at the start
		# gate bias b	sigmoid(b), how open the gate starts	layer at the start
		# 0		0.50 (50%)	half transforms, half copies
		# −1	0.27 (27%)	copies
		# −2	0.12 (12%)	copies a lot
		# −3	0.05 (5%)	almost only copies
		nn.init.constant_(self.gate.bias, gate_bias)

	def forward(self, x):
		H = self.transform(x)
		T = torch.sigmoid(self.gate(x))
		return H * T + x * (1 - T)


if __name__ == "__main__":
	# Data prep
	train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR)
	print("sizes:", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

	images, labels = next(iter(train_loader))
	print("batch:", tuple(images.shape), tuple(labels.shape))


	# Testing the Highway layer
	layer = HighwayNetwork(in_channels=16, out_channels=16)
	x = torch.randn(8, 16, 32, 32)
	y = layer(x)
	print("in :", tuple(x.shape), "out:", tuple(y.shape))
	print("gate mean at init:", torch.sigmoid(layer.gate(x)).mean().item())
	print("params:", sum(p.numel() for p in layer.parameters()))
