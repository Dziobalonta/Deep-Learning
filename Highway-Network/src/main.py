# The CIFAR-10 dataset
# The CIFAR-10 dataset consists of 60 000 32x32 colour images in 10 classes, with 6000 images per class. There are 50 000 training images and 10 000 test images

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
import math
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
    def __init__(self, channels, kernel_size=3, gate_bias=-2.0):
        super().__init__()

        # A convolution's window can't hang over the edge, so the image shrinks by kernel_size − 1 pixels.
        # Adding zeros on each side, which exactly replaces the lost pixels
        padding = kernel_size // 2

        # H
        self.transform = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding)
        # T
        self.gate = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding) 

        # Gate bias starts negative -> layer mostly carries x at the start
        # gate bias b	sigmoid(b), how open the gate starts layer at the start
        # 0		0.50 (50%)	half transforms, half copies
        # −1	0.27 (27%)	copies
        # −2	0.12 (12%)	copies a lot
        # −3	0.05 (5%)	almost only copies
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, x):
        H = self.transform(x)
        T = torch.sigmoid(self.gate(x))
        return H * T + x * (1 - T)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train() # training mode
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device) # loading data to GPU if available, else CPU
        labels = labels.to(device)

        optimizer.zero_grad() # Clearing old gradient

        output = model(images) # Forward pass, 10 numbers raw scores (logits) for each class
        # airplane  car   bird  cat   deer  dog   frog  horse ship  truck
        # 0.3    -1.2   0.8   2.1   0.1   1.5  -0.4   0.2  -0.9  -1.1
        # Bigger means the model believes it more. TYhe highest is "cat" (2.1), so the model guesses cat

        loss = criterion(output, labels)
        loss.backward() # Backpropagation, computing gradient of the loss with respect to every weight

        optimizer.step() # Updating weights based on the computed gradients

        total_loss += loss.item() * len(labels) # average loss for the batch * batch size

        # argmax returns the position of the largest number. dim=1 means "search along each row"
        # .sum() counts the Trues
        # .item() turns PyTorch object, (possibly on the GPU) into a plain Python integer you can add up
        correct += (output.argmax(dim=1) == labels).sum().item()
        total += len(labels)


    return total_loss / total, correct / total

@torch.no_grad() # no gradients needed - faster, less memory
def evaluate(model, loader, criterion, device):
    model.eval() # evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total

if __name__ == "__main__":
    # Data prep
    train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR)
    print("sizes:", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

    images, labels = next(iter(train_loader))
    print("batch:", tuple(images.shape), tuple(labels.shape))


    # Testing the Highway layer
    layer = HighwayNetwork(channels=16)
    x = torch.randn(8, 16, 32, 32)
    y = layer(x)
    print("in :", tuple(x.shape), "out:", tuple(y.shape))
    print("gate mean at init:", torch.sigmoid(layer.gate(x)).mean().item())
    print("params:", sum(p.numel() for p in layer.parameters()))

    
    # criterion = nn.CrossEntropyLoss().to(DEVICE) # softmax + cross-entropy
    # train_one_epoch(layer, train_loader, criterion, torch.optim.Adam(layer.parameters()), DEVICE)
    
    # Testing the training loop with the simplest model: softmax regression
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Base line: 5,000 imgs, 10 classes, dataset is balanced = Random guessing accuracy = 1/10 = 0.1
    print("baseline accuracy:", 1 / 10)

    # The loss looks at the probability the model gives to the true class
    # loss = −ln(p_true)
    # p = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)      # 10 classes, each 1/10
    loss_base = - math.log(0.1)
    print("baseline loss:", loss_base)



    for epoch in range(3):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        print(f"epoch {epoch + 1}: train loss {loss:.3f}, train accur {acc:.3f}")

