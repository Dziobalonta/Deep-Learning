# The CIFAR-10 dataset
# The CIFAR-10 dataset consists of 60 000 32x32 colour images in 10 classes, with 6000 images per class. There are 50 000 training images and 10 000 test images

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import math
import copy
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# Fixed seed - same numbers, every time
SEED = 42
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# data/ next to src/, no matter where we run from
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


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

class HighwayLayer(nn.Module):
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
        H = F.relu(self.transform(x))
        T = torch.sigmoid(self.gate(x))
        return H * T + x * (1 - T)

class PlainConv(nn.Module):
    """Plain conv layer: y = ReLU(conv(x)). Same shape in and out, no gate."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        return F.relu(self.conv(x))


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

        loss = criterion(output, labels)
        if torch.isnan(loss):
            raise RuntimeError("loss is nan - training aborted")

        loss.backward() # Backpropagation, computing gradient of the loss with respect to every weight

        #  if the total size of the gradient is above 5, scale the whole gradient down to size 5
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) 

        optimizer.step() # Updating weights based on the computed gradients

        total_loss += loss.item() * len(labels) # average loss for the batch * batch size

        # argmax returns the position of the largest number. dim=1 means "search along each row"
        # .sum() counts the Trues
        # .item() turns PyTorch object, (possibly on the GPU) into a plain Python integer you can add up
        correct += (output.argmax(dim=1) == labels).sum().item()
        total += len(labels)


    return total_loss / total, correct / total

class Net(nn.Module):
    """CIFAR-10 network built from highway or plain layers.
    widths = channels per stage, layers_per_stage = N, layer_type = HighwayConv or PlainConv
    """
    def __init__(self, layer_type, layers_per_stage=3, widths=(32, 64, 128), num_classes=10, conv_init=False):
        super().__init__()
        layers = []
        in_ch = 3
        for i, width in enumerate(widths):

            if i > 0:
                # MaxPool2d(2): halves height and width (32→16→8), channels stay the same
                layers.append(nn.MaxPool2d(2))
            layers.append(nn.Conv2d(in_ch, width, 3, padding=1)) # plain conv: change channels
            layers.append(nn.ReLU())
            for _ in range(layers_per_stage):
                layers.append(layer_type(width)) # highway (or plain) layers
            in_ch = width

        self.features = nn.Sequential(*layers) # all the layers packed into one block
        self.pool = nn.AdaptiveAvgPool2d(1)	 # squash each grid to one number(C, 8, 8) -> (C, 1, 1)
        self.classifier = nn.Linear(widths[-1], num_classes)	# computes 10 weighted sums, one per class

        if conv_init:
            # He init for all convs: keeps signal size stable through conv+ReLU (paper Sec 3.1)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")


    def forward(self, x):
        x = self.features(x) # pushes the batch through all the bricks in order
        x = self.pool(x).flatten(1)		# avg each 8×8 grid → 1 number, drop the 1×1
        return self.classifier(x)		# raw scores (logits), no softmax


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

def run_experiment(model, train_loader, val_loader, epochs, lr, momentum, device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss() # softmax + cross-entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(f"epoch {epoch + 1:2d} | train loss {train_loss:.3f} acc {train_acc:.3f} "
              f"| val loss {val_loss:.3f} acc {val_acc:.3f}")

    model.load_state_dict(best_state)   # go back to the best epoch
    return history, best_acc


if __name__ == "__main__":
    # Data prep
    train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR)
    print("sizes:", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

    images, labels = next(iter(train_loader))
    print("batch:", tuple(images.shape), tuple(labels.shape))


    # Testing the Highway layer
    layer = HighwayLayer(channels=16)
    x = torch.randn(8, 16, 32, 32)
    y = layer(x)
    print("in :", tuple(x.shape), "out:", tuple(y.shape))
    print("gate mean at init:", torch.sigmoid(layer.gate(x)).mean().item())
    print("params:", sum(p.numel() for p in layer.parameters()))
    print()

    # Base line: 5,000 imgs, 10 classes, dataset is balanced = Random guessing accuracy = 1/10 = 0.1
    print("baseline accuracy:", 1 / 10)

    # The loss looks at the probability the model gives to the true class
    # loss = −ln(p_true)
    # p = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)      # 10 classes, each 1/10
    loss_base = - math.log(0.1)
    print("baseline loss:", loss_base)
    print()

    print("=== Highway ===")
    for lr in [0.3, 0.1, 0.03, 0.01, 0.003]:
        torch.manual_seed(SEED) # same starting weights for every lr
        model = Net(HighwayLayer, layers_per_stage=3, conv_init=True)
        print(f"\n--- lr = {lr} ---")
        history, best_acc = run_experiment(model, train_loader, val_loader, epochs=6, lr=lr, momentum=0.9, device=DEVICE)
        print("best val acc:", best_acc)
    print()

    print("=== PlainConv ===")
    for lr in [0.3, 0.1, 0.03, 0.01, 0.003]:
        torch.manual_seed(SEED) # same starting weights for every lr
        model = Net(PlainConv, layers_per_stage=3, conv_init=True)
        print(f"\n--- lr = {lr} ---")
        history, best_acc = run_experiment(model, train_loader, val_loader, epochs=6, lr=lr, momentum=0.9, device=DEVICE)
        print("best val acc:", best_acc)

