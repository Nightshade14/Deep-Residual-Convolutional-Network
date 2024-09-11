# Importing the Libraries
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision.transforms import v2

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_label_names_from_bytes_dict(x:dict) -> list:
    data = x.get(b"label_names")
    labels =  list(map(lambda m: m.decode("utf-8"),data))
    label_dict = dict()
    for i in range(10):
        label_dict.update({labels[i]: i})
    return label_dict

def is_my_model_under_5m_params(model):
    FIVE_MILLION = 5_000_000
    sum = 0
    for x in model.parameters():
        sum += x.numel()
    print(sum)
    if sum <= FIVE_MILLION:
        print("less than 5 million params")
    else:
        print(f"Decrease {sum - FIVE_MILLION} params!!!")


META_FILE_PATH = "./dataset/train/batches.meta"
TRAINING_FILE_PATH = "./dataset/train/"
VALIDATION_FILE_PATH = "./dataset/val/test_batch"
TEST_FILE_PATH = "./dataset/test/cifar_test_nolabels.pkl"

IS_LOGGING_ENABLED = True

torch.manual_seed(0)



# TRANSFORM THE INPUT DATA

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Composing multiple transformations to apply on the image when creating the dataset
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# The validation images would only be normalized as we want to infer on it
transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# loading the data

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
val_loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)




# DEFINE MODEL

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 32, layers[0])
        self.layer2 = self.make_layer(block, 64, layers[1], 2)
        self.layer3 = self.make_layer(block, 128, layers[2], 2)
        self.layer4 = self.make_layer(block, 256, layers[3], 2)
        self.avg_pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(256, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

model = ResNet(ResidualBlock, [3, 3, 3, 3])




# MODEL TRAINING LOOP

#Try the Learning Rate Scheduler
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Choosing loss and optimizers
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=6e-4)

# Define the learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

num_epochs = 150

loss_values = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    loss_values.append(running_loss/len(train_loader))


#  EVALUATE MODEL

import pickle

def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch

cifar10_batch = load_cifar_batch('/content/dataset/test/cifar_test_nolabels.pkl')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

images = cifar10_batch[b'data'].reshape(-1, 3, 32, 32)
ids = cifar10_batch[b'ids']
pred_labels = np.zeros(ids.shape, dtype="int32")

model = model.cuda()
model.eval()

for i in range(len(images)):
    with torch.no_grad():
        test_image = images[i]
        test_image = np.transpose(test_image, (1,2,0))
        test_image = transform_test(test_image)
        test_image = test_image.unsqueeze(0).cuda()

        pred = model(test_image)
        _, predicted = torch.max(pred, 1)
        pred_labels[i] = predicted

import pandas as pd
test_results = pd.DataFrame({'ID': ids, 'Labels': pred_labels})
test_results.to_csv('/content/dataset/result.csv', index=False)


torch.save(model.state_dict(), "./model.pth")
