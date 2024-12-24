import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # CONV1 Block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv2 Block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv3 Block
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv4 Block
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv5 Block
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc14 = nn.Linear(512 * 7 * 7, 4096)
        self.fc15 = nn.Linear(4096, 4096)
        self.fc16 = nn.Linear(4096, 2622)

        # Activation functions and dropout
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        # Define the transform (with normalization)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=(93.59396362304688 / 255, 104.76238250732422 / 255, 129.186279296875 / 255),
                std=(1, 1, 1)
            )
            
        ])

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.pool3(x)
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.pool4(x)
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))
        x = self.pool5(x)
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc14(x)))
        x = self.dropout(self.relu(self.fc15(x)))
        x = self.fc16(x)
        return x

    def facial_features(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.pool3(x)
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.pool4(x)
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))
        x = self.pool5(x)
        x = torch.flatten(x, 1)
        x = self.fc14(x)
        return x

    # __call__ = forward_fp

    # To extract embeddings
    def feature_extractor(self, x: torch.Tensor):
        x = self.transform(x)  # Apply the transform (normalization)
        return self.facial_features(x).flatten().detach().numpy()

face_model = VGG16()
file_path = '/home/sriram/Sriramm/Sriram/FP/FP_final/VGG/vgg_face_dag.pth'

model_data = dict(torch.load(file_path, map_location=torch.device('cpu'), weights_only=True))

lis = [
    'conv1_1.weight', 'conv1_1.bias', 'conv1_2.weight', 'conv1_2.bias',
    'conv2_1.weight', 'conv2_1.bias', 'conv2_2.weight', 'conv2_2.bias',
    'conv3_1.weight', 'conv3_1.bias', 'conv3_2.weight', 'conv3_2.bias',
    'conv3_3.weight', 'conv3_3.bias', 'conv4_1.weight', 'conv4_1.bias',
    'conv4_2.weight', 'conv4_2.bias', 'conv4_3.weight', 'conv4_3.bias',
    'conv5_1.weight', 'conv5_1.bias', 'conv5_2.weight', 'conv5_2.bias',
    'conv5_3.weight', 'conv5_3.bias', 'fc6.weight', 'fc6.bias',
    'fc7.weight', 'fc7.bias', 'fc8.weight', 'fc8.bias'
]

lis1 = [
    'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
    'conv3.weight', 'conv3.bias', 'conv4.weight', 'conv4.bias',
    'conv5.weight', 'conv5.bias', 'conv6.weight', 'conv6.bias',
    'conv7.weight', 'conv7.bias', 'conv8.weight', 'conv8.bias',
    'conv9.weight', 'conv9.bias', 'conv10.weight', 'conv10.bias',
    'conv11.weight', 'conv11.bias', 'conv12.weight', 'conv12.bias',
    'conv13.weight', 'conv13.bias', 'fc14.weight', 'fc14.bias',
    'fc15.weight', 'fc15.bias', 'fc16.weight', 'fc16.bias'
]

for i in range(len(lis)):
    try:
        if lis[i] in model_data and lis1[i] in face_model.state_dict():
            face_model.state_dict()[lis1[i]].copy_(model_data[lis[i]])
        else:
            raise KeyError(f"Key mismatch: {lis[i]} or {lis1[i]} not found.")
    except (RuntimeError, KeyError) as e:
        print(f"Error during model weight loading: {e}")
        exit(1)

    
# print("Model weights loaded successfully!")


