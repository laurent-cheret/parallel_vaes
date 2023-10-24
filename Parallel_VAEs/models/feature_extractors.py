# models/feature_extractors.py

import torch
import torch.nn as nn
import torchvision.models as models


class SmallCNNFeatureExtractor(nn.Module):
    def __init__(self, num_channels):
        super(SmallCNNFeatureExtractor, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap_layer = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Forward pass through the CNN layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.gap_layer(x)

        # Flatten the feature maps
        x = x.view(x.size(0), -1)
        return x


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()

        # Load the pre-trained VGG16 model from torchvision
        vgg16 = models.vgg16(pretrained=True)

        # Remove the final classification layer
        self.features = nn.Sequential(*list(vgg16.children())[:-1])
        # self.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        self.gap_layer = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Forward pass through the feature extractor
        features = self.features(x)
        output = self.gap_layer(features).squeeze()
        return output
