import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights,ResNet50_Weights


model_efficient = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT, dropout=0.5)
model_efficient.classifier[1] = nn.Linear(in_features=1536, out_features=1)


class ResNet18FeatureExtractor(nn.Module):
    """ A CNN pytorch model for marker positivity prediction, based on ResNet. Meant to be used on single channel transformed cell images 
    """
    def __init__(self, num_classes=2, feature_dim=64, dropout_rate=0.5):
        super(ResNet18FeatureExtractor, self).__init__()
        # Load a pre-trained ResNet18 model
        self.resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Modify the first convolutional layer to accept 1 channel input if needed
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Number of features in the layer before the fully connected layer
        num_ftrs = self.resnet18.fc.in_features
        # Replace the final fully connected layer for binary classification
        self.resnet18.fc = nn.Linear(num_ftrs, num_classes)
        # Additional layer to compress the features to the desired dimensionality
        self.feature_compress = nn.Linear(num_ftrs, feature_dim)
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
		
    def forward(self, x):
        # Extract features from the next-to-last layer
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        
        # Apply average pooling
        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply dropout before feature compression to enhance robustness
        x = self.dropout(x)
        
        # Compress the features to the desired dimensionality
        compressed_features = self.feature_compress(x)
        
        # Apply dropout before the final classification layer to prevent overfitting
        x = self.dropout(x)
        
        # Classification output
        out = self.resnet18.fc(x)
        
        return out, compressed_features


class ResNet18CellWithSegFeatureExtractor(nn.Module):
    """ A CNN pytorch model for marker positivity prediction, based on ResNet. Meant to be used on 3 channel cell images:
    marker intensity, cell segmentation and neighbour cell segmentation.
    """
    def __init__(self, num_classes=2, feature_dim=64, dropout_rate=0.5, resnet_scale=18):
        super(ResNet18CellWithSegFeatureExtractor, self).__init__()
        # Load a pre-trained ResNet18 model
        if resnet_scale == 18:
            self.resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif resnet_scale == 50:
            self.resnet18 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Modify the first convolutional layer to accept 1 channel input if needed
        # self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Number of features in the layer before the fully connected layer
        num_ftrs = self.resnet18.fc.in_features
        # Replace the final fully connected layer for binary classification
        self.resnet18.fc = nn.Linear(num_ftrs, num_classes)
        # Additional layer to compress the features to the desired dimensionality
        self.feature_compress = nn.Linear(num_ftrs, feature_dim)
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        # Extract features from the next-to-last layer
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        
        # Apply average pooling
        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply dropout before feature compression to enhance robustness
        x = self.dropout(x)
        
        # Compress the features to the desired dimensionality
        compressed_features = self.feature_compress(x)
        
        # Apply dropout before the final classification layer to prevent overfitting - WHY TWICE???
        # x = self.dropout(x)
        
        # Classification output
        out = self.resnet18.fc(x)
        
        return out, compressed_features
    

class ResNet18CellWithSegAndExpressionType(nn.Module):
    """ A CNN pytorch model for marker positivity prediction, based on ResNet. Meant to be used on 3 channel cell images:
    marker intensity, cell segmentation and neighbour cell segmentation. Also gets the marker type encoding (membranal, cytoplasmic or nuclear) as additional input to the fc layer.
    """

    def __init__(self, num_classes=2, feature_dim=64, dropout_rate=0.5, marker_type_encoding_dim=1):
        super(ResNet18CellWithSegAndExpressionType, self).__init__()
        # Load a pre-trained ResNet18 model
        self.resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Modify the first convolutional layer to accept 1 channel input if needed
        # self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Number of features in the layer before the fully connected layer
        num_ftrs = self.resnet18.fc.in_features
        # Replace the final fully connected layer for binary classification
        self.resnet18.fc = nn.Linear(num_ftrs+1, num_classes)
        # Additional layer to compress the features to the desired dimensionality
        self.feature_compress = nn.Linear(num_ftrs, feature_dim)
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        self.marker_type_encoding_layer = nn.Linear(marker_type_encoding_dim, 1)
		
    def forward(self, x, marker_type):
        # Extract features from the next-to-last layer
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        
        # Apply average pooling
        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        
        
        # Apply dropout before feature compression to enhance robustness
        x = self.dropout(x)


        # Compress the features to the desired dimensionality
        compressed_features = self.feature_compress(x)
        
        # Apply dropout before the final classification layer to prevent overfitting
        # x = self.dropout(x)
                
        # Marker type
        marker_type = self.marker_type_encoding_layer(marker_type)
        x = torch.cat((x, marker_type), dim=1)
        # Classification output
        out = self.resnet18.fc(x)
        
        return out, compressed_features
