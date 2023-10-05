import torch
import torch.nn as nn


def print_layer_info(model, input_tensor):
    def forward_hook(module, input, output):
        nonlocal layer_num
        nonlocal cum_params
        layer_num += 1
        print(f"Layer {layer_num}: {module.__class__.__name__}")
        print(f"  Input Shape: {str(input[0].shape)}")
        print(f"  Output Shape: {str(output.shape)}")
        total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        cum_params += total_params
        print(f"  Parameters: {total_params}\n")
        print(f"  Cumulative Parameters: {cum_params}\n")

    layer_num = 0
    cum_params = 0
    hooks = []

    for layer_name, layer in model.named_children():
        hook = layer.register_forward_hook(forward_hook)
        hooks.append(hook)

    print("Model Architecture:")
    print(model)
    print("\nLayer Information:")

    with torch.no_grad():
        model(input_tensor)

    for hook in hooks:
        hook.remove()



class CNNClassifier(nn.Module):
    def __init__(self, num_classes=100):
        super(CNNClassifier, self).__init__()
        
        # Layer 1: Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Layer 2: Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Layer 3: Convolutional Layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,  stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Layer 4: Convolutional Layer
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Layer 5: Convolutional Layer
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(inplace=True)
        
        # Layer 6: Convolutional Layer
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # Fully Connected Layers
        self.fc1 = nn.Linear(256, 100)  # Adjusted input size for the final fully connected layer
        
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.avgpool(x)
        # Flatten the feature map
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.fc1(x)
        
        return x

# Create an instance of the CNNClassifier
model = CNNClassifier(num_classes=100)


# Example input tensor (adjust the dimensions as needed)
input_tensor = torch.randn(1, 9, 224, 224)  # Batch size of 1, 3 channels, 224x224 image

# Print layer information
print_layer_info(model, input_tensor)
