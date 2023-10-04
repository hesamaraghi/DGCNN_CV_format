import torch
import torchvision.models as models

def print_layer_info(model, input_tensor):
    def forward_hook(module, input, output):
        nonlocal layer_num
        layer_num += 1
        print(f"Layer {layer_num}: {module.__class__.__name__}")
        print(f"  Input Shape: {str(input[0].shape)}")
        print(f"  Output Shape: {str(output.shape)}")
        total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  Parameters: {total_params}\n")

    layer_num = 0
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

# Load the pre-trained ResNet-34 model
resnet34 = models.resnet34(pretrained=True)

# Example input tensor (adjust the dimensions as needed)
input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image

# Print layer information
print_layer_info(resnet34, input_tensor)
