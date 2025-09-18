from torchvision import transforms

# Data preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create label mapping dictionary
label_map = {
    '半环': 0,  # Half-ring
    '半璧': 1,  # Half-bi
    '半圆': 2,  # Semicircle
    '桥': 3     # Bridge
}

# Reverse label mapping for interpretation
rev_label_map = {v: k for k, v in label_map.items()}