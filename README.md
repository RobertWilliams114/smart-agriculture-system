# Plant Disease Classification

A deep learning model that classifies plant diseases from leaf images using transfer learning with ResNet-18. Achieves **98.36% accuracy** across 38 disease categories.

## Overview

This project uses a pre-trained ResNet-18 model fine-tuned on the PlantVillage dataset to identify diseases in crop plants. It can classify diseases across 14 different plant species including tomatoes, potatoes, apples, grapes, and more.

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 98.64% |
| Total Classes | 38 |
| Dataset Size | 108,611 images |
| Model | ResNet-18 (Transfer Learning) |

## Supported Plants and Diseases

The model can classify the following:

- **Apple**: Scab, Black Rot, Cedar Apple Rust, Healthy
- **Blueberry**: Healthy
- **Cherry**: Powdery Mildew, Healthy
- **Corn**: Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
- **Grape**: Black Rot, Esca, Leaf Blight, Healthy
- **Orange**: Huanglongbing (Citrus Greening)
- **Peach**: Bacterial Spot, Healthy
- **Pepper Bell**: Bacterial Spot, Healthy
- **Potato**: Early Blight, Late Blight, Healthy
- **Raspberry**: Healthy
- **Soybean**: Healthy
- **Squash**: Powdery Mildew
- **Strawberry**: Leaf Scorch, Healthy
- **Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy

## Project Structure

```
smart-agriculture-system/
├── ImageClasses.ipynb          # Main training notebook
├── plant_disease_model.pth     # Trained model weights (43MB)
├── requirements-backup.txt     # Dependencies
├── data/
│   └── plantvillage dataset/   # Training data (not included)
└── venv/                       # Virtual environment (not included)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/smart-agriculture-system.git
cd smart-agriculture-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install torch torchvision pillow numpy matplotlib scikit-learn tqdm
```

4. Download the PlantVillage dataset from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) and place it in the `data/` folder.

## Usage

### Training

Open `ImageClasses.ipynb` in Jupyter Notebook and run all cells to train the model.

### Inference

```python
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 38)
model.load_state_dict(torch.load('plant_disease_model.pth', map_location='cpu'))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Predict
image = Image.open('path/to/leaf_image.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted_idx = torch.max(probabilities, 1)

print(f"Predicted class index: {predicted_idx.item()}")
print(f"Confidence: {confidence.item() * 100:.2f}%")
```

## Model Architecture

- **Base Model**: ResNet-18 pre-trained on ImageNet
- **Modification**: Final fully connected layer replaced (512 -> 38 classes)
- **Input Size**: 224x224 RGB images
- **Training**: 10 epochs with Adam optimizer (lr=0.001)

## Data Augmentation

Training transforms:
- Resize to 256x256
- Random crop to 224x224
- Random horizontal flip
- ImageNet normalization

## Dataset

The model was trained on the [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) containing 108,611 images of healthy and diseased plant leaves.

## Requirements

- Python 3.11+
- PyTorch 2.0+
- torchvision
- Pillow
- NumPy
- scikit-learn
- matplotlib
- tqdm

## License

This project is for educational purposes.

## Acknowledgments

- [PlantVillage Dataset](https://plantvillage.psu.edu/)
- PyTorch and torchvision teams
