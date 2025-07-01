# model.py

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import io

# --- CONFIGURATION ---
# Use CPU for inference, as it's more common for deployment environments
DEVICE = "cpu"
# Path to your trained model weights
MODEL_PATH = "house_plant_classifier_v1.pth"
# Path to the text file containing class names
CLASS_NAMES_PATH = "class_names.txt"


# --- MODEL ARCHITECTURE ---
# This function must be identical to the one in your training script
def create_efficientnet_b0(num_classes: int) -> nn.Module:
    """Creates an EfficientNet-B0 model with a custom classifier head."""
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=None) # Load architecture without weights

    # Freeze all base layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Recreate the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=num_classes),
    )
    return model


# --- MODEL LOADING ---
def load_model_and_classes():
    """Loads the model, class names, and transformations."""
    # Load class names
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    
    num_classes = len(class_names)

    # Create model with the correct number of classes
    model = create_efficientnet_b0(num_classes=num_classes)
    
    # Load the state dictionary
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    
    # Set model to evaluation mode
    model.to(DEVICE)
    model.eval()

    # Get the appropriate transforms
    weights = models.EfficientNet_B0_Weights.DEFAULT
    auto_transforms = weights.transforms()

    return model, class_names, auto_transforms


# --- PREDICTION FUNCTION ---
def predict(image_bytes: bytes, model: nn.Module, class_names: list, transform: transforms.Compose) -> tuple[str, float]:
    """
    Makes a prediction on a single image.

    Args:
        image_bytes: The image file in bytes.
        model: The trained PyTorch model.
        class_names: A list of class names.
        transform: The torchvision transform to apply to the image.

    Returns:
        A tuple containing the predicted class name and the confidence score.
    """
    # Open the image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert image to RGB (handles PNGs with alpha channels)
    image = image.convert("RGB")

    # Transform the image and add a batch dimension
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Make prediction
    with torch.inference_mode():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        
    # Get top prediction
    confidence, predicted_idx = torch.max(probabilities, 1)
    predicted_class = class_names[predicted_idx.item()]
    
    return predicted_class, confidence.item()