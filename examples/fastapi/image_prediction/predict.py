# predict.py

import sys
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from torchvision import models, transforms


# ---------------------------
# Validate CLI argument
# ---------------------------
if len(sys.argv) != 2:
    print("Usage: python predict.py <image_url>")
    sys.exit(1)

image_url = sys.argv[1]


# ---------------------------
# Download Image
# ---------------------------
def download_image(url: str):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


# ---------------------------
# Load Model
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()
model.to(device)

weights = models.ResNet18_Weights.DEFAULT
class_names = weights.meta["categories"]


# ---------------------------
# Preprocessing
# ---------------------------
transform = weights.transforms()


# ---------------------------
# Prediction
# ---------------------------
def predict(image: Image.Image):
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        top_probs, top_classes = torch.topk(probs, 5)

    results = []
    for prob, cls in zip(top_probs[0], top_classes[0]):
        results.append({
            "class": class_names[cls.item()],
            "confidence": round(prob.item(), 4)
        })

    return results


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    print(f"\nDownloading image from: {image_url}")
    image = download_image(image_url)

    print("Running inference...\n")
    predictions = predict(image)

    print("Top-5 Predictions:")
    for p in predictions:
        print(f"{p['class']}: {p['confidence']}")