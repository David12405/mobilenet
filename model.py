import io
import json
from typing import List, Dict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ===== Config =====
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load labels =====
with open("labels.json", "r", encoding="utf-8") as f:
    classes = json.load(f)

num_classes = len(classes)

# ===== Build model giống lúc train =====
model = models.mobilenet_v3_large(weights=None)

# Không bắt buộc freeze khi inference, nhưng để kiến trúc đồng nhất cũng được
for param in model.features.parameters():
    param.requires_grad = False

in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, num_classes)

# ===== Load checkpoint =====
state_dict = torch.load("C:\\Users\\huuda\\Downloads\\best_mobilenetv3_finetune.pth", map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# ===== Transform phải khớp val/test transform trong notebook =====
infer_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    x = infer_transform(image).unsqueeze(0).to(device)
    return x


@torch.no_grad()
def predict_image(image: Image.Image, top_k: int = 3) -> Dict:
    x = preprocess_image(image)
    outputs = model(x)
    probs = torch.softmax(outputs, dim=1)

    top_k = min(top_k, len(classes))
    top_probs, top_idxs = torch.topk(probs, k=top_k, dim=1)

    predictions: List[Dict] = []
    for p, i in zip(top_probs[0], top_idxs[0]):
        idx = int(i.item())
        predictions.append({
            "label_id": idx,
            "label": classes[idx],
            "confidence": float(p.item())
        })

    return {
        "top_k": top_k,
        "predictions": predictions
    }