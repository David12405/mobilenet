import json
import torch
import torch.nn as nn
from torchvision import models

# ===== Config =====
IMG_SIZE = 224
ONNX_OUTPUT = "mobilenetv3_finetune.onnx"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load labels =====
with open("labels.json", "r", encoding="utf-8") as f:
    classes = json.load(f)

num_classes = len(classes)

# ===== Build model (same architecture as training) =====
model = models.mobilenet_v3_large(weights=None)

for param in model.features.parameters():
    param.requires_grad = False

in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, num_classes)

# ===== Load checkpoint =====
state_dict = torch.load("best_mobilenetv3_finetune.pth", map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# ===== Export to ONNX =====
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_OUTPUT,
    export_params=True,         # Store trained weights inside the .onnx file
    opset_version=17,           # Latest stable opset, compatible with onnxruntime
    do_constant_folding=True,   # Optimize constants for faster inference
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input":  {0: "batch_size"},   # Allow variable batch size
        "output": {0: "batch_size"},
    },
)

print(f"✅ Exported to {ONNX_OUTPUT}")

# ===== Verify the exported model =====
import onnx
onnx_model = onnx.load(ONNX_OUTPUT)
onnx.checker.check_model(onnx_model)
print("✅ ONNX model verified successfully")
print(f"   Inputs : {[i.name for i in onnx_model.graph.input]}")
print(f"   Outputs: {[o.name for o in onnx_model.graph.output]}")