import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet50_Weights
from PIL import Image
import os
import json
from tqdm import tqdm

# Load ResNet50 without classification layer
resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()
resnet.eval()

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = resnet(input_tensor)
    return embedding.squeeze(0).tolist()  # 2048-dim list

# Directory to walk
dataset_root = "dataset"
output = []

print(f"🔍 Scanning dataset in: {dataset_root}\n")

# Walk through subfolders
for label in os.listdir(dataset_root):
    label_path = os.path.join(dataset_root, label)
    if not os.path.isdir(label_path):
        continue

    for fname in tqdm(os.listdir(label_path), desc=f"Processing {label}"):
        fpath = os.path.join(label_path, fname)
        try:
            vector = get_embedding(fpath)
            output.append({
                "filename": fname,
                "label": label,
                "embedding": vector
            })
        except Exception as e:
            print(f"⚠️ Skipped {fpath}: {e}")

# Save to JSON for testing / loading into MongoDB later
with open("embeddings.json", "w") as f:
    json.dump(output, f)

print(f"\n✅ Extracted embeddings for {len(output)} images.")
print("📁 Saved to embeddings.json")
