import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from torchvision import models
import torch.nn as nn

# Load model
# model = torch.load('model.pth', map_location='cpu')

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)  # 10 classes
model.load_state_dict(torch.load('model_weights.pth', map_location='cpu'))
model.eval()
model.eval()

# Class labels
classes = [
    "Abrasions", "Bruises", "Burns", "Cut", "Diabetic Wounds",
    "Laseration", "Normal", "Pressure Wounds", "Surgical Wounds", "Venous Wounds"
]

# Streamlit app
st.title("Wound Type Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        predicted = classes[output.argmax().item()]

    st.success(f"Predicted Wound Type: {predicted}")
