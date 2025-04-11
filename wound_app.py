import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import os
from pymongo import MongoClient
from datetime import datetime
from torchvision.models import ResNet50_Weights

# Load trained classifier
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('model_weights.pth', map_location='cpu'))
model.eval()

# Load ResNet50 for embedding extraction
embedder = models.resnet50(weights=ResNet50_Weights.DEFAULT)
embedder.fc = torch.nn.Identity()
embedder.eval()

# Class labels
classes = [
    "Abrasions", "Bruises", "Burns", "Cut", "Diabetic Wounds",
    "Laseration", "Normal", "Pressure Wounds", "Surgical Wounds", "Venous Wounds"
]

# MongoDB connection (load from env or hardcode)
uri = os.getenv("MONGO_URI")
client = MongoClient(uri)
db = client["wound_ai_db"]
collection = db["embeddings"]

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# UI
st.title("Wound Type Detection with Similar Image Search")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted = classes[output.argmax().item()]
        embedding = embedder(input_tensor).squeeze(0).tolist()

    st.success(f"Predicted Wound Type: {predicted}")

    # Save to DB
    doc = {
        "filename": uploaded_file.name,
        "label": predicted,
        "embedding": embedding,
        "timestamp": datetime.now()
    }
    collection.insert_one(doc)

    st.info("🔎 Searching for similar wound images...")

    # Query MongoDB vector search
    results = db.command({
        "aggregate": "embeddings",
        "pipeline": [
            {
                "$vectorSearch": {
                    "index": "vector_index_2",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 100,
                    "limit": 5
                }
            },
            {
                "$project": {
                    "filename": 1,
                    "label": 1,
                    "score": { "$meta": "vectorSearchScore" }
                }
            }
        ],
        "cursor": {}
    })

    docs = results.get("cursor", {}).get("firstBatch", [])

    if not docs:
        st.warning("⚠️ No similar wound images found.")
    else:
        st.subheader("🧠 Most Similar Wound Images")
        cols = st.columns(len(docs))

        for i, doc in enumerate(docs):
            with cols[i]:
                label = doc.get("label", "Unknown")
                filename = doc.get("filename", "Unknown")
                score = doc.get("score", 0.0)
                image_path = f"dataset/{label}/{filename}"

                st.markdown(f"**{filename}**")
                if os.path.exists(image_path):
                    st.image(image_path, caption=f"{label} ({score:.2f})", use_container_width=True)
                else:
                    st.warning(f"Image not found: `{image_path}`")
