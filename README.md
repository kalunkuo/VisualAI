# 🩹 Wound Detection AI

A deep learning-powered image classification app that identifies wound types from medical images. Built using **PyTorch** and **Streamlit**, this project allows users to upload images and receive predictions across 10 common wound categories.

---

## 🔍 Features

- ✅ Classifies wounds into 10 types:
  - Abrasions, Bruises, Burns, Cuts, Diabetic Wounds, Lacerations, Normal, Pressure Wounds, Surgical Wounds, Venous Wounds
- 🧠 Uses **ResNet50 + transfer learning**
- ⚡ Fast and lightweight with **Streamlit**
- 📦 Easy to retrain or integrate into other workflows

---

## 🧪 Tech Stack

- Python
- PyTorch
- Torchvision
- Streamlit
- PIL (Python Imaging Library)

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/kalunkuo/wound-detection-ai.git
   cd wound-detection-ai
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Place your dataset in the dataset/ folder (each wound type should be a folder with images inside).
  - Kaggle: https://www.kaggle.com/datasets/ibrahimfateen/wound-classification/data
4. Train the model:
   ```bash
   python train.py
   
4. Run the Streamlit app:
   ```bash
   streamlit run wound_app.py
