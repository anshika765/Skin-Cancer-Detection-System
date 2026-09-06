# DermAI: Skin Lesion Multi-Class Diagnostic Assistant

An end-to-end medical AI application that classifies dermatoscope skin lesion images into 9 distinct diagnostic categories using transfer learning with **MobileNetV2** and an interactive **Flask** web interface.

---

## 📌 Project Overview
* **Architecture:** MobileNetV2 pretrained on ImageNet
* **Training Strategy:** 2-Phase Training (Classifier Head Training followed by Fine-Tuning with frozen BatchNormalization layers for stability)
* **Dataset:** ISIC Skin Cancer Dataset (9 Classes)
* **Web UI:** Modern Glassmorphic Dark-Mode UI with real-time scan preview and multi-class probability distribution bars.

---

## 🧬 Disease Classes (9 Categories)
1. Actinic Keratosis
2. Basal Cell Carcinoma
3. Dermatofibroma
4. Melanoma
5. Nevus
6. Pigmented Benign Keratosis
7. Seborrheic Keratosis
8. Squamous Cell Carcinoma
9. Vascular Lesion

---

## 📁 Project Structure
```text
Skin_Cancer_Detection/
├── dataset/                    # ISIC Dataset (Train & Test folders)
├── static/
│   ├── css/
│   │   └── style.css          # Glassmorphism Dark-Mode styling
│   └── uploads/               # Temporary uploads
├── templates/
│   └── index.html             # Diagnostic Dashboard UI
├── train.py                   # Transfer Learning pipeline & metrics
├── app.py                     # Flask inference backend
├── requirements.txt           # Python dependencies
├── confusion_matrix.png       # Test evaluation heatmap
└── skin_cancer_best_model.keras # Serialized weights file

## 💾 Dataset Link
The model was trained on the official ISIC Skin Lesion dataset. You can download the dataset directly from Kaggle / ISIC Archive:
* [Download ISIC Skin Cancer Dataset on Kaggle](https://www.kaggle.com/datasets/nodoubttome/skin-cancer-isic-the-international-skin-imaging-collaboration)