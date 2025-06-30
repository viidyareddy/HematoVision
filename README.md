# 🗑️ Waste Classifier using VGG16

This project is a deep learning-based waste classification system that uses transfer learning with the **VGG16** model to identify types of waste such as **Biodegradable**, **Recyclable**, and **Trash**.

## 🧠 Key Features

- 📸 Image preprocessing and augmentation
- 📊 Model evaluation on test dataset
- 📂 Classifies waste into 3 categories
- 💾 Trained model saved as `vgg.h5`
- ✅ Transfer learning with pre-trained VGG16


## 📁 Dataset Structure

Ensure your dataset is organized as:

output_dataset/
├── train/
│ ├── Biodegradable Images/
│ ├── Recyclable Images/
│ └── Trash Images/
├── val/
│ ├── Biodegradable Images/
│ ├── Recyclable Images/
│ └── Trash Images/
└── test/
├── Biodegradable Images/
├── Recyclable Images/
└── Trash Images/
