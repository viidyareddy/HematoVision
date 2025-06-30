# 🗑️ Waste Classifier using VGG16

This project is a deep learning-based waste classification system that uses transfer learning with the **VGG16** model to identify types of waste such as **Biodegradable**, **Recyclable**, and **Trash**.

## 🧠 Key Features

- ✅ Transfer learning with pre-trained VGG16
- 📂 Classifies waste into 3 categories
- 📸 Image preprocessing and augmentation
- 📊 Model evaluation on test dataset
- 💾 Trained model saved as `waste_classifier_vgg16.h5`

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



## 🛠️ Requirements

- Python 3.x
- TensorFlow
- NumPy
- IPython (for displaying images)

Install dependencies:

```bash
pip install tensorflow numpy ipython

## 🛠️ Requirements

- Python 3.x
- TensorFlow
- NumPy
- IPython (for displaying images)

Install dependencies:

```bash
pip install tensorflow numpy ipython
