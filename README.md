
# 🥬 Leaf Disease Classification with CNN

This project uses a Convolutional Neural Network (CNN) to classify bell pepper plant leaves as either **Healthy** or **Bacterial Spot Infected**, based on image data. It also includes Digital Image Processing (DIP) techniques like grayscale conversion, thresholding, and contour detection to visualize key stages of image preprocessing.

## 🧠 Project Overview

- 🔍 Binary classification: Healthy vs Bacterial Spot
- 📷 Dataset: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- 🛠️ Tools: TensorFlow, OpenCV, Matplotlib, Scikit-learn
- 📁 Input: RGB leaf images
- ✅ Output: Classification label (`Healthy` or `Unhealthy`)

## 🧪 Digital Image Processing (DIP) Steps

For each image:
1. Resize to 128x128
2. Convert to Grayscale
3. Apply Binary Thresholding
4. Extract and Draw Contours

```python
plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1), plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(1, 4, 2), plt.imshow(gray, cmap='gray'), plt.title('Grayscale')
plt.subplot(1, 4, 3), plt.imshow(thresh, cmap='gray'), plt.title('Threshold')
plt.subplot(1, 4, 4), plt.imshow(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB)), plt.title('Contours')
plt.show()
```

## 📂 Directory Structure

```
DIP PROJECT/
│
├── PlantVillage/
│   ├── Pepper__bell___healthy/
│   └── Pepper__bell___Bacterial_spot/
│
├── leaf_disease_code.py          
├── leaf_disease_model.h5         
├── leaf_disease_model.keras      
├── code-and-output.html          
├── requirements.txt              
├── README.md                     
└── leaf_test.jpg, leaf_test2.jpg 
```

## 🚀 Model Architecture

```text
Input: (128x128x3)

Conv2D (32 filters, 3x3) → ReLU → MaxPooling (2x2)
Conv2D (64 filters, 3x3) → ReLU → MaxPooling (2x2)
Flatten
Dense (64 units) → ReLU
Dropout (0.5)
Dense (2 units) → Softmax
```

## 📊 Accuracy & Loss Plots

Training and validation performance were tracked across epochs. See `leaf_disease_code.py` for graphs.

## 🧪 Testing the Model

```python
test_paths = [r"C:\Users\HP\Desktop\leaf_test.jpg", r"C:\Users\HP\Desktop\leaf_test2.jpg"]
```

## 📦 Installation

```bash
git clone https://github.com/OROT-ENG/leaf-disease-cnn.git
cd leaf-disease-cnn
pip install -r requirements.txt
```

## ✅ Requirements

- Python 3.10
- TensorFlow 2.x
- OpenCV
- Matplotlib
- Scikit-learn
- NumPy

## ✨ Credits

- Dataset: [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Author: [OROT-ENG](https://github.com/OROT-ENG)
