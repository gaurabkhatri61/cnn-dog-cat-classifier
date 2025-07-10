# 🐶🐱 Pet Image Classifier (CNN-based)

A deep learning project for classifying pet images (dogs vs. cats) using Convolutional Neural Networks (CNN) with TensorFlow/Keras. This repository covers both model training and deployment.

---

## 📁 Project Structure

```
cnn-dog&cat-classifier/
├── model/                   # Saved trained model(s)
├── PetImages/               # Dataset: Dog/ and Cat/ folders
├── .gitignore
├── classify-training.ipynb  # Notebook for training
├── classify-deploy.ipynb    # Notebook for inference/deployment
├── image.webp               # Sample image for prediction
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/cnn-dog-cat-classifier.git
cd cnn-dog-cat-classifier
```

### 2. Install Requirements

```bash
pip install tensorflow numpy matplotlib opencv-python pillow
# Optional: for running notebooks
pip install jupyter
```

---

## 🧠 Training the Model

1. **Prepare the Dataset:**  
   Place your images in the `PetImages/` directory, organized as follows:
   ```
   PetImages/
   ├── Dog/
   │   ├── dog1.jpg
   │   └── ...
   └── Cat/
       ├── cat1.jpg
       └── ...
   ```
   > **Tip:** Remove or skip corrupted images for smooth training.

2. **Start Training:**  
   Open the training notebook:
   ```bash
   jupyter notebook classify-training.ipynb
   ```
   - Loads and preprocesses the dataset
   - Builds a CNN model (custom or with Transfer Learning)
   - Trains and evaluates the model
   - Saves the trained model to `model/`

---

## 🤖 Deployment & Prediction

1. **Open the Deployment Notebook:**
   ```bash
   jupyter notebook classify-deploy.ipynb
   ```
2. **Predict:**  
   - Loads the saved model from `model/`
   - Preprocesses your input image (e.g., `image.webp`)
   - Predicts and displays the class label (Dog or Cat)

---

## 🧩 Model Details

- **Architecture:** Custom CNN or Transfer Learning (e.g., MobileNetV2)
- **Input Size:** 128x128 or 224x224 pixels
- **Loss:** Categorical Crossentropy
- **Optimizer:** Adam
- **Output:** Softmax class probabilities

---

## 📈 Evaluation

- Accuracy & loss curves
- Model summary and metrics
- Example predictions

---

## 🚧 Future Improvements

- [ ] Web interface (Streamlit or Flask)
- [ ] Grad-CAM visual explanations
- [ ] Data augmentation
- [ ] CI/CD and automated testing

---

## 📄 License

MIT License

---

> 🔗 *Built for educational and prototyping purposes.*