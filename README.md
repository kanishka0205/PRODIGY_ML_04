# PRODIGY_ML_04
Hello Everyone! This is my Internship task Repository where I created a Hand Gesture Recognition model using Python , keras and tensorflow
The dataset for the code is given in https://www.kaggle.com/datasets/roobansappani/hand-gesture-recognition
for your reference

# ✋ Hand Gesture Recognition using Deep Learning

This project focuses on developing a **Hand Gesture Recognition** model that can accurately identify and classify different hand gestures from **image**. It aims to enable **intuitive human-computer interaction** and **gesture-based control systems**, making systems smarter and more accessible.

---

## 📌 Project Objective

To build a robust machine learning model that can:

- Recognize hand gestures from visual input (images or video frames)
- Classify gestures into predefined categories (e.g., thumbs up, peace, stop)
- Enable interaction with systems using hand movements (HCI)

---

## 🗃️ Dataset

- **Source**: https://www.kaggle.com/datasets/roobansappani/hand-gesture-recognition
- **Data Type**: Images or video frames containing hand gestures
- **Classes**: e.g., call_me , fingers_crossed , okay , paper , peace , rock , rock_on , scissor , thumbs ,up
- **Format**: PNG / JPG images or frame sequences from video

---

## 🛠️ Technologies Used

- Python 🐍
- OpenCV 🎥
- NumPy & Pandas 🔢
- TensorFlow / Keras 🤖
- Matplotlib / Seaborn 📊
- MediaPipe (optional) ✋
- Jupyter Notebook 📒

---

## 🔍 Data Preprocessing

- Converted videos to individual frames (if video data used)
- Resized all gesture images to a fixed size (e.g., 64x64 or 128x128)
- Normalized pixel values
- One-hot encoded gesture labels
- Augmented dataset with rotation, flipping, zoom, etc.

---

## 🚀 Model Workflow

1. **Data Preparation**
   - Load and preprocess image/video gesture data
   - Split into training and testing sets

2. **Model Building**
   - Used a Convolutional Neural Network (CNN) for feature extraction
   - Applied Dense layers with Softmax for classification

3. **Training**
   - Trained using cross-entropy loss and Adam optimizer
   - Validated model on unseen gestures

4. **Evaluation**
   - Accuracy, Precision, Recall, and F1 Score
   - Confusion matrix for class-wise performance

---

## 📈 Results

- **Model Accuracy**: 99.98% 
- **Training Loss**: Converged after 10 epochs
- **Confusion Matrix**: Demonstrates reliable classification performance

> Example Insight: The model successfully differentiates between visually similar gestures when background is uniform and lighting is consistent.

---

## 📚 Learning Outcomes

- Learned to preprocess visual gesture data
- Built and trained CNN models for classification
- Gained insights into gesture recognition for human-computer interaction

---

## 💡 Future Enhancements

- Integrate **MediaPipe** or **OpenCV Hand Detection** for real-time recognition
- Expand to **dynamic gestures** using LSTM on video sequences
- Deploy the model in a GUI or embedded application

---
