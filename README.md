# Chest X-ray Prediction Model

This project uses a Convolutional Neural Network (CNN) to classify chest X-ray images into two categories: **Normal** and **Pneumonia**. The model was trained on a dataset containing labeled chest X-ray images, and it predicts whether a given image is normal or shows signs of pneumonia.

## Dataset

The dataset used for this project consists of chest X-ray images, with two classes:
- **Normal**: Healthy lungs.
- **Pneumonia**: Pneumonia-affected lungs.

You can find the dataset [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

## Installation

To run this project, you need Python 3.x installed along with the required libraries.

### 1. Clone the repository:
```bash
git clone https://github.com/evans25575/CHEST-X_RAY-PREDICTION-MODEL.git
cd CHEST-X_RAY-PREDICTION-MODEL


2. Install dependencies:
First, make sure you have pip installed, then run the following:
pip install -r requirements.txt

3. Dependencies
TensorFlow (for training the model)
Keras (for building neural networks)
Matplotlib (for plotting)
Pandas (for data manipulation)
NumPy (for numerical operations)
OpenCV (for image processing)
Scikit-learn (for evaluation)


##Usage
Training the Model

##To train the model, run the following in your terminal:
python train_model.py
This script will load the dataset, preprocess the images, train the model, and save it to a file.

##Predicting with the Model

Once the model is trained, you can use the predict script to predict whether a given chest X-ray image shows signs of pneumonia.

Example:
python predict_image.py --image path_to_image
This will output whether the image is classified as Normal or Pneumonia.

##Model Architecture
The model is built using a Convolutional Neural Network (CNN) with the following layers:

Conv2D layers for feature extraction
MaxPooling2D layers to reduce spatial dimensions
Dense layers for classification
Dropout layers for regularization
The model is trained using the Adam optimizer and binary crossentropy loss for binary classification.

Evaluation
The model's performance can be evaluated using accuracy, loss, and other metrics such as precision, recall, and F1 score. During training, the accuracy is tracked for both training and validation sets.

Results
The model achieves an accuracy of over 95% on the test data, making it effective for detecting pneumonia in chest X-ray images.

Future Work
Data Augmentation: Enhance the dataset by applying techniques like rotation, flipping, and zooming to generate more diverse training examples.
Model Improvement: Experiment with more advanced architectures like ResNet or VGG for potentially better accuracy.
Deployment: Deploy the model as a web service where users can upload X-ray images and get predictions.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
The dataset used in this project is from Kaggle: Chest X-ray Images (Pneumonia).
Special thanks to the creators of TensorFlow and Keras for providing excellent tools for deep learning.



