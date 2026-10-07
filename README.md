# ♟️ Chess Piece Classification with Deep Learning

A deep learning computer vision project that automatically identifies chess pieces from images using **TensorFlow/Keras** and **MobileNetV2 transfer learning**.

The project includes both the model training workflow and an interactive **Streamlit web application** where users can upload an image of a chess piece and receive a prediction with its confidence score.

## 🚀 Project Overview

The goal of this project is to build an automated multi-class image classification system capable of recognizing different chess pieces.

The model classifies images into five categories:

* ♕ Queen
* ♖ Rook
* ♗ Bishop
* ♘ Knight
* ♙ Pawn

The training process includes:

1. Exploratory Data Analysis (EDA)
2. Image preprocessing
3. Image resizing and normalization
4. Transfer learning with MobileNetV2
5. Model training
6. Model evaluation and visualization
7. Saving the trained model
8. Deployment with Streamlit

## 🧠 Model Architecture

This project uses **MobileNetV2**, pretrained on ImageNet, as the feature extraction backbone.

The pretrained convolutional base is frozen and a custom classification head is added:

```text
Input Image (224 × 224 × 3)
            │
            ▼
       MobileNetV2
     (ImageNet Weights)
            │
            ▼
 Global Average Pooling
            │
            ▼
     Dense Layer (128)
            │
            ▼
        Dropout (0.5)
            │
            ▼
      Softmax Output
            │
            ▼
  5 Chess Piece Classes
```

The model is compiled using:

* **Optimizer:** Adam
* **Learning Rate:** 0.0001
* **Loss:** Categorical Crossentropy
* **Metric:** Accuracy
* **Epochs:** 15
* **Batch Size:** 32

## 📊 Dataset

The project uses approximately **650 images** across five chess-piece categories.

The images are organized into class-specific folders and loaded using TensorFlow/Keras utilities.

Before training, the images are:

* Resized to **224 × 224 pixels**
* Converted to RGB
* Normalized to a `[0, 1]` range
* Split into training and test/validation sets using stratified sampling

The dataset is split using an **80/20 train-test split**.

## 🔍 Exploratory Data Analysis

The notebook includes several EDA steps to understand the dataset:

* Class distribution analysis
* Sample image visualization
* Random image visualization
* Inspection of individual classes
* Visualization of representative chess-piece images

These steps help verify that the dataset is correctly structured before training the model.

## 🔄 Transfer Learning

Instead of training a convolutional neural network from scratch, this project uses **transfer learning**.

MobileNetV2 was selected with pretrained ImageNet weights. The convolutional base is frozen, allowing the model to use previously learned visual features while training a smaller classification head specifically for chess pieces.

This approach is particularly useful when working with a relatively small image dataset.

## 📈 Training

The model is trained for 15 epochs using the prepared image generators.

Training and validation performance are tracked using:

* Accuracy
* Validation accuracy
* Loss
* Validation loss

The notebook also generates training curves to visualize model performance throughout the training process.

## 💾 Saved Model

After training, the model is saved as:

```text
chess_model.keras
```

The Streamlit application loads this trained model and uses it to classify new images.

## 🖥️ Streamlit Application

The project includes an interactive web application built with **Streamlit**.

Users can:

1. Upload a `.jpg`, `.jpeg`, or `.png` image.
2. The image is converted to RGB.
3. The image is resized to `224 × 224`.
4. Pixel values are normalized.
5. The trained model generates class probabilities.
6. The application displays the predicted chess piece.
7. The confidence score is shown.
8. A probability chart displays the model's predictions for all five classes.

### Application Preview

```text
♟️ Chess Piece Classifier

Upload a photo of a chess piece
              │
              ▼
        [ Uploaded Image ]
              │
              ▼
       Prediction: Queen
       Confidence: XX.X%
              │
              ▼
       Class Probability Chart
```

The Streamlit application is implemented in `chessimages.py`.

## 📁 Project Structure

```text
Chess-Classification/
│
├── ChessClassification.ipynb
├── chessimages.py
├── chess_model.keras
├── README.md
│
└── dataset/
    ├── Queen/
    ├── Rook/
    ├── bishop/
    ├── knight/
    └── pawn/
```

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* MobileNetV2
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* Plotly
* Pillow


## 🎯 Results

The trained model successfully demonstrates that transfer learning can be used to classify chess pieces from images.

The project achieved high classification performance while using a relatively small dataset, demonstrating the potential of pretrained convolutional neural networks for specialized computer vision tasks.


## 👨‍💻 Author

**Furkan Ozdemir**
