# Image Classification with TensorFlow

This project demonstrates the use of TensorFlow for training a binary image classification model. The model is designed to classify images based on labels provided in a CSV file. It uses convolutional neural networks (CNNs) to process and classify images and is structured to handle the dataset efficiently.

# Prerequisites
Make sure to install the required dependencies before running the script:


pip install pandas numpy tensorflow scikit-learn matplotlib

# File Structure
The following files are expected in the project directory:

labels.csv: A CSV file containing image file paths and their corresponding labels.
model.weights.h5: Model weights file (saved after training).
model.h5: Complete model file (including architecture and weights).

# Dataset
The dataset is expected to be organized with a column plot_path containing paths to image files, and a column label containing the binary classification labels. The images should be stored in a directory accessible via these paths.

Example of labels.csv:

plot_path	label
images/img1.jpg	0
images/img2.jpg	1
images/img3.jpg	0
...	...

# Parameters
The following parameters are used for training:

img_size: Tuple defining the size to which images will be resized (default is 128x128).
batch_size: Number of images per batch during training (default is 32).
epochs: The number of times the entire dataset is passed through the model during training (set to 10).

# Model Architecture
The model architecture consists of three convolutional layers, each followed by a max-pooling layer. After flattening the output of the final convolutional layer, the model has a fully connected layer with dropout for regularization and a final output layer with a sigmoid activation for binary classification.

model = models.Sequential([
    Input(shape=(img_size[0], img_size[1], 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Usage
Prepare your dataset: Ensure that labels.csv contains the paths to your images and their corresponding labels.

Run the script: After setting up the dataset, run the script using:

python model.py
This will:

Load the dataset and preprocess the images.
Split the data into training, validation, and test sets.
Train the model using the training dataset and validate on the validation dataset.
Save the trained model and its weights in model.h5 and model.weights.h5.
Model Training and Evaluation
The model will be trained using the Adam optimizer with binary cross-entropy loss function.
The training progress, including loss and accuracy, will be printed during training.
Saving and Loading the Model
After training, the model is saved to disk:

model.weights.h5: Contains the trained weights.
model.h5: Contains both the model architecture and weights.
You can load the model using the following code:

from tensorflow.keras.models import load_model
model = load_model("model.h5")