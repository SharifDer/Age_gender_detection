# UTKFace Age, Gender, and Race Classification

## Overview
This project processes the UTKFace dataset to build a deep learning model for age prediction. It involves data preprocessing, image normalization, and training a Convolutional Neural Network (CNN) to estimate age from facial images.

## Dataset
This project uses the UTKFace dataset, available at: [UTKFace Dataset on Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new).

### Dataset Information
- **Images**: Facial images labeled with age, gender, and race.
- **Age**: Integer values from 0 to 116.
- **Gender**: 0 (Male), 1 (Female).
- **Race**: 0 (White), 1 (Black), 2 (Asian), 3 (Indian), 4 (Others).

## Project Workflow
1. **Data Preprocessing**
    - Extracts age, gender, and race information from image filenames.
    - Filters out specific race categories.
    - Saves processed data as a CSV file (`Images.csv`).

2. **Image Processing**
    - Reads and resizes images to 168x168 pixels.
    - Normalizes pixel values to a [0,1] range.
    - Saves processed dataset as `Images.csv` with corresponding labels.

3. **Train-Test Split**
    - Splits image data into training and testing sets for model training.

4. **Model Training**
    - Implements a CNN model using TensorFlow/Keras.
    - Uses Conv2D, MaxPooling2D, Flatten, Dense, and Dropout layers.
    - Optimizes using Adam and mean squared error (MSE) loss function.
    - Trains the model to predict age from facial images.

5. **Evaluation and Prediction**
    - Evaluates model performance on the test set.
    - Uses trained model to predict age from new images.
    - Example inference is performed on an external image (`Sharif.png`).

## Requirements
Install the required Python packages before running the script:
```bash
pip install numpy pandas matplotlib tensorflow pillow scikit-learn opencv-python
```

## Running the Project
1. Place the `UTKFace` dataset in the working directory.
2. Run the script to preprocess data and train the model.
3. Evaluate model performance and test with new images.

## Output
- Processed dataset saved as `Images.csv`.
- Trained model performance metrics.
- Visualization of dataset distributions and predictions.

## Author
Sharif

