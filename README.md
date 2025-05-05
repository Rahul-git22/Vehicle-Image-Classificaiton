# Vehicle Classification using Convolutional Neural Networks (CNN)

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify different types of vehicles from images. It demonstrates building a standard CNN architecture, handling image data using `ImageDataGenerator`, applying data augmentation and regularization techniques, and leveraging transfer learning for improved performance.

## Problem Addressed

The project tackles the task of automatically identifying and categorizing vehicles in images. This is useful in various applications such as:

*   Traffic monitoring and analysis
*   Automated parking systems
*   Vehicle inventory management
*   Assisting in insurance assessments

Automating this classification process significantly reduces the need for manual labeling, improves efficiency, and allows for large-scale analysis of visual data.

## Key Features

*   Builds a CNN model using TensorFlow/Keras for image classification.
*   Loads image data directly from a directory structure with subfolders representing classes.
*   Includes image preprocessing (resizing, rescaling).
*   Applies data augmentation techniques during training to improve model generalization.
*   Uses regularization (Dropout, Early Stopping) to prevent overfitting.
*   Demonstrates the use of Transfer Learning with a pre-trained MobileNetV2 model.
*   Trains the CNN model.
*   Saves the trained model for later use.
*   Provides functionality to load the saved model and make predictions on new images.

## Technologies Used

*   Python
*   TensorFlow / Keras
*   NumPy
*   Pillow (PIL - used by Keras for image handling)
*   Pre-trained Model: MobileNetV2 (for Transfer Learning)


## Challenges Faced and Solutions

*   **Overfitting:** The model initially trained from scratch easily memorized the training data but performed poorly on unseen data. Solutions implemented:
    *   **Data Augmentation:** Artificially expanding the training set variability.
    *   **Dropout:** Regularizing layers during training.
    *   **Early Stopping:** Stopping training based on validation performance.
*   **Limited Data / Low Learning Capacity:** Even with regularization, training accuracy from scratch was modest. Solution:
    *   **Transfer Learning:** Leveraging a powerful pre-trained model (MobileNetV2) to utilize features learned from a vast dataset, providing a much stronger starting point.

## Future Enhancements

Possible areas for further development include:

*   **Hyperparameter Tuning:** Conducting a more systematic search for optimal training configurations.
*   **Exploring Other Architectures:** Experimenting with different pre-trained models (ResNet, EfficientNet, VGG, etc.) or slightly modifying the current architecture.
*   **Fine-tuning:** Unfreezing and training some layers of the pre-trained base model with a very low learning rate after initial training.
*   **Expanding Dataset:** Incorporating more images and potentially more granular vehicle categories.
*   **Comparison:** Benchmarking different models and techniques rigorously.


---

