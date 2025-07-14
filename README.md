# Dog vs Cat Image Classifier

This project is a simple image classification tool that uses PyTorch and a pre-trained ResNet18 model to classify images of dogs and cats. It was created as part of my application for the Research Assistant position at The Hospital for Sick Children and demonstrates my ability to build and train deep learning models using PyTorch.

## Features

- Binary classification of cat vs. dog images
- Utilizes a pre-trained ResNet18 model from torchvision
- Fine-tuned on a small custom dataset
- Reports training loss per epoch

## Requirements

- Python 3
- PyTorch
- torchvision

Install dependencies:


## Running the Project

Run the classifier using: python3 dog_vs_cat_classifier.py

## Sample output:

Epoch 1 Loss: 0.9974
Epoch 2 Loss: 0.0745
Epoch 3 Loss: 0.0170

## Notes
- The dataset is small and only for demonstration purposes.
- The model trains for 3 epochs by default.
- It can be improved by adding more labeled images under data/train and data/val.
