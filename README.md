# FurDetect：K-Nearest Neighbors (KNN) Classifier - Cats & Dogs

This project implements a custom K-Nearest Neighbors (KNN) classifier to classify animals (Cats and Dogs) based on their physical attributes: height, weight, and length. The project includes custom KNN logic and a comparison with the standard implementation from `scikit-learn`.

## Features
- Implements KNN from scratch with support for Euclidean and Manhattan distance metrics.
- Visualizes the dataset and predictions in 3D scatter plots.
- Evaluates the accuracy of the custom implementation and compares it with the standard `scikit-learn` KNN.
- Provides a step-by-step pipeline for data preprocessing, training, testing, and evaluation.

## Dataset
The dataset used in this project can be downloaded from Kaggle:

[CatsAndDogs Dataset](https://www.kaggle.com/datasets/scarb7/catsanddogs-dummy)

The dataset contains the following features:
- **Height (cm)**: The height of the animal.
- **Weight (kg)**: The weight of the animal.
- **Length (cm)**: The length of the animal.
- **Animal**: Target label (1 for Dog, 0 for Cat).

Example dataset preview:
| Height | Weight | Length | Animal |
|--------|--------|--------|--------|
| 25     | 4      | 30     | 0      |
| 35     | 7      | 42     | 1      |

## Installation
To run this project, you need Python and the following libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Install the dependencies using pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn


Run your project:
```bash
python YOUR_name_project.py

