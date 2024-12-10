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
```
Run your project:
```bash
python YOUR_name_project.py
```

## Data Visualization

To better understand the dataset distribution, we created a 3D scatter plot showing the relationships between height, weight, and length for each animal. The points are color-coded to indicate their class:
- **Blue points** represent dogs.
- **Red points** represent cats.

The plot helps visualize the separability of the two classes in the feature space.

![3D Scatter Plot of Dogs and Cats](https://github.com/lishenyu1024/FurDetect/blob/cd37ea1de39d8b57ab5a4f9f1333e0f0645f99b0/image/3D%20Scatter%20Plot%20of%20Dogs%20and%20Cats.png))

### KNN Model Prediction Visualization

The KNN model predicts the class of a test point by calculating the distance between the test data and all training data points based on the selected metric (e.g., Euclidean distance). Below are visualizations that demonstrate this process:

1. **Test Point and Neighbors**:  
   In the 3D scatter plots below, each yellow triangle represents a test point, while the red and blue points represent the training data for Cats and Dogs, respectively.

2. **Nearest Neighbor Selection**:  
   Each test point is connected to its `k` nearest neighbors (lines are not shown in this image but are processed during prediction). These neighbors determine the predicted class of the test point.

#### Image: Test Point Location in Feature Space
The yellow triangle marks the location of the test point relative to the training data. This visualization helps understand the spatial relationships and distances used in the KNN algorithm.

![Test Point Visualization ](https://github.com/lishenyu1024/FurDetect/blob/cd37ea1de39d8b57ab5a4f9f1333e0f0645f99b0/image/Test%20Point%20Location%20in%20Feature%20Space.png)

#### Image: Test Point Classification
The second plot highlights multiple test points and their relationships with the training data classes. The yellow triangles show test points, with classification determined by their nearest neighbors.

![Test Point Visualization ](https://github.com/lishenyu1024/FurDetect/blob/cd37ea1de39d8b57ab5a4f9f1333e0f0645f99b0/image/Test%20Point%20Visualization%20-%20Right.png)

### KNN Prediction Process Visualization

The following visualizations demonstrate the step-by-step prediction process of the KNN algorithm for test data points:

1. **Prediction Process**:  
   - The KNN algorithm calculates the distance between each test data point and all training data points.
   - The nearest `k` neighbors are identified, and their classes are used to predict the label of the test point.
   - The predicted label is compared with the true label for evaluation.

2. **Visualization Details**:  
   - **Yellow triangles** represent the test points.
   - **Blue and red points** represent the training data (Dogs and Cats, respectively).
   - The surrounding wireframe sphere indicates the range within which the nearest neighbors are located.

#### Left Image: KNN Prediction 1
This image shows the first test point classified as a **Dog**, with the actual label also being **Dog**. The predicted label matches the true label.

![KNN Prediction 1](https://github.com/lishenyu1024/FurDetect/blob/cd37ea1de39d8b57ab5a4f9f1333e0f0645f99b0/image/knn_prediction_1.png)

#### Middle Image: KNN Prediction 2
In this image, another test point is correctly classified as a **Dog**. The wireframe sphere highlights the range of neighbors contributing to this prediction.

![KNN Prediction 2](https://github.com/lishenyu1024/FurDetect/blob/cd37ea1de39d8b57ab5a4f9f1333e0f0645f99b0/image/knn_prediction_2.png)

#### Right Image: KNN Prediction 6
Here, the test point is classified as a **Cat**, but the actual label is **Dog**. This demonstrates a misclassification by the KNN algorithm.

![KNN Prediction 6](https://github.com/lishenyu1024/FurDetect/blob/cd37ea1de39d8b57ab5a4f9f1333e0f0645f99b0/image/knn_prediction_6.png)

## **Conclusion**

This project successfully implemented a custom K-Nearest Neighbors (KNN) algorithm to classify cats and dogs based on their physical features. Key takeaways:
- The KNN algorithm is simple, efficient, and ideal for small datasets as it does not require complex model training.
- Our custom implementation achieved accuracy comparable to the Scikit-learn library, validating the effectiveness of the code and algorithm design.
- However, challenges like handling blurred boundaries in the dataset and optimizing the model's performance still remain.

---

## **Challenges**

During the development and testing of the KNN model, we faced the following key challenges:
- **Choosing the optimal `k` value**: Balancing between underfitting (k too large) and overfitting (k too small) was critical.
- **Distance metrics**: While Euclidean distance performed well, other metrics like Manhattan or cosine distance might be better for specific cases.
- **Dataset boundary issues**: Some cat and dog samples had unclear separations, requiring additional data preprocessing or feature extraction techniques.

Overall, this project provided insights into the strengths and limitations of the KNN algorithm and served as a foundation for future improvements.








