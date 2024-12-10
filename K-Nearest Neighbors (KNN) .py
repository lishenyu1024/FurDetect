# -*- coding: utf-8 -*-
"""CS5800_final_project.ipynb

Automatically generated by Colab.
Team Members: Er Zhao, Shenyu Li, Shan Meng, Zifeng Li
Date: Dec 10, 2024
Notes: CS5800 Final Project, K-Nearest Neighbors (KNN) Classifier - Cats & Dogs

Original file is located at
    https://colab.research.google.com/drive/1t2-vIME_ExaTfvexrpaJDg3QjXYH4p8W
"""

#K-Nearest Neighbors (KNN) Classifier - Cats & Dogs

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.4)
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
import time
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

df = pd.read_csv("sample_data/CatsAndDogs.csv")
df = df.dropna()
df.head(5)

fig = plt.figure(figsize=(10, 8))
# Add a 3D subplot to the figure
ax = fig.add_subplot(111, projection='3d')

# Define colors for each class (1: Dog, 0: Cat)
colors = {1: 'blue', 0: 'red'}
# Create a 3D scatter plot
# Map the 'Animal' column to colors, and plot the points with specific sizes and transparency
ax.scatter(
    df['Height'],
    df['Weight'],
    df['Length'],
    c=df['Animal'].map(colors),
    s=100,
    alpha=0.7,
    edgecolor='w'
)
# Label the axes with appropriate names
ax.set_xlabel('Height(cm)')
ax.set_ylabel('Weight(kg)')
ax.set_zlabel('Length(cm)')
ax.set_title('3D Scatter Plot of Dogs and Cats')

# Create custom legend elements for Dog and Cat
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Dog', markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Cat', markerfacecolor='red', markersize=10)
]
ax.legend(handles=legend_elements, title='Animal')

plt.show()

class KNN:
    def __init__(self, k=3, metric="euclidean"):
        self.k = k
        self.metric = metric

    # Euclidean distance (l2 norm)
    def euclidean(self, v1, v2):
        return np.sqrt(np.sum((v1 - v2) ** 2))

    # Manhattan distance (l1 norm)
    def manhattan(self, v1, v2):
        return np.sum(np.abs(v1 - v2))

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    # Get nearest neighbours and distances
    def get_neighbours(self, test_row):
        distances = []
        for (train_row, train_class) in zip(self.X_train, self.y_train):
            if self.metric == 'euclidean':
                dist = self.euclidean(train_row, test_row)
            elif self.metric == 'manhattan':
                dist = self.manhattan(train_row, test_row)
            else:
                raise NameError("Supported metrics are euclidean and manhattan")
            distances.append((dist, train_row, train_class))
        distances.sort(key=lambda x: x[0])
        return distances[:self.k]

    # Predict using KNN
    def predict(self, X_test):
        preds = []
        for test_row in X_test:
            # Find k nearest neighbours
            neighbours = self.get_neighbours(test_row)

            # Predict the majority class using Counter
            neighbour_classes = [n[2] for n in neighbours]
            majority = Counter(neighbour_classes).most_common(1)[0][0]
            preds.append(majority)
        return np.array(preds)

    # Calculate accuracy score
    def score(self, preds, y_test):
        return 100 * (preds == y_test).mean()

    # Plot predictions in 3D, each test point in a separate figure
    def plot_predictions(self, X_test, y_test, preds, feature_names=["Height", "Weight", "Length"]):
        unique_classes = np.unique(self.y_train)
        class_labels = {0: 'Cat', 1: 'Dog'}

        for i, (test_row, pred_class, actual_class) in enumerate(zip(X_test, preds, y_test)):
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            for idx, label in enumerate(unique_classes):
                ax.scatter(
                    self.X_train[self.y_train == label][:, 0],
                    self.X_train[self.y_train == label][:, 1],
                    self.X_train[self.y_train == label][:, 2],
                    c=colors[idx % len(colors)],
                    label=class_labels.get(label, f'Class {label}'),
                    alpha=0.5
                )

            neighbours = self.get_neighbours(test_row)

            for _, point, _ in neighbours:
                ax.plot(
                    [test_row[0], point[0]],
                    [test_row[1], point[1]],
                    [test_row[2], point[2]],
                    color="blue",
                    linestyle="--",
                    linewidth=1
                )

            ax.scatter(
                test_row[0], test_row[1], test_row[2],
                color="yellow",
                marker="^",
                s=200,
                edgecolor='k',
                label="Test Point"
            )


            max_distance = max([n[0] for n in neighbours])
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = test_row[0] + max_distance * np.cos(u) * np.sin(v)
            y = test_row[1] + max_distance * np.sin(u) * np.sin(v)
            z = test_row[2] + max_distance * np.cos(v)
            ax.plot_wireframe(x, y, z, color="blue", alpha=0.1)


            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
            ax.set_zlabel(feature_names[2])
            ax.set_title(f"KNN Prediction {i+1}: Predicted = {class_labels.get(pred_class, pred_class)}, Actual = {class_labels.get(actual_class, actual_class)}")

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), title="Classes and Test Point")

            plt.show()

X = df[['Height', 'Weight', 'Length']].values
y = df['Animal'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

#Initalization
my_knn = KNN(k=3, metric="euclidean")

#Fit model
my_knn.fit(X_train, y_train)

#Prediction
preds = my_knn.predict(X_test)

#check accuracy
accuracy = my_knn.score(preds, y_test)
print(f"Accuracy: {accuracy:.2f}%")

#visiualization
my_knn.plot_predictions(X_test, y_test, preds, feature_names=["Height", "Weight", "Length"])

#standard KNN from sklearn
# Initialize the KNN classifier with k=3 (3 nearest neighbors)
knn_model = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
# Fit the KNN model using the training data (X_train, y_train)
knn_model.fit(X_train,y_train)
# Predict the class labels for the test set (X_test)
preds = knn_model.predict(X_test)

# Evaluate the model by calculating the accuracy score on the test data
accuracy = knn_model.score(X_test, y_test)
print(f"{round(accuracy*100,2)}%")