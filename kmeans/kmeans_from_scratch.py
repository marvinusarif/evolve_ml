import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import math
import sys
from week4.minist_dataset import generate_training_test
from week4.dunn_index import compute_dunn_index

def initialize_clusters(points, k):
    """Initializes clusters as k randomly selected points from points."""
    return points[np.random.randint(points.shape[0], size=k)]


# Function for calculating the distance between centroids
def get_distance(centroid, point):
    """Returns the distance the centroid is from each data point in points."""
    distance = 0
    for i in range(len(point)):
        distance += math.pow((centroid[i] - point[i]), 2)
    return math.sqrt(distance)


def get_distances(centroid, points):
    """Returns the distance the centroid is from each data point in points."""
    distances = []
    for i in range(len(points)):
        distances.append(get_distance(centroid, points[i]))
    return distances


def generate_train_test_dataset():
    # Generate dataset
    X, y = make_blobs(centers=3, n_samples=500, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    return X_train, y_train, X_test, y_test


def visualize_data(X):
    # Visualize
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')


def train(X, k):
    maxiter = 50

    # Initialize our centroids by picking random data points
    centroids = initialize_clusters(X, k)

    # Initialize the vectors in which we will store the
    # assigned classes of each data point and the
    # calculated distances from each centroid
    classes = np.zeros(X.shape[0], dtype=np.float64)
    distances = np.zeros([X.shape[0], k], dtype=np.float64)

    # Loop for the maximum number of iterations
    for i in range(maxiter):

        # Assign all points to the nearest centroid
        for i, c in enumerate(centroids):
            distances[:, i] = get_distances(c, X)

        # Determine class membership of each point
        # by picking the closest centroid
        classes = np.argmin(distances, axis=1)

        # Update centroid location using the newly
        # assigned data point classes
        for c in range(k):
            centroids[c] = np.mean(X[classes == c], 0)

    return classes, centroids


def predict(test_data, centroids):
    classes = np.zeros(test_data.shape[0], dtype=np.float64)
    distances = np.zeros([test_data.shape[0], k], dtype=np.float64)

    # Assign all points to the nearest centroid
    for i, c in enumerate(centroids):
        distances[:, i] = get_distances(c, test_data)

    # Determine class membership of each point
    # by picking the closest centroid
    classes = np.argmin(distances, axis=1)

    return classes


def plot_result(X, centroids, classes):
    group_colors = ['skyblue', 'coral', 'lightgreen']
    colors = [group_colors[j] for j in classes]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(X[:, 0], X[:, 1], color=colors, alpha=0.5)
    ax.scatter(centroids[:, 0], centroids[:, 1], color=['blue', 'darkred', 'green'], marker='o', lw=2)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$');



x_train, y_train, x_test, y_test=generate_train_test_dataset()
k=3
y_train_hat, centroids=train(x_train, k)
y_test_hat=predict(x_test, centroids)
predicted_train_centroids=predict(x_train, centroids)
di=compute_dunn_index(get_distance, predicted_train_centroids,x_train, centroids)
print('dunn index',di)
visualize_data(x_train)
plot_result(x_train, centroids, y_train_hat)
plot_result(x_test, centroids, y_test_hat)


plt.show()



