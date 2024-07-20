import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('captured_photo.png')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize HOG descriptor
hog = cv2.HOGDescriptor()

# Compute HOG features
hog.compute(gray)

# Get HOG descriptor parameters
winSize = hog.winSize
blockSize = hog.blockSize
blockStride = hog.blockStride
cellSize = hog.cellSize
nbins = hog.nbins

# Calculate number of cells in each direction
cells_x = winSize[0] // cellSize[0]
cells_y = winSize[1] // cellSize[1]

# Get HOG features
hog_features = hog.getDescriptor()

# Reshape HOG features into 2D array representing cells and their histograms
hog_features_reshaped = hog_features.reshape(cells_y, cells_x, nbins)

# Visualize HOG features by plotting histograms of gradient orientations for each cell
fig, axs = plt.subplots(cells_y, cells_x, figsize=(10, 10))

for i in range(cells_y):
    for j in range(cells_x):
        histogram = hog_features_reshaped[i, j, :]
        axs[i, j].bar(np.arange(nbins), histogram)
        axs[i, j].set_xticks(np.arange(nbins))
        axs[i, j].set_title(f'Cell ({j}, {i})')

plt.tight_layout()
plt.show()
